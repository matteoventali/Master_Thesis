"""Continuous task predicates and their grid over-approximations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


OBSERVATION_X_BOUNDS = (-1.0, 1.0)
OBSERVATION_Y_BOUNDS = (0.0, 1.5)


@dataclass(frozen=True)
class CircularRegion:
    """A circular proposition region in LunarLander observation coordinates."""

    center_x: float
    center_y: float
    radius: float

    def __post_init__(self) -> None:
        values = (self.center_x, self.center_y, self.radius)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("Region center coordinates and radius must be numbers")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Region center coordinates and radius must be finite")
        if self.radius <= 0:
            raise ValueError("Region radius must be greater than zero")
        if not OBSERVATION_X_BOUNDS[0] <= self.center_x <= OBSERVATION_X_BOUNDS[1]:
            raise ValueError(f"Region center x={self.center_x} is outside {OBSERVATION_X_BOUNDS}")
        if not OBSERVATION_Y_BOUNDS[0] <= self.center_y <= OBSERVATION_Y_BOUNDS[1]:
            raise ValueError(f"Region center y={self.center_y} is outside {OBSERVATION_Y_BOUNDS}")

    def contains(self, x: float, y: float) -> bool:
        """Return whether a continuous observation point belongs to the region."""
        dx = float(x) - self.center_x
        dy = float(y) - self.center_y
        return dx * dx + dy * dy <= self.radius * self.radius

    def intersects_cell(self, bounds: Sequence[float]) -> bool:
        """Return whether this predicate is true anywhere inside a grid cell."""
        return circle_intersects_cell(self, bounds)

    @classmethod
    def from_dict(cls, data: Mapping[str, object], name: str = "region") -> "CircularRegion":
        if not isinstance(data, Mapping):
            raise ValueError(f"{name!r} must be a JSON object")
        center = data.get("center")
        if isinstance(center, (str, bytes)) or not isinstance(center, Sequence) or len(center) != 2:
            raise ValueError(f"{name!r}.center must contain exactly [x, y]")
        if "radius" not in data:
            raise ValueError(f"{name!r} must define radius")
        return cls(center_x=center[0], center_y=center[1], radius=data["radius"])

    def as_dict(self) -> dict[str, object]:
        return {"center": [self.center_x, self.center_y], "radius": self.radius}


@dataclass(frozen=True)
class BoxPredicate:
    """An axis-aligned rectangular proposition in observation space."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("Box bounds must be numbers")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Box bounds must be finite")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("Box minimum bounds must not exceed maximum bounds")

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= float(x) <= self.x_max and self.y_min <= float(y) <= self.y_max

    def intersects_cell(self, bounds: Sequence[float]) -> bool:
        if len(bounds) != 4:
            raise ValueError("Cell bounds must be (x_min, x_max, y_min, y_max)")
        x_min, x_max, y_min, y_max = bounds
        return (
            x_max >= self.x_min
            and x_min <= self.x_max
            and y_max >= self.y_min
            and y_min <= self.y_max
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "box",
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


@dataclass(frozen=True)
class HalfPlanePredicate:
    """An axis-aligned threshold proposition in observation space."""

    axis: str
    operator: str
    threshold: float

    def __post_init__(self) -> None:
        if self.axis not in {"x", "y"}:
            raise ValueError("A half-plane axis must be x or y")
        if self.operator not in {"<", "<=", ">", ">="}:
            raise ValueError("A half-plane operator must be <, <=, >, or >=")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("A half-plane threshold must be finite")

    def contains(self, x: float, y: float) -> bool:
        coordinate = float(x) if self.axis == "x" else float(y)
        return {
            "<": coordinate < self.threshold,
            "<=": coordinate <= self.threshold,
            ">": coordinate > self.threshold,
            ">=": coordinate >= self.threshold,
        }[self.operator]

    def intersects_cell(self, bounds: Sequence[float]) -> bool:
        if len(bounds) != 4:
            raise ValueError("Cell bounds must be (x_min, x_max, y_min, y_max)")
        x_min, x_max, y_min, y_max = bounds
        cell_min, cell_max = (x_min, x_max) if self.axis == "x" else (y_min, y_max)
        return {
            "<": cell_min < self.threshold,
            "<=": cell_min <= self.threshold,
            ">": cell_max > self.threshold,
            ">=": cell_max >= self.threshold,
        }[self.operator]

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "half_plane",
            "axis": self.axis,
            "operator": self.operator,
            "threshold": self.threshold,
        }


def load_regions(raw_regions: Mapping[str, object]) -> dict[str, CircularRegion]:
    """Parse and validate a proposition-to-region configuration mapping."""
    if not isinstance(raw_regions, Mapping) or not raw_regions:
        raise ValueError("trajectory.json must define a non-empty 'regions' object")
    regions = {}
    for name, data in raw_regions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Region proposition names must be non-empty strings")
        regions[name] = CircularRegion.from_dict(data, name=name)
    return regions


def load_spatial_predicates(
    raw_predicates: Mapping[str, object] | None,
    regions: Mapping[str, CircularRegion],
) -> dict[str, object]:
    """Parse atomic circle, box, and half-plane spatial predicates."""
    if raw_predicates is None:
        return {}
    if not isinstance(raw_predicates, Mapping):
        raise ValueError("trajectory.json predicates must be a JSON object")

    predicates = {}
    for name, data in raw_predicates.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Predicate names must be non-empty strings")
        if name in regions:
            raise ValueError(f"Predicate {name!r} duplicates a region proposition")
        if not isinstance(data, Mapping):
            raise ValueError(f"Predicate {name!r} must be a JSON object")
        predicate_type = data.get("type")
        if predicate_type not in {"circle", "box", "half_plane"}:
            raise ValueError(f"Predicate {name!r}.type must be circle, box, or half_plane")

        if predicate_type == "circle":
            unknown_fields = sorted(set(data) - {"type", "center", "radius"})
            if unknown_fields:
                raise ValueError(f"Predicate {name!r} contains unknown fields: {unknown_fields}")
            predicates[name] = CircularRegion.from_dict(data, name=name)
            continue

        if predicate_type == "box":
            box_fields = {"x_min", "x_max", "y_min", "y_max"}
            unknown_fields = sorted(set(data) - ({"type"} | box_fields))
            if unknown_fields:
                raise ValueError(f"Predicate {name!r} contains unknown fields: {unknown_fields}")
            missing_fields = sorted(box_fields - set(data))
            if missing_fields:
                raise ValueError(f"Predicate {name!r} is missing fields: {missing_fields}")
            predicates[name] = BoxPredicate(**{field: data[field] for field in box_fields})
            continue

        unknown_fields = sorted(set(data) - {"type", "axis", "operator", "threshold"})
        if unknown_fields:
            raise ValueError(f"Predicate {name!r} contains unknown fields: {unknown_fields}")
        axis = data.get("axis")
        operator = data.get("operator")
        if axis not in {"x", "y"}:
            raise ValueError(f"Predicate {name!r}.axis must be x or y")
        if operator not in {"<", "<=", ">", ">="}:
            raise ValueError(f"Predicate {name!r}.operator must be <, <=, >, or >=")
        threshold = data.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
            raise ValueError(f"Predicate {name!r}.threshold must be a finite number")
        predicates[name] = HalfPlanePredicate(
            axis=axis,
            operator=operator,
            threshold=float(threshold),
        )
    return predicates


def load_task_propositions(raw_regions, raw_predicates=None):
    """Load circular regions and derived predicates into one proposition map."""
    regions = load_regions(raw_regions)
    predicates = load_spatial_predicates(raw_predicates, regions)
    return regions, predicates, {**regions, **predicates}


def truth_assignment_from_observation(
    regions: Mapping[str, object], observation: Sequence[float]
) -> dict[str, bool]:
    """Evaluate task propositions exactly on continuous environment coordinates."""
    if len(observation) < 2:
        raise ValueError("An environment observation must contain x and y")
    x, y = float(observation[0]), float(observation[1])
    return {name: region.contains(x, y) for name, region in regions.items()}


def grid_cell_bounds(x: int, y: int, width: int, height: int) -> tuple[float, float, float, float]:
    """Return continuous observation bounds for one uniform abstract cell."""
    if width <= 0 or height <= 0:
        raise ValueError("Grid dimensions must be positive")
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"Cell ({x}, {y}) is outside the {width}x{height} grid")
    x_step = (OBSERVATION_X_BOUNDS[1] - OBSERVATION_X_BOUNDS[0]) / width
    y_step = (OBSERVATION_Y_BOUNDS[1] - OBSERVATION_Y_BOUNDS[0]) / height
    x_min = OBSERVATION_X_BOUNDS[0] + x * x_step
    y_min = OBSERVATION_Y_BOUNDS[0] + y * y_step
    return x_min, x_min + x_step, y_min, y_min + y_step


def circle_intersects_cell(region: CircularRegion, bounds: Sequence[float]) -> bool:
    """Return whether a circle intersects an axis-aligned cell rectangle."""
    if len(bounds) != 4:
        raise ValueError("Cell bounds must be (x_min, x_max, y_min, y_max)")
    x_min, x_max, y_min, y_max = bounds
    closest_x = min(max(region.center_x, x_min), x_max)
    closest_y = min(max(region.center_y, y_min), y_max)
    dx = closest_x - region.center_x
    dy = closest_y - region.center_y
    return dx * dx + dy * dy <= region.radius * region.radius


def rasterize_regions(
    regions: Mapping[str, object], width: int, height: int
) -> dict[str, frozenset[tuple[int, int]]]:
    """Over-approximate every region with all intersected cells of one grid."""
    rasterized = {}
    for name, region in regions.items():
        cells = frozenset(
            (x, y)
            for x in range(width)
            for y in range(height)
            if region.intersects_cell(grid_cell_bounds(x, y, width, height))
        )
        if not cells and isinstance(region, CircularRegion):
            raise ValueError(f"Region {name!r} does not intersect the {width}x{height} grid")
        rasterized[name] = cells
    return rasterized


def truth_assignment_from_cell(
    region_cells: Mapping[str, frozenset[tuple[int, int]]], x: int, y: int
) -> dict[str, bool]:
    """Evaluate abstract propositions using the grid over-approximation."""
    return {name: (x, y) in cells for name, cells in region_cells.items()}
