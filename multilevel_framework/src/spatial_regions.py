"""Continuous circular task regions and their grid over-approximations."""

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
class HalfPlanePredicate:
    """A generic above/below/left/right proposition in observation space."""

    relation: str
    threshold: float
    reference: str | None = None
    boundary: str = "edge"
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.relation not in {"above", "below", "left_of", "right_of"}:
            raise ValueError("A relative-position relation must be above, below, left_of, or right_of")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("A relative-position threshold must be finite")

    def contains(self, x: float, y: float) -> bool:
        coordinate = float(y) if self.relation in {"above", "below"} else float(x)
        if self.relation in {"above", "right_of"}:
            return coordinate > self.threshold
        return coordinate < self.threshold

    def intersects_cell(self, bounds: Sequence[float]) -> bool:
        if len(bounds) != 4:
            raise ValueError("Cell bounds must be (x_min, x_max, y_min, y_max)")
        x_min, x_max, y_min, y_max = bounds
        if self.relation == "above":
            return y_max > self.threshold
        if self.relation == "below":
            return y_min < self.threshold
        if self.relation == "right_of":
            return x_max > self.threshold
        return x_min < self.threshold

    def as_dict(self) -> dict[str, object]:
        if self.reference is not None:
            return {"type": "relative_position", "relation": self.relation, "reference": self.reference, "boundary": self.boundary, "offset": self.offset}
        return {"type": "relative_position", "relation": self.relation, "threshold": self.threshold}


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
) -> dict[str, HalfPlanePredicate]:
    """Parse generic half-plane predicates, optionally anchored to a region."""
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
        if data.get("type", "relative_position") != "relative_position":
            raise ValueError(f"Predicate {name!r}.type must be 'relative_position'")
        relation = data.get("relation")
        if relation not in {"above", "below", "left_of", "right_of"}:
            raise ValueError(f"Predicate {name!r}.relation must be above, below, left_of, or right_of")
        offset = data.get("offset", 0.0)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(float(offset)):
            raise ValueError(f"Predicate {name!r}.offset must be a finite number")

        reference_name = data.get("reference")
        explicit_threshold = data.get("threshold")
        if (reference_name is None) == (explicit_threshold is None):
            raise ValueError(f"Predicate {name!r} must define exactly one of reference or threshold")

        boundary = data.get("boundary", "edge")
        if boundary != "edge":
            raise ValueError(f"Predicate {name!r}.boundary must be 'edge': relative predicates use the outermost extent of their reference region")
        if reference_name is not None:
            if reference_name not in regions:
                raise ValueError(f"Predicate {name!r} references unknown region {reference_name!r}")
            if float(offset) < 0.0:
                raise ValueError(f"Predicate {name!r}.offset must be non-negative when a reference is used")
            reference_region = regions[reference_name]
            vertical = relation in {"above", "below"}
            threshold = reference_region.center_y if vertical else reference_region.center_x
            direction = 1.0 if relation in {"above", "right_of"} else -1.0
            threshold += direction * (reference_region.radius + float(offset))
        else:
            if isinstance(explicit_threshold, bool) or not isinstance(explicit_threshold, (int, float)) or not math.isfinite(float(explicit_threshold)):
                raise ValueError(f"Predicate {name!r}.threshold must be a finite number")
            threshold = float(explicit_threshold) + float(offset)
            reference_name = None

        predicates[name] = HalfPlanePredicate(
            relation=relation,
            threshold=threshold,
            reference=reference_name,
            boundary=boundary,
            offset=float(offset),
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
        if isinstance(region, HalfPlanePredicate) and region.reference is not None:
            continue
        cells = frozenset(
            (x, y)
            for x in range(width)
            for y in range(height)
            if region.intersects_cell(grid_cell_bounds(x, y, width, height))
        )
        if not cells and isinstance(region, CircularRegion):
            raise ValueError(f"Region {name!r} does not intersect the {width}x{height} grid")
        rasterized[name] = cells

    # A referenced directional predicate is aligned to the bounding rows or
    # columns of the cells assigned to its waypoint. The resulting predicate
    # is a global half-plane: x is irrelevant for above/below and y is
    # irrelevant for left/right.
    for name, predicate in regions.items():
        if not isinstance(predicate, HalfPlanePredicate) or predicate.reference is None:
            continue
        reference_cells = rasterized[predicate.reference]
        reference_x = [x for x, _ in reference_cells]
        reference_y = [y for _, y in reference_cells]
        min_x, max_x = min(reference_x), max(reference_x)
        min_y, max_y = min(reference_y), max(reference_y)
        x_step = (OBSERVATION_X_BOUNDS[1] - OBSERVATION_X_BOUNDS[0]) / width
        y_step = (OBSERVATION_Y_BOUNDS[1] - OBSERVATION_Y_BOUNDS[0]) / height
        x_margin_cells = math.ceil(predicate.offset / x_step)
        y_margin_cells = math.ceil(predicate.offset / y_step)
        if predicate.relation == "above":
            first_y = max_y + 1 + y_margin_cells
            cells = ((x, y) for x in range(width) for y in range(first_y, height))
        elif predicate.relation == "below":
            last_y = min_y - y_margin_cells
            cells = ((x, y) for x in range(width) for y in range(0, last_y))
        elif predicate.relation == "right_of":
            first_x = max_x + 1 + x_margin_cells
            cells = ((x, y) for x in range(first_x, width) for y in range(height))
        else:
            last_x = min_x - x_margin_cells
            cells = ((x, y) for x in range(0, last_x) for y in range(height))
        rasterized[name] = frozenset(cells)
    return rasterized


def truth_assignment_from_cell(
    region_cells: Mapping[str, frozenset[tuple[int, int]]], x: int, y: int
) -> dict[str, bool]:
    """Evaluate abstract propositions using the grid over-approximation."""
    return {name: (x, y) in cells for name, cells in region_cells.items()}
