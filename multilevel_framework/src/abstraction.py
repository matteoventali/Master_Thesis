"""Configuration and mappings for a hierarchy of rectangular grid abstractions.

All mappings operate in the normalised square ``[0, 1] x [0, 1]``.  This makes
them independent of LunarLander's observation bounds and, importantly, allows
both the source and destination grids to have arbitrary dimensions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LearningConfig:
    """Tabular Q-learning parameters for one abstract level."""

    episodes: int = 10_000
    max_steps: int = 100
    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999
    gamma_shaping: float | None = None
    seed: int = 0
    log_interval: int = 1000
    eval_interval: int = 10_000
    eval_episodes: int = 500
    eval_seed: int = 100_000

    def __post_init__(self):
        for name in ("episodes", "max_steps", "log_interval", "eval_interval", "eval_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"learning.{name} must be a positive integer")
        numeric_fields = ("alpha", "epsilon_start", "epsilon_min", "epsilon_decay")
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"learning.{name} must be a finite number")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("learning.alpha must be in the interval (0, 1]")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("learning epsilon values must satisfy 0 <= epsilon_min <= epsilon_start <= 1")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("learning.epsilon_decay must be in the interval (0, 1]")
        if self.gamma_shaping is not None:
            if isinstance(self.gamma_shaping, bool) or not isinstance(self.gamma_shaping, (int, float)) or not math.isfinite(self.gamma_shaping):
                raise ValueError("learning.gamma_shaping must be a finite number")
            if not 0.0 < self.gamma_shaping <= 1.0:
                raise ValueError("learning.gamma_shaping must be in the interval (0, 1]")
        for name in ("seed", "eval_seed"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f"learning.{name} must be an integer")

    @classmethod
    def from_dict(cls, data, level_name):
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(f"{level_name}.learning must be a JSON object")
        allowed = {
            "episodes",
            "max_steps",
            "alpha",
            "epsilon_start",
            "epsilon_min",
            "epsilon_decay",
            "gamma_shaping",
            "seed",
            "log_interval",
            "eval_interval",
            "eval_episodes",
            "eval_seed",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"{level_name}.learning contains unknown fields: {unknown}")
        return cls(**data)


@dataclass(frozen=True)
class GridLevel:
    """One configured abstraction level."""

    width: int
    height: int
    name: str
    algorithm: str
    learning: LearningConfig = field(default_factory=LearningConfig)
    value_function_method: str = "max"
    checkpoint: str | None = None

    def __post_init__(self):
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(f"{self.name}.grid_w must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(f"{self.name}.grid_h must be a positive integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Every abstraction level must have a non-empty name")
        if self.algorithm not in ("value_iteration", "learning"):
            raise ValueError(f"{self.name}.algorithm must be either 'value_iteration' or 'learning'")
        if self.value_function_method not in ("max", "policy_evaluation"):
            raise ValueError(f"{self.name}.value_function_method must be either 'max' or 'policy_evaluation'")
        if self.checkpoint is not None and (not isinstance(self.checkpoint, str) or not self.checkpoint.strip()):
            raise ValueError(f"{self.name}.checkpoint must be a non-empty path")

    @property
    def shape(self):
        """Return ``(width, height)`` for mapping helpers."""
        return self.width, self.height


@dataclass(frozen=True)
class AbstractionConfig:
    """Validated ordered collection of abstraction levels.

    ``levels[0]`` is always the abstraction used by the automaton and training.
    The remaining levels are ordered dependencies: states at level *i* are
    mapped online to level *i + 1*, whose V-function is used as potential.
    """

    levels: tuple[GridLevel, ...]

    def __post_init__(self):
        if not self.levels:
            raise ValueError("abstraction.json must define at least one level")
        names = [level.name for level in self.levels]
        if len(names) != len(set(names)):
            raise ValueError("Abstraction level names must be unique")
        if any(level.algorithm != "learning" for level in self.levels[:-1]):
            raise ValueError("Every non-top abstraction level must use learning")

    @property
    def primary(self):
        """Return level 1, whose coordinates define the automaton semantics."""
        return self.levels[0]

    def algorithm_for_index(self, index):
        """Return the validated algorithm assigned to one hierarchy level."""
        if not 0 <= index < len(self.levels):
            raise IndexError("Abstraction level index out of range")
        return self.levels[index].algorithm

    @classmethod
    def from_dict(cls, data, base_dir=None):
        if not isinstance(data, dict):
            raise ValueError("The abstraction configuration must be a JSON object")
        raw_levels = data.get("levels")
        if not isinstance(raw_levels, list) or not raw_levels:
            raise ValueError("abstraction.json must contain a non-empty 'levels' array")

        levels = []
        for index, raw_level in enumerate(raw_levels, start=1):
            if not isinstance(raw_level, dict):
                raise ValueError(f"levels[{index - 1}] must be a JSON object")
            name = raw_level.get("name", f"level{index}")
            width = raw_level.get("grid_w", raw_level.get("width"))
            height = raw_level.get("grid_h", raw_level.get("height"))
            if width is None or height is None:
                raise ValueError(f"{name} must define grid_w/grid_h (or width/height)")
            is_top_level = index == len(raw_levels)
            if "solver" in raw_level:
                raise ValueError(f"{name} must use 'algorithm', not 'solver'")
            if not is_top_level and "algorithm" in raw_level:
                raise ValueError(f"{name} is not the top level: lower levels always use learning and must not define algorithm")
            algorithm = raw_level.get("algorithm", "value_iteration") if is_top_level else "learning"
            if algorithm not in ("value_iteration", "learning"):
                raise ValueError(f"{name}.algorithm must be either 'value_iteration' or 'learning'")
            if is_top_level and algorithm == "value_iteration" and "learning" in raw_level:
                raise ValueError(f"{name} uses VI and must not define learning parameters")
            if is_top_level and algorithm == "learning" and isinstance(raw_level.get("learning"), dict) and "gamma_shaping" in raw_level["learning"]:
                raise ValueError(f"{name} is the top level and must not define gamma_shaping because it has no upper potential")
            learning = LearningConfig.from_dict(raw_level.get("learning"), name)
            value_function_method = raw_level.get("value_function_method", "max")
            checkpoint = raw_level.get("checkpoint")
            if checkpoint is not None:
                if not isinstance(checkpoint, str) or not checkpoint.strip():
                    raise ValueError(f"{name}.checkpoint must be a non-empty path")
                checkpoint_path = Path(checkpoint).expanduser()
                if base_dir is not None and not checkpoint_path.is_absolute():
                    checkpoint_path = Path(base_dir) / checkpoint_path
                checkpoint = str(checkpoint_path.resolve())
            levels.append(GridLevel(width=width, height=height, name=name, algorithm=algorithm, learning=learning, value_function_method=value_function_method, checkpoint=checkpoint))
        return cls(tuple(levels))

    @classmethod
    def load(cls, path):
        """Load and validate an ``abstraction.json`` file."""
        path = Path(path)
        with path.open(encoding="utf-8") as config_file:
            return cls.from_dict(json.load(config_file), base_dir=path.resolve().parent)


def _validate_dimensions(width, height, label):
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{label} dimensions must be positive integers")


def _validate_cell(cell, width, height, label="source"):
    if not isinstance(cell, (tuple, list)) or len(cell) != 2:
        raise ValueError("A grid cell must contain exactly two coordinates")
    x, y = cell
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError("Grid x-coordinate must be an integer")
    if isinstance(y, bool) or not isinstance(y, int):
        raise ValueError("Grid y-coordinate must be an integer")
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"Cell ({x}, {y}) is outside the {label} grid {width}x{height}")
    return x, y


def map_cell(cell, source_width, source_height, target_width, target_height):
    """Map one cell centre between arbitrary rectangular grids.

    The same function is used in either direction by swapping source and target
    dimensions.  A centre-based mapping is deterministic even when a coarse
    cell overlaps multiple finer cells.
    """
    _validate_dimensions(source_width, source_height, "Source")
    _validate_dimensions(target_width, target_height, "Target")
    x, y = _validate_cell(cell, source_width, source_height)
    target_x = min(int(((x + 0.5) / source_width) * target_width), target_width - 1)
    target_y = min(int(((y + 0.5) / source_height) * target_height), target_height - 1)
    return target_x, target_y


def map_state(state, source_width, source_height, target_width, target_height):
    """Map the spatial part of ``(x, y, q)`` while preserving DFA state ``q``."""
    if not isinstance(state, (tuple, list)) or len(state) != 3:
        raise ValueError("An abstract state must be (x, y, q)")
    x, y = map_cell(state[:2], source_width, source_height, target_width, target_height)
    return x, y, state[2]


def overlapping_cells(cell, source_width, source_height, target_width, target_height):
    """Return every destination cell with positive area overlap.

    This is the set-valued counterpart of :func:`map_cell`, useful for exact
    coarse-to-fine and fine-to-coarse relationships.
    """
    _validate_dimensions(source_width, source_height, "Source")
    _validate_dimensions(target_width, target_height, "Target")
    x, y = _validate_cell(cell, source_width, source_height)

    min_x = int(math.floor(x * target_width / source_width))
    max_x = int(math.ceil((x + 1) * target_width / source_width) - 1)
    min_y = int(math.floor(y * target_height / source_height))
    max_y = int(math.ceil((y + 1) * target_height / source_height) - 1)
    return [
        (target_x, target_y)
        for target_x in range(max(0, min_x), min(target_width - 1, max_x) + 1)
        for target_y in range(max(0, min_y), min(target_height - 1, max_y) + 1)
    ]


def map_waypoints(waypoints, source_width, source_height, target_width, target_height):
    """Map a proposition-to-cell dictionary between arbitrary grids."""
    return {
        name: map_cell(tuple(coordinates), source_width, source_height, target_width, target_height)
        for name, coordinates in waypoints.items()
    }
