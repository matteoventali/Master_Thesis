"""Visualise the abstract grid on top of the LunarLander environment.

The conversion used here is the inverse of ``utils.phi_mapping_grid``.  Grid
cells in the resulting image therefore represent exactly the abstract states
used by the trainer, rather than an evenly spaced, screen-only decoration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = SCRIPT_DIR.parent
SRC_DIR = FRAMEWORK_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Ellipse, Rectangle

from abstraction import AbstractionConfig
from spatial_regions import OBSERVATION_X_BOUNDS, OBSERVATION_Y_BOUNDS, BoxPredicate, CircularRegion, HalfPlanePredicate, load_task_propositions, rasterize_regions
from utils import phi_mapping_grid, spatial_grid_boundaries


DEFAULT_CONFIG = FRAMEWORK_DIR / "config" / "trajectory.json"
DEFAULT_ABSTRACTION_CONFIG = FRAMEWORK_DIR / "config" / "abstraction.json"


@dataclass(frozen=True)
class LunarLanderGeometry:
    """Screen/world constants required to project observations onto a frame."""

    viewport_width: int
    viewport_height: int
    scale: float
    helipad_y: float
    leg_down: float


def geometry_from_env(env: gym.Env) -> LunarLanderGeometry:
    """Read projection constants from a reset LunarLander environment."""
    from gymnasium.envs.box2d import lunar_lander

    base_env = env.unwrapped
    if not hasattr(base_env, "helipad_y"):
        raise TypeError("The supplied environment is not a LunarLander environment")
    return LunarLanderGeometry(
        viewport_width=lunar_lander.VIEWPORT_W,
        viewport_height=lunar_lander.VIEWPORT_H,
        scale=lunar_lander.SCALE,
        helipad_y=float(base_env.helipad_y),
        leg_down=lunar_lander.LEG_DOWN,
    )


def observation_to_pixel(
    observation: Sequence[float], geometry: LunarLanderGeometry
) -> tuple[float, float]:
    """Project LunarLander's normalised (x, y) observation onto RGB pixels."""
    half_world_width = geometry.viewport_width / geometry.scale / 2.0
    half_world_height = geometry.viewport_height / geometry.scale / 2.0
    world_x = (float(observation[0]) + 1.0) * half_world_width
    world_y = (
        float(observation[1]) * half_world_height
        + geometry.helipad_y
        + geometry.leg_down / geometry.scale
    )
    pixel_x = world_x * geometry.scale
    pixel_y = geometry.viewport_height - world_y * geometry.scale
    return pixel_x, pixel_y


def pixel_to_observation(
    pixel_x: float,
    pixel_y: float,
    geometry: LunarLanderGeometry,
) -> tuple[float, float]:
    """Invert the frame projection for the two discretised coordinates."""
    half_world_width = geometry.viewport_width / geometry.scale / 2.0
    half_world_height = geometry.viewport_height / geometry.scale / 2.0
    world_x = float(pixel_x) / geometry.scale
    world_y = (geometry.viewport_height - float(pixel_y)) / geometry.scale
    observation_x = world_x / half_world_width - 1.0
    observation_y = (
        world_y - geometry.helipad_y - geometry.leg_down / geometry.scale
    ) / half_world_height
    return observation_x, observation_y


def _grid_boundaries(
    grid_w: int,
    grid_h: int,
    geometry: LunarLanderGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the boundaries implied by the active spatial discretizer."""
    x_normalised, y_normalised = spatial_grid_boundaries(grid_w, grid_h)
    x_pixels = np.array(
        [observation_to_pixel((x, 0.0), geometry)[0] for x in x_normalised]
    )
    y_pixels = np.array(
        [observation_to_pixel((0.0, y), geometry)[1] for y in y_normalised]
    )

    # phi_mapping_grid clips observations outside its nominal domain into its
    # edge cells. Extend those cells to the RGB viewport edges whenever the
    # corresponding border observation maps to index 0 or to the last index.
    # Internal boundaries remain entirely inferred from the active mapper.
    left_observation_x, top_observation_y = pixel_to_observation(
        0.0, 0.0, geometry
    )
    right_observation_x, bottom_observation_y = pixel_to_observation(
        geometry.viewport_width, geometry.viewport_height, geometry
    )
    if phi_mapping_grid((left_observation_x, 0.0), grid_w, grid_h)[0] == 0:
        x_pixels[0] = min(x_pixels[0], 0.0)
    if phi_mapping_grid((right_observation_x, 0.0), grid_w, grid_h)[0] == grid_w - 1:
        x_pixels[-1] = max(x_pixels[-1], float(geometry.viewport_width))
    if phi_mapping_grid((0.0, bottom_observation_y), grid_w, grid_h)[1] == 0:
        y_pixels[0] = max(y_pixels[0], float(geometry.viewport_height))
    if phi_mapping_grid((0.0, top_observation_y), grid_w, grid_h)[1] == grid_h - 1:
        y_pixels[-1] = min(y_pixels[-1], 0.0)

    return x_pixels, y_pixels


def abstract_cell_to_pixel(
    grid_x: int,
    grid_y: int,
    grid_w: int,
    grid_h: int,
    geometry: LunarLanderGeometry,
) -> tuple[float, float]:
    """Return the pixel coordinates of an abstract cell's centre."""
    if not (0 <= grid_x < grid_w and 0 <= grid_y < grid_h):
        raise ValueError(f"Abstract cell ({grid_x}, {grid_y}) is outside the grid")
    x_lines, y_lines = _grid_boundaries(grid_w, grid_h, geometry)
    return (
        float((x_lines[grid_x] + x_lines[grid_x + 1]) / 2.0),
        float((y_lines[grid_y] + y_lines[grid_y + 1]) / 2.0),
    )


def draw_abstract_grid(
    frame: np.ndarray,
    geometry: LunarLanderGeometry,
    grid_w: int,
    grid_h: int,
    regions: Mapping[str, object] | None = None,
    observation: Sequence[float] | None = None,
    title: str = "LunarLander with Abstract Grid",
    show_grid: bool = False,
):
    """Draw spatial propositions and optionally the effective abstract grid."""
    if grid_w < 2 or grid_h < 2:
        raise ValueError("grid_w and grid_h must both be at least 2")

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.imshow(frame)
    x_lines, y_lines = _grid_boundaries(grid_w, grid_h, geometry)
    x_centres = (x_lines[:-1] + x_lines[1:]) / 2.0
    y_centres = (y_lines[:-1] + y_lines[1:]) / 2.0
    grid_color = "#ff1744"

    if show_grid:
        for x_pixel in x_lines:
            axis.axvline(x_pixel, color=grid_color, linewidth=1.6, alpha=0.95)
        for y_pixel in y_lines:
            axis.axhline(y_pixel, color=grid_color, linewidth=1.6, alpha=0.95)

    # The mapping clips everything outside its stated observation domain into
    # an edge cell; tint the currently occupied abstract cell when requested.
    if show_grid and observation is not None:
        abstract_x, abstract_y = phi_mapping_grid(observation, grid_w, grid_h)
        x0, x1 = sorted((x_lines[abstract_x], x_lines[abstract_x + 1]))
        y0, y1 = sorted((y_lines[abstract_y], y_lines[abstract_y + 1]))
        x0, x1 = np.clip((x0, x1), 0, geometry.viewport_width)
        y0, y1 = np.clip((y0, y1), 0, geometry.viewport_height)
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#00e5ff",
                edgecolor="#00e5ff",
                linewidth=2.5,
                alpha=0.25,
                label=f"Current cell ({abstract_x}, {abstract_y})",
            )
        )

    # Every proposition first highlights the abstract cells it intersects. Its
    # exact continuous geometry is then drawn on top: circles as ellipses,
    # boxes as rectangles, and half-planes as a shaded side of a threshold.
    # Highly saturated colours keep overlapping task regions recognisable on
    # top of both the dark sky and the bright LunarLander terrain.
    region_palette = (
        "#ffea00",  # vivid yellow
        "#d500f9",  # electric purple
        "#00e676",  # neon green
        "#ff3d00",  # vivid orange-red
        "#00e5ff",  # electric cyan
        "#2979ff",  # vivid blue
        "#ff1744",  # vivid red
    )
    rasterized = rasterize_regions(regions, grid_w, grid_h) if show_grid and regions else {}
    for region_index, (name, region) in enumerate((regions or {}).items()):
        color = region_palette[region_index % len(region_palette)]
        if show_grid:
            cell_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
            for grid_x, grid_y in rasterized[name]:
                cell_mask[grid_y, grid_x] = 1
            cell_colormap = ListedColormap(((0.0, 0.0, 0.0, 0.0), to_rgba(color, alpha=0.48)))
            axis.pcolormesh(x_lines, y_lines, cell_mask, shading="flat", cmap=cell_colormap, vmin=0, vmax=1, zorder=2)
        if isinstance(region, CircularRegion):
            center = observation_to_pixel((region.center_x, region.center_y), geometry)
            radius_x = abs(observation_to_pixel((region.center_x + region.radius, region.center_y), geometry)[0] - center[0])
            radius_y = abs(observation_to_pixel((region.center_x, region.center_y + region.radius), geometry)[1] - center[1])
            axis.add_patch(
                Ellipse(
                    center, 2 * radius_x, 2 * radius_y,
                    facecolor=color, edgecolor="black", linewidth=3.2, alpha=0.78,
                    label=f"{name}: circle", zorder=4,
                )
            )
        elif isinstance(region, BoxPredicate):
            corner_a = observation_to_pixel((region.x_min, region.y_min), geometry)
            corner_b = observation_to_pixel((region.x_max, region.y_max), geometry)
            pixel_x_min, pixel_x_max = sorted((corner_a[0], corner_b[0]))
            pixel_y_min, pixel_y_max = sorted((corner_a[1], corner_b[1]))
            axis.add_patch(
                Rectangle(
                    (pixel_x_min, pixel_y_min), pixel_x_max - pixel_x_min, pixel_y_max - pixel_y_min,
                    facecolor=color, edgecolor="black", linewidth=3.2, alpha=0.74,
                    label=f"{name}: box", zorder=4,
                )
            )
        elif isinstance(region, HalfPlanePredicate):
            x_min, x_max = OBSERVATION_X_BOUNDS
            y_min, y_max = OBSERVATION_Y_BOUNDS
            if region.axis == "x":
                if region.operator in {">", ">="}:
                    x_min = max(x_min, region.threshold)
                else:
                    x_max = min(x_max, region.threshold)
            elif region.operator in {">", ">="}:
                y_min = max(y_min, region.threshold)
            else:
                y_max = min(y_max, region.threshold)

            # A threshold outside the observation domain can describe either
            # the empty set or the complete domain. Only non-empty valid areas
            # receive a patch; the boundary is shown when it is in-domain.
            if x_min <= x_max and y_min <= y_max:
                corner_a = observation_to_pixel((x_min, y_min), geometry)
                corner_b = observation_to_pixel((x_max, y_max), geometry)
                pixel_x_min, pixel_x_max = sorted((corner_a[0], corner_b[0]))
                pixel_y_min, pixel_y_max = sorted((corner_a[1], corner_b[1]))
                axis.add_patch(
                    Rectangle(
                        (pixel_x_min, pixel_y_min), pixel_x_max - pixel_x_min, pixel_y_max - pixel_y_min,
                        facecolor=color, edgecolor=color, linewidth=2.4, alpha=0.58,
                        label=f"{name}: {region.axis} {region.operator} {region.threshold:g}", zorder=3,
                    )
                )

            if region.axis == "x" and OBSERVATION_X_BOUNDS[0] <= region.threshold <= OBSERVATION_X_BOUNDS[1]:
                threshold_pixel = observation_to_pixel((region.threshold, 0.0), geometry)[0]
                axis.axvline(threshold_pixel, color="black", linewidth=5.2, alpha=0.9, zorder=5)
                axis.axvline(threshold_pixel, color=color, linestyle="--", linewidth=3.4, alpha=1.0, zorder=6)
            if region.axis == "y" and OBSERVATION_Y_BOUNDS[0] <= region.threshold <= OBSERVATION_Y_BOUNDS[1]:
                threshold_pixel = observation_to_pixel((0.0, region.threshold), geometry)[1]
                axis.axhline(threshold_pixel, color="black", linewidth=5.2, alpha=0.9, zorder=5)
                axis.axhline(threshold_pixel, color=color, linestyle="--", linewidth=3.4, alpha=1.0, zorder=6)
        else:
            raise TypeError(f"Unsupported spatial proposition {name!r}: {type(region).__name__}")

    axis.set_xlim(0, geometry.viewport_width)
    # A small part of the configured y-domain can lie above the RGB viewport.
    # Keep it in view so that no abstract row or coordinate label disappears.
    visible_top = min(0.0, float(np.min(y_lines)))
    axis.set_ylim(geometry.viewport_height, visible_top)
    # Put the abstract coordinates at cell centres. Since image coordinates
    # grow downwards while abstract y grows upwards, y_centres is descending:
    # label 0 consequently appears at the bottom and grid_h - 1 at the top.
    axis.set_title(title)
    if show_grid:
        axis.set_xticks(x_centres, labels=range(grid_w))
        axis.set_yticks(y_centres, labels=range(grid_h))
        axis.set_xlabel("Abstract x-coordinate")
        axis.set_ylabel("Abstract y-coordinate")
        axis.tick_params(axis="both", which="major", color=grid_color, labelcolor=grid_color, labelsize=10, width=1.5, length=5)
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontweight("bold")
    else:
        axis.set_xticks([])
        axis.set_yticks([])

    if observation is not None:
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    else:
        figure.tight_layout()
    return figure


def generate_overlay(
    output_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    seed: int | None = 0,
    abstraction_config_path: str | Path = DEFAULT_ABSTRACTION_CONFIG,
    level: int = 1,
    show_grid: bool = False,
    show_regions: bool = True,
) -> Path:
    """Reset LunarLander and save one annotated RGB frame as a PNG."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    abstraction_config = AbstractionConfig.load(abstraction_config_path)
    if not 1 <= level <= len(abstraction_config.levels):
        raise ValueError(
            f"level must be between 1 and {len(abstraction_config.levels)}"
        )
    selected_level = abstraction_config.levels[level - 1]
    _regions, _predicates, task_propositions = load_task_propositions(config.get("regions"), config.get("predicates"))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed)
        frame = env.render()
        geometry = geometry_from_env(env)
        figure = draw_abstract_grid(
            frame=frame,
            geometry=geometry,
            grid_w=selected_level.width,
            grid_h=selected_level.height,
            regions=task_propositions if show_regions else None,
            observation=observation,
            title=(
                f"LunarLander — level{level} ({selected_level.name}, "
                f"{selected_level.width}x{selected_level.height})"
            ),
            show_grid=show_grid,
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
    finally:
        env.close()
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LunarLander frame with the abstract grid overlaid."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--abstraction-config",
        type=Path,
        default=DEFAULT_ABSTRACTION_CONFIG,
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="1-based abstraction level to draw (default: 1).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-grid", action="store_true", help="Draw abstract grid lines, coordinates, current cell, and rasterized proposition cells.")
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Draw the abstract grid without task regions or rasterized proposition cells.",
    )
    args = parser.parse_args()
    output_path = args.output or FRAMEWORK_DIR / "results" / "abstract_grid_overlay.png"
    saved_path = generate_overlay(
        output_path,
        args.config,
        args.seed,
        args.abstraction_config,
        args.level,
        args.show_grid or args.grid_only,
        not args.grid_only,
    )
    print(f"Image saved to: {saved_path}")


if __name__ == "__main__":
    main()
