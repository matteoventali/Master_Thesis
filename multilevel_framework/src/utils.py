"""Spatial mapping and plotting utilities for the multilevel framework."""

# ==============================
# Standard library imports
# ==============================

import os
import math

# ==============================
# External imports
# ==============================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


LEARNING_REWARD_COLOR = "#0072B2"
TASK_REWARD_COLOR = "#D55E00"
EPSILON_COLOR = "#E6AB02"
RAW_DATA_COLOR = "#777777"
REFERENCE_LINE_COLOR = "#666666"
SERIES_COLORS = (
    LEARNING_REWARD_COLOR,
    TASK_REWARD_COLOR,
    "#CC79A7",
    "#009E73",
    "#56B4E9",
    "#000000",
)
EPSILON_LINESTYLES = ("--", (0, (5, 2, 1, 2)), ":", "-.")
PAPER_COLORS = SERIES_COLORS

plt.rcParams.update( { "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10, "axes.labelsize": 10, "axes.linewidth": 0.8, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 300, "savefig.facecolor": "white", } )


# ==============================
# Spatial discretization and grid geometry
# ==============================

# Legacy discretization. To reactivate it, uncomment this function and comment
# out the active phi_mapping_grid implementation immediately below.
#
# def phi_mapping_grid(obs, grid_w=12, grid_h=12):
#     """Map coordinates using the original grid_size - 1 discretization."""
#     x, y = float(obs[0]), float(obs[1])
#     abstract_x = int(np.clip((x + 1.0) / 2.0 * (grid_w - 1), 0, grid_w - 1))
#     abstract_y = int(np.clip(y / 1.5 * (grid_h - 1), 0, grid_h - 1))
#     return abstract_x, abstract_y


def phi_mapping_grid(obs, grid_w=12, grid_h=12):
    """Map LunarLander coordinates to uniform bins over x=[-1,1], y=[0,1.5]."""
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("grid_w and grid_h must be positive")

    x, y = float(obs[0]), float(obs[1])
    abstract_x = int(np.floor((x + 1.0) / 2.0 * grid_w))
    abstract_y = int(np.floor(y / 1.5 * grid_h))
    abstract_x = int(np.clip(abstract_x, 0, grid_w - 1))
    abstract_y = int(np.clip(abstract_y, 0, grid_h - 1))
    return abstract_x, abstract_y


def _axis_boundaries(map_axis, size, lower, upper):
    """Infer bin boundaries from the active mapper."""
    if size <= 0:
        raise ValueError("grid dimensions must be positive")

    boundaries = [float(lower)]
    iterations = 60
    for target_index in range(1, size):
        left, right = float(lower), float(upper)
        for _ in range(iterations):
            midpoint = (left + right) / 2.0
            if map_axis(midpoint) < target_index:
                left = midpoint
            else:
                right = midpoint
        boundaries.append(right)
    boundaries.append(float(upper))
    return np.asarray(boundaries, dtype=float)


def spatial_grid_boundaries(grid_w=12, grid_h=12):
    """Return x/y bin boundaries implied by the active phi_mapping_grid."""
    x_boundaries = _axis_boundaries( lambda x: phi_mapping_grid((x, 0.0), grid_w, grid_h)[0], grid_w, -1.0, 1.0, )
    y_boundaries = _axis_boundaries( lambda y: phi_mapping_grid((0.0, y), grid_w, grid_h)[1], grid_h, 0.0, 1.5, )
    return x_boundaries, y_boundaries


def phi_mapping_sequential(obs, q, grid_w=12, grid_h=12):
    abstract_x, abstract_y = phi_mapping_grid(obs, grid_w, grid_h)
    return abstract_x, abstract_y, q


def lunar_lander_visible_observation_bounds():
    """Return the normalised x/y bounds covered by LunarLander's RGB viewport."""
    # Gymnasium's rendering geometry is constant; keeping the values here lets
    # post-processing run without importing the optional Box2D runtime.
    scale, viewport_width, viewport_height, leg_down = 30.0, 600.0, 400.0, 18.0
    viewport_world_width = viewport_width / scale
    viewport_world_height = viewport_height / scale
    helipad_y = viewport_world_height / 4.0
    lander_y_offset = helipad_y + leg_down / scale
    half_world_height = viewport_world_height / 2.0
    visible_y_min = (0.0 - lander_y_offset) / half_world_height
    visible_y_max = (viewport_world_height - lander_y_offset) / half_world_height
    return -1.0, 1.0, visible_y_min, visible_y_max


def _draw_visible_area_overlay(axis, width, height):
    """Mark which portion of the active abstract grid lies in the RGB viewport."""
    visible_x_min, visible_x_max, visible_y_min, visible_y_max = (
        lunar_lander_visible_observation_bounds()
    )
    x_boundaries, y_boundaries = spatial_grid_boundaries(width, height)

    def to_plot(value, boundaries):
        value = float(np.clip(value, boundaries[0], boundaries[-1]))
        index = int(np.searchsorted(boundaries, value, side="right") - 1)
        index = int(np.clip(index, 0, len(boundaries) - 2))
        lower, upper = boundaries[index], boundaries[index + 1]
        fraction = 0.0 if upper <= lower else (value - lower) / (upper - lower)
        return index - 0.5 + fraction

    left = float(to_plot(visible_x_min, x_boundaries))
    right = float(to_plot(visible_x_max, x_boundaries))
    bottom = float(to_plot(visible_y_min, y_boundaries))
    top = float(to_plot(visible_y_max, y_boundaries))

    axis.add_patch( Rectangle( (left, bottom), right - left, top - bottom, fill=False, edgecolor="#ff1744", linewidth=1.4, linestyle="--", label="Visible RGB viewport", zorder=5, ) )
    axis.legend( loc="upper center", bbox_to_anchor=(0.5, -0.08), borderaxespad=0.0, frameon=False, fontsize=8, )

# ==============================
# Abstract-potential heatmaps
# ==============================


def save_sequential_heatmaps( abstract_mdp, filename_prefix="v_star", output_dir=None, annotate_cells=False, ):
    """
    Generates and saves a separate heatmap for V* for each phase defined in the MDP,
    without any waypoint or goal markers (clean heatmap).
    """
    # A caller can isolate each abstraction in img/heatmaps/level1, level2, ...
    output_dir = output_dir or os.path.join("img", "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = os.path.basename(filename_prefix)
    
    width, height = abstract_mdp.width, abstract_mdp.height
    
    for current_q in abstract_mdp.automaton.states:
        matrix = np.full((height, width), np.nan)
        for (x, y, q), value in abstract_mdp.v_star.items():
            if (
                q == current_q
                and 0 <= x < width
                and 0 <= y < height
            ):
                matrix[y, x] = value
                
        plt.figure(figsize=(6.4, 5.4), constrained_layout=True)
        # Round colors so invisible floating-point residuals cannot span the
        # whole colormap.
        color_matrix = np.round(matrix, decimals=1)
        finite_values = color_matrix[np.isfinite(color_matrix)]
        if len(finite_values) > 0 and np.all(finite_values == finite_values[0]):
            constant_value = float(finite_values[0])
            if constant_value > 0.0:
                im = plt.imshow(color_matrix, cmap='viridis', origin='lower', vmin=0.0, vmax=constant_value)
            elif constant_value < 0.0:
                im = plt.imshow(color_matrix, cmap='viridis', origin='lower', vmin=constant_value, vmax=0.0)
            else:
                im = plt.imshow(color_matrix, cmap='viridis', origin='lower', vmin=0.0, vmax=1.0)
        else:
            im = plt.imshow(color_matrix, cmap='viridis', origin='lower')
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*)")
        
        ax = plt.gca()
        if annotate_cells:
            color_min = float(finite_values.min()) if len(finite_values) else 0.0
            color_max = float(finite_values.max()) if len(finite_values) else 0.0
            color_midpoint = (color_min + color_max) / 2.0
            for y in range(height):
                for x in range(width):
                    value = matrix[y, x]
                    if np.isnan(value):
                        continue
                    truth_assignment = abstract_mdp._get_truth_assignment(x, y)
                    next_q = abstract_mdp.automaton.get_next_q(current_q, truth_assignment)
                    changes_automaton_state = next_q != current_q
                    if changes_automaton_state:
                        ax.text(
                            x,
                            y,
                            f"→q{next_q}",
                            ha="center",
                            va="center",
                            color="#d32f2f",
                            fontsize=6.5,
                            fontweight="bold",
                        )
                    else:
                        text_color = "white" if color_matrix[y, x] < color_midpoint else "black"
                        ax.text(x, y, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=7)
        ax.set_xlabel("Grid x")
        ax.set_ylabel("Grid y")
        _draw_visible_area_overlay(ax, width, height)
        
        # Keep the heatmap free of waypoint and goal markers.
            
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_q{current_q}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print( f" -> Generated V* Heatmap for {abstract_mdp.level_name}, " f"DFA State q={current_q}" )


def save_multilevel_heatmaps( multilevel_mdp, filename_prefix="v_star", output_root=None, annotate_cells=False, ):
    """Save each level's heatmaps under ``level1``, ``level2``, and so on."""
    output_root = output_root or os.path.join("img", "heatmaps")
    generated_directories = []
    for level_number, abstract_mdp in enumerate(multilevel_mdp.levels, start=1):
        level_directory = os.path.join(output_root, f"level{level_number}")
        save_sequential_heatmaps( abstract_mdp, filename_prefix=filename_prefix, output_dir=level_directory, annotate_cells=annotate_cells, )
        generated_directories.append(level_directory)
    return generated_directories


def save_multilevel_value_functions(multilevel_mdp, output_root=None):
    """Save every abstract V-function as a dense numeric NPZ array."""
    output_root = output_root or os.path.join("results", "abstract_value_functions")
    generated_files = []
    for level_number, abstract_mdp in enumerate(multilevel_mdp.levels, start=1):
        level_directory = os.path.join(output_root, f"level{level_number}")
        os.makedirs(level_directory, exist_ok=True)
        dfa_states = np.asarray(sorted(abstract_mdp.automaton.states), dtype=np.int64)
        q_indices = {int(q): index for index, q in enumerate(dfa_states)}

        def dense_values(value_function, label):
            dense = np.full((len(dfa_states), abstract_mdp.height, abstract_mdp.width), np.nan, dtype=np.float64)
            for (x, y, q), value in value_function.items():
                dense[q_indices[q], y, x] = value
            if np.isnan(dense).any():
                raise ValueError(f"{abstract_mdp.level_name} {label} V-function is missing one or more product states")
            return dense

        unbiased_values = dense_values(abstract_mdp.unbiased_v_star, "unbiased")
        value_path = os.path.join(level_directory, "value_function.npz")
        value_data = {"values": unbiased_values, "unbiased_values": unbiased_values, "v_function_unbiased": unbiased_values, "dfa_states": dfa_states, "accepting_dfa_states": np.asarray(sorted(abstract_mdp.automaton.accepting_states), dtype=np.int64), "failure_dfa_states": np.asarray(sorted(abstract_mdp.automaton.failure_states), dtype=np.int64), "width": np.int64(abstract_mdp.width), "height": np.int64(abstract_mdp.height), "gamma": np.float64(abstract_mdp.gamma), "goal_reward": np.float64(abstract_mdp.goal_reward), "level_name": np.asarray(abstract_mdp.level_name), "solution_algorithm": np.asarray(abstract_mdp.solution_algorithm), "value_function_method": np.asarray(abstract_mdp.value_function_method), "has_biased_values": np.bool_(abstract_mdp.biased_v_star is not None)}
        if abstract_mdp.unbiased_q is not None:
            q_values = np.full((len(dfa_states), abstract_mdp.height, abstract_mdp.width, len(abstract_mdp.actions)), np.nan, dtype=np.float64)
            for state, action_values in zip(abstract_mdp.states, abstract_mdp.unbiased_q):
                x, y, q = state
                q_values[q_indices[q], y, x, :] = np.asarray(action_values, dtype=np.float64)
            if np.isnan(q_values).any():
                raise ValueError(f"{abstract_mdp.level_name} unbiased Q-function is missing one or more product states")
            value_data["q_function_unbiased"] = q_values
        if abstract_mdp.biased_v_star is not None:
            value_data["biased_values"] = dense_values(abstract_mdp.biased_v_star, "biased")
        np.savez_compressed(value_path, **value_data)
        generated_files.append(value_path)
        saved_variants = "unbiased Q/V" if abstract_mdp.unbiased_q is not None else "unbiased V"
        if abstract_mdp.biased_v_star is not None:
            saved_variants += " + biased V"
        print(f" -> Abstract checkpoint saved to: {value_path} ({saved_variants}; V[q_index, y, x], Q[q_index, y, x, action])")
    return generated_files


def save_abstract_learning_curves(multilevel_mdp, output_root=None, smoothing_window=100):
    """Save the reward and epsilon curve for every Q-learned abstraction."""
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be greater than zero")
    output_root = output_root or os.path.join("img", "abstract_learning")
    generated_files = []
    for level_number, abstract_mdp in enumerate(multilevel_mdp.levels, start=1):
        learning_history = abstract_mdp.learning_history
        if learning_history is None:
            continue
        level_directory = os.path.join(output_root, f"level{level_number}")
        os.makedirs(level_directory, exist_ok=True)
        for obsolete_name in ("convergence.png", "convergence_data.npz"):
            obsolete_path = os.path.join(level_directory, obsolete_name)
            if os.path.isfile(obsolete_path):
                os.remove(obsolete_path)
        data_path = os.path.join(level_directory, "reward_epsilon_data.npz")
        np.savez_compressed(data_path, **{key: np.asarray(values, dtype=np.float64) for key, values in learning_history.items()})

        episodes = np.asarray(learning_history["episodes"], dtype=np.float64)

        reward_figure, reward_axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        epsilon_axis = reward_axis.twinx()
        if abstract_mdp.upper_level_mdp is not None:
            reward_axis.plot(episodes, _trailing_mean(learning_history["biased_episode_reward"], smoothing_window), color=LEARNING_REWARD_COLOR, linewidth=1.7, label="Learning reward (task + shaping)")
            reward_axis.plot(episodes, _trailing_mean(learning_history["unbiased_episode_reward"], smoothing_window), color=TASK_REWARD_COLOR, linewidth=1.6, label="Task reward")
        else:
            reward_axis.plot(episodes, _trailing_mean(learning_history["unbiased_episode_reward"], smoothing_window), color=TASK_REWARD_COLOR, linewidth=1.7, label="Task reward")
        epsilon_axis.plot(episodes, learning_history["epsilon"], color=EPSILON_COLOR, linestyle="--", linewidth=1.4, label="Epsilon")
        reward_axis.set_xlabel("#Episode")
        reward_axis.set_ylabel("Episode reward")
        epsilon_axis.set_ylabel("Epsilon")
        epsilon_axis.set_ylim(0.0, 1.0)
        epsilon_axis.grid(False)
        _style_paper_axis(reward_axis)
        reward_handles, reward_labels = reward_axis.get_legend_handles_labels()
        epsilon_handles, epsilon_labels = epsilon_axis.get_legend_handles_labels()
        _place_legend_above(reward_figure, reward_handles + epsilon_handles, reward_labels + epsilon_labels, max_columns=3)
        reward_path = os.path.join(level_directory, "reward_epsilon.png")
        reward_figure.savefig(reward_path, dpi=300, bbox_inches="tight")
        plt.close(reward_figure)
        generated_files.extend([reward_path, data_path])
        print(f" -> Abstract reward/epsilon curve saved to: {reward_path}")
    return generated_files

# ==============================
# Training diagnostics and learning curves
# ==============================


def _prepare_plot_path(filename):
    """Create the destination directory, if one was provided."""
    output_directory = os.path.dirname(os.fspath(filename))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)


def _style_paper_axis(axis, grid_axis="y"):
    """Apply the unobtrusive axis treatment used by all paper figures."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    axis.set_axisbelow(True)
    axis.grid( axis=grid_axis, color="#d9d9d9", linestyle="-", linewidth=0.6, alpha=0.8, )


def _trailing_mean(values, window_size):
    """Return the mean of the current and previous N-1 episodes."""
    return (
        pd.Series(np.asarray(values, dtype=np.float64)) .rolling(window=window_size, min_periods=1, center=False) .mean() .to_numpy()
    )


def _place_legend_above(figure, handles, labels, max_columns=4):
    """Place a variable-size legend above an axes without shrinking the plot."""
    if not labels:
        return
    columns = max(1, min(len(labels), max_columns))
    rows = math.ceil(len(labels) / columns)
    base_width, base_height = 7.2, 4.4
    extra_height = 0.32 * (rows - 1)
    total_height = base_height + extra_height
    figure.set_layout_engine(None)
    figure.set_size_inches(base_width, total_height, forward=True)
    figure.subplots_adjust(left=0.11, right=0.89, bottom=0.14 * base_height / total_height, top=0.82 * base_height / total_height)
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.835 * base_height / total_height), ncol=columns, frameon=False)


def plot_training_variance(reward_histories, window_size=100, title="Training Performance Across Seeds", filename="img/training_variance.png", label="Learning reward", epsilon_histories=None, epsilon_labels=None):
    """Plot aggregate rewards with variance and optional raw epsilon curves."""
    runs = np.asarray(reward_histories, dtype=np.float64)
    if runs.ndim == 1:
        runs = runs[np.newaxis, :]
    if runs.ndim != 2 or runs.shape[0] == 0 or runs.shape[1] == 0:
        raise ValueError("reward_histories must have shape (num_seeds, episodes)")
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")

    smoothed_runs = (
        pd.DataFrame(runs.T) .rolling(window=window_size, min_periods=1, center=False) .mean() .to_numpy()
        .T
    )
    mean_reward = np.mean(smoothed_runs, axis=0)
    std_reward = np.std(smoothed_runs, axis=0)
    episodes = np.arange(1, runs.shape[1] + 1)

    _prepare_plot_path(filename)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(episodes, mean_reward, color=LEARNING_REWARD_COLOR, linewidth=1.7, label=f"Mean {label}")
    ax.fill_between( episodes, mean_reward - std_reward, mean_reward + std_reward, color=LEARNING_REWARD_COLOR, alpha=0.18, linewidth=0, label="±1 SD across seeds", )
    ax.set_xlabel("#Episode")
    ax.set_ylabel(label)
    _style_paper_axis(ax)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if epsilon_histories is not None:
        epsilon_runs = np.asarray(epsilon_histories, dtype=np.float64)
        if epsilon_runs.ndim == 1:
            epsilon_curves = epsilon_runs[np.newaxis, :]
        elif epsilon_runs.ndim == 2:
            if epsilon_runs.shape[0] == runs.shape[0]:
                epsilon_curves = np.mean(epsilon_runs, axis=0, keepdims=True)
            else:
                epsilon_curves = epsilon_runs
        elif epsilon_runs.ndim == 3:
            epsilon_curves = np.mean(epsilon_runs, axis=0)
        else:
            raise ValueError( "epsilon_histories must have shape (episodes), " "(seeds, episodes), or (seeds, epsilon_series, episodes)" )
        if epsilon_curves.shape[1] != runs.shape[1]:
            raise ValueError("epsilon_histories must contain one value per episode")
        if epsilon_labels is not None and len(epsilon_labels) != len(epsilon_curves):
            raise ValueError("epsilon_labels must match the number of epsilon curves")

        epsilon_axis = ax.twinx()
        for index, epsilon_curve in enumerate(epsilon_curves):
            if epsilon_labels is not None:
                epsilon_label = f"Epsilon q={epsilon_labels[index]}"
            elif len(epsilon_curves) == 1:
                epsilon_label = "Epsilon"
            else:
                epsilon_label = f"Epsilon {index + 1}"
            epsilon_axis.plot( episodes, epsilon_curve, color=EPSILON_COLOR, linestyle=EPSILON_LINESTYLES[index % len(EPSILON_LINESTYLES)], linewidth=1.4, label=epsilon_label, )
        epsilon_axis.set_ylabel("Epsilon")
        epsilon_axis.set_ylim(0.0, 1.0)
        epsilon_axis.grid(False)
        epsilon_axis.spines["top"].set_visible(True)
        epsilon_axis.spines["right"].set_visible(True)
        epsilon_handles, epsilon_legend_labels = epsilon_axis.get_legend_handles_labels()
        legend_handles += epsilon_handles
        legend_labels += epsilon_legend_labels

    _place_legend_above(fig, legend_handles, legend_labels, max_columns=4)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n>>> Training variance plot saved to: {filename}")
    plt.close(fig)


def plot_tabular_training_diagnostics(table_sizes, updated_state_actions, state_action_coverage, positive_updates, filename="img/tabular_training_diagnostics.png"):
    """Plot sparse Q-table growth and coverage across training seeds."""
    table_runs = np.atleast_2d(np.asarray(table_sizes, dtype=np.float64))
    pair_runs = np.atleast_2d(np.asarray(updated_state_actions, dtype=np.float64))
    coverage_runs = np.atleast_2d(np.asarray(state_action_coverage, dtype=np.float64))
    positive_runs = np.atleast_2d(np.asarray(positive_updates, dtype=np.float64))
    if table_runs.shape != pair_runs.shape or table_runs.shape != coverage_runs.shape or table_runs.shape != positive_runs.shape:
        raise ValueError("tabular metrics must share shape (num_seeds, episodes)")
    _prepare_plot_path(filename)
    episodes = np.arange(1, table_runs.shape[1] + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), constrained_layout=True)
    axes[0].plot(episodes, np.nanmean(table_runs, axis=0), color=LEARNING_REWARD_COLOR, label="Q-table states")
    axes[0].plot(episodes, np.nanmean(pair_runs, axis=0), color=TASK_REWARD_COLOR, label="Updated state-action pairs")
    axes[0].plot(episodes, np.nanmean(positive_runs, axis=0), color=EPSILON_COLOR, label="Positive-reward updates")
    axes[0].set_xlabel("#Training episode")
    axes[0].set_ylabel("Cumulative count")
    axes[0].legend()
    axes[1].plot(episodes, np.nanmean(coverage_runs, axis=0), color=LEARNING_REWARD_COLOR, label="State-action coverage")
    axes[1].set_xlabel("#Training episode")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    for axis in axes:
        _style_paper_axis(axis)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n>>> Tabular diagnostics plot saved to: {filename}")
    plt.close(fig)


def plot_evaluation_performance(
    evaluation_steps,
    success_rates,
    task_rewards,
    episode_lengths,
    filename="img/evaluation_performance.png",
    title="Greedy Evaluation Performance",
):
    """Plot greedy-evaluation metrics, aggregating multiple seeds when present."""
    steps = np.asarray(evaluation_steps)
    success_runs = np.atleast_2d(np.asarray(success_rates, dtype=np.float64))
    reward_runs = np.atleast_2d(np.asarray(task_rewards, dtype=np.float64))
    length_runs = np.atleast_2d(np.asarray(episode_lengths, dtype=np.float64))
    if steps.ndim == 2:
        if not np.all(steps == steps[0]):
            raise ValueError("evaluation steps must be identical across seeds")
        steps = steps[0]
    if steps.ndim != 1 or len(steps) == 0:
        raise ValueError("evaluation_steps must be a non-empty vector")
    expected_shape = success_runs.shape
    if reward_runs.shape != expected_shape or length_runs.shape != expected_shape:
        raise ValueError("evaluation metrics must share shape (num_seeds, evaluations)")
    if expected_shape[1] != len(steps):
        raise ValueError("evaluation metrics must contain one value per evaluation step")

    _prepare_plot_path(filename)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True, constrained_layout=True)
    series = (
        (success_runs, "Success rate", (0.0, 1.0), LEARNING_REWARD_COLOR),
        (reward_runs, "Mean task reward", None, TASK_REWARD_COLOR),
        (length_runs, "Mean episode length", None, SERIES_COLORS[2]),
    )
    for axis, (runs, ylabel, limits, color) in zip(axes, series):
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)
        axis.plot(steps, mean, color=color, linewidth=1.8)
        if runs.shape[0] > 1:
            lower = mean - std
            upper = mean + std
            if limits is not None:
                lower = np.clip(lower, *limits)
                upper = np.clip(upper, *limits)
            axis.fill_between(steps, lower, upper, color=color, alpha=0.18, linewidth=0)
        axis.set_ylabel(ylabel)
        if limits is not None:
            axis.set_ylim(*limits)
        _style_paper_axis(axis)
    axes[-1].set_xlabel("#Training episode")
    fig.suptitle(title)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n>>> Evaluation performance plot saved to: {filename}")
    plt.close(fig)

def plot_buffer_variance(buffer_histories_runs, window_size=100, filename="img/buffer_variance.png", state_labels=None, title="Replay Buffer Composition Across Seeds"):
    """Plot mean replay-buffer fractions with a ±1 std band across seeds."""
    runs = np.asarray(buffer_histories_runs, dtype=np.float64)
    if runs.ndim == 2:
        runs = runs[np.newaxis, ...]
    if runs.ndim != 3 or 0 in runs.shape:
        raise ValueError( "buffer_histories_runs must have shape (num_seeds, num_states, episodes)" )
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")

    smoothed_runs = np.empty_like(runs, dtype=np.float64)
    for seed_index in range(runs.shape[0]):
        for state_index in range(runs.shape[1]):
            smoothed_runs[seed_index, state_index] = (
                pd.Series(runs[seed_index, state_index]) .rolling(window=window_size, min_periods=1, center=False) .mean() .to_numpy()
            )

    mean_fractions = np.mean(smoothed_runs, axis=0)
    std_fractions = np.std(smoothed_runs, axis=0)
    episodes = np.arange(1, runs.shape[2] + 1)
    colors = [SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(runs.shape[1])]
    _prepare_plot_path(filename)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for state_index, color in enumerate(colors):
        state_label = state_labels[state_index] if state_labels is not None else state_index
        mean = mean_fractions[state_index]
        std = std_fractions[state_index]
        ax.plot(episodes, mean, color=color, linewidth=1.7, label=f"DFA state q={state_label}")
        ax.fill_between( episodes, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), color=color, alpha=0.16, linewidth=0, )

    ax.set_xlabel("#Episode")
    ax.set_ylabel("Buffer fraction")
    ax.set_ylim(0, 1.0)
    _style_paper_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    _place_legend_above(fig, handles, labels, max_columns=3)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n>>> Buffer variance plot saved to: {filename}")
    plt.close(fig)

def plot_buffer_fractions(buffer_histories, window_size=100, filename="img/buffer_fractions.png", state_labels=None, title="Replay Buffer Composition"):
    """
    Plots the replay buffer composition for N phases dynamically.
    """
    _prepare_plot_path(filename)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    x_axis = np.arange(1, len(buffer_histories[0]) + 1)
    
    colors = [SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(len(buffer_histories))]
    for idx, history in enumerate(buffer_histories):
        ma = _trailing_mean(history, window_size)
        state_label = state_labels[idx] if state_labels is not None else idx
        ax.plot(x_axis, ma, color=colors[idx], linewidth=1.7, label=f'DFA state q={state_label}')
    
    ax.set_xlabel("#Episode")
    ax.set_ylabel("Buffer fraction")
    ax.set_ylim(0, 1.0)
    _style_paper_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    _place_legend_above(fig, handles, labels, max_columns=3)
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_shaping_reward_breakdown(true_rewards, total_rewards, eps_histories, window_size=100, filename="img/shaping_reward_breakdown.png", title="Shaping Agent Reward Analysis"):
    """Plot rewards and epsilon together using independent, well-defined axes."""
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")
    true_rewards = np.asarray(true_rewards, dtype=np.float64)
    total_rewards = np.asarray(total_rewards, dtype=np.float64)
    if true_rewards.ndim != 1 or total_rewards.shape != true_rewards.shape:
        raise ValueError("true_rewards and total_rewards must be equally sized vectors")

    exploration_runs = np.asarray(eps_histories, dtype=np.float64)
    if exploration_runs.ndim == 1:
        exploration_runs = exploration_runs[np.newaxis, :]
    elif exploration_runs.ndim == 2 and exploration_runs.shape[1] != len(true_rewards):
        if exploration_runs.shape[0] == len(true_rewards):
            exploration_runs = exploration_runs.T
    if exploration_runs.ndim != 2 or exploration_runs.shape[1] != len(true_rewards):
        raise ValueError("eps_histories must contain one value per episode")

    _prepare_plot_path(filename)
    figure, reward_axis = plt.subplots( figsize=(7.2, 4.4), constrained_layout=True, )
    epsilon_axis = reward_axis.twinx()
    episodes = np.arange(1, len(true_rewards) + 1)
    reward_axis.plot( episodes, _trailing_mean(true_rewards, window_size), color=TASK_REWARD_COLOR, linewidth=1.6, label="Task reward", )
    reward_axis.plot( episodes, _trailing_mean(total_rewards, window_size), color=LEARNING_REWARD_COLOR, linewidth=1.7, label="Learning reward (goal + shaping)", )
    reward_axis.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=0.7, alpha=0.7)
    reward_axis.set_xlabel("#Episode")
    reward_axis.set_ylabel("Episode Reward")
    _style_paper_axis(reward_axis)

    for index, history in enumerate(exploration_runs):
        phase = "Goal" if index == len(exploration_runs) - 1 else f"WP {index + 1}"
        label = "Epsilon" if len(exploration_runs) == 1 else f"Epsilon q={index} ({phase})"
        epsilon_axis.plot( episodes, history, color=EPSILON_COLOR, linestyle=EPSILON_LINESTYLES[index % len(EPSILON_LINESTYLES)], linewidth=1.4, label=label, )
    epsilon_axis.set_ylabel("Epsilon")
    epsilon_axis.set_ylim(0.0, 1.0)
    epsilon_axis.grid(False)
    epsilon_axis.spines["top"].set_visible(True)
    epsilon_axis.spines["right"].set_visible(True)

    reward_lines, reward_labels = reward_axis.get_legend_handles_labels()
    epsilon_lines, epsilon_labels = epsilon_axis.get_legend_handles_labels()
    _place_legend_above(figure, reward_lines + epsilon_lines, reward_labels + epsilon_labels, max_columns=4)
    figure.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
