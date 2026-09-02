"""Evaluate LTLf or cyclic-waypoint LunarLander policies."""

# ==============================
# Standard library imports
# ==============================

import argparse
import json
import time
from pathlib import Path

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from abstraction import AbstractionConfig
from abstract_mdps import LTLfWaypointMDP, build_task_automaton
from agent import DuelingQNetwork, QNetwork, TabularQLearner
from grid_overlay import (
    abstract_cell_to_pixel,
    draw_abstract_grid,
    geometry_from_env,
)
from spatial_regions import load_task_propositions, rasterize_regions
from utils import (
    LEARNING_REWARD_COLOR,
    RAW_DATA_COLOR,
    SERIES_COLORS,
    phi_mapping_sequential,
)


# ==============================
# Paths and generic helpers
# ==============================

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = SCRIPT_DIR.parent
EXPERIMENTS_DIR = FRAMEWORK_DIR / "results"


def moving_average(data, window_size):
    """Return a moving average, or the original values when the window is larger."""
    values = np.asarray(data, dtype=np.float64)
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size) / window_size, mode="valid")


def _resolve_policy_path(policy, policy_dir):
    """Accept explicit paths as well as filenames relative to the policy directory."""
    supplied_path = Path(policy).expanduser()
    if supplied_path.is_file():
        return supplied_path.resolve()

    policy_root = Path(policy_dir).expanduser()
    for policy_path in (
        policy_root / supplied_path,
        policy_root / "best" / supplied_path,
        policy_root / "last" / supplied_path,
    ):
        if policy_path.is_file():
            return policy_path.resolve()

    raise FileNotFoundError(f"Policy '{policy}' not found either as an explicit path or under '{policy_dir}'.")


def _load_state_dict(policy_path, device):
    """Load both plain state dictionaries and common wrapped checkpoints."""
    checkpoint = torch.load(policy_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("policy_state_dict", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _abstract_position(observation, q, grid_w, grid_h):
    """Map an environment observation to its abstract grid coordinates."""
    x, y, _ = phi_mapping_sequential(observation, q, grid_w, grid_h)
    return x, y


# ==============================
# Policy evaluation
# ==============================

def evaluate_policy(policy, policy_dir, episodes, render, task_config, regions, goal_reward, grid_w, grid_h, seed, trace_episodes=0, network_type="standard", no_limit=False):
    """Load and evaluate one policy using the same task semantics as training."""
    # Rebuild the same automaton and abstract MDP used during training.
    policy_path = _resolve_policy_path(policy, policy_dir)
    policy_name = policy_path.name
    automaton = build_task_automaton(task_config)
    abstract_mdp = LTLfWaypointMDP(regions=regions, ltlf_automaton=automaton, width=grid_w, height=grid_h, goal_reward=goal_reward)
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}

    # Create the environment and a network with one extra feature per DFA state.
    render_mode = "human" if render else ("rgb_array" if trace_episodes else None)
    environment_options = {"continuous": False, "render_mode": render_mode}
    if no_limit:
        environment_options["max_episode_steps"] = 5000
    env = gym.make("LunarLander-v3", **environment_options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if network_type not in {"standard", "dueling"}:
        raise ValueError("network_type must be one of: standard, dueling")
    network = None
    tabular_learner = None

    # Load the trained parameters before starting any episode.
    try:
        if policy_path.suffix.lower() == ".pkl":
            tabular_learner = TabularQLearner(env=env, num_phases=len(automaton_states), random_seed=seed)
            tabular_learner.load_policy(policy_path)
        else:
            network_cls = DuelingQNetwork if network_type == "dueling" else QNetwork
            network = network_cls(env.observation_space.shape[0] + len(automaton_states), env.action_space.n).to(device)
            network.load_state_dict(_load_state_dict(policy_path, device))
            network.eval()
    except Exception:
        env.close()
        raise

    task_returns = []
    environment_returns = []
    episode_lengths = []
    successes = 0
    failures = 0
    completed_cycles = []
    state_reach_counts = {q: 0 for q in automaton_states}
    grid_traces = []
    trace_frames = []
    trace_geometries = []

    # Run every requested episode sequentially.
    try:
        for episode in range(episodes):
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)
            tracing = episode < trace_episodes
            if tracing:
                trace_frames.append(env.render())
                trace_geometries.append(geometry_from_env(env))
                initial_cell = _abstract_position(observation, automaton.get_initial_q(), grid_w, grid_h)
                cell_trace = [initial_cell]

            # Training consumes the valuation at s0 before choosing the first action.
            initial_q = automaton.get_initial_q()
            initial_truth_assignment = abstract_mdp.get_environment_truth_assignment(observation)
            q = automaton.get_next_q(initial_q, initial_truth_assignment)
            if q not in state_to_index:
                raise RuntimeError(f"DFA returned unknown state {q!r}")

            reached_states = {q}
            success = automaton.is_goal_reached(q)
            failed = automaton.is_failure(q)
            terminated = truncated = False
            environment_return = 0.0
            steps = 0
            episode_completed_cycles = 0

            while not (failed or terminated or truncated or (success and not automaton.is_continuing)):
                # Append the current DFA state as a one-hot vector.
                one_hot = np.zeros(len(automaton_states), dtype=np.float32)
                one_hot[state_to_index[q]] = 1.0
                augmented_state = np.concatenate((observation, one_hot)).astype(np.float32)

                # Evaluation is greedy: always select the action with maximum Q-value.
                if tabular_learner is not None:
                    action = tabular_learner.greedy_action(augmented_state)
                else:
                    with torch.inference_mode():
                        state_tensor = torch.as_tensor(augmented_state, device=device).unsqueeze(0)
                        action = network(state_tensor).argmax(dim=1).item()

                next_observation, env_reward, terminated, truncated, _ = env.step(action)
                environment_return += float(env_reward)
                steps += 1

                # Advance the DFA using the propositions true in the arrival state.
                x, y = _abstract_position(next_observation, q, grid_w, grid_h)
                if tracing and (x, y) != cell_trace[-1]:
                    cell_trace.append((x, y))
                truth_assignment = abstract_mdp.get_environment_truth_assignment(next_observation)
                automaton_step = automaton.advance(q, truth_assignment)
                next_q = automaton_step.next_state
                if next_q not in state_to_index:
                    raise RuntimeError(f"DFA returned unknown state {next_q!r}")

                # Report every effective DFA transition during evaluation.
                if next_q != q:
                    if automaton_step.accepted:
                        print(f"[{policy_name} | Episode {episode + 1}] DFA transition {q} -> {next_q}: final goal reached.")
                    elif automaton_step.failed:
                        print(f"[{policy_name} | Episode {episode + 1}] DFA transition {q} -> {next_q}: irreversible task failure.")
                    else:
                        print(f"[{policy_name} | Episode {episode + 1}] DFA transition {q} -> {next_q}: intermediate waypoint reached.")
                if automaton_step.completed_cycle:
                    episode_completed_cycles += 1
                    print(f"[{policy_name} | Episode {episode + 1}] cycle {episode_completed_cycles} completed; continuing from q={next_q}.")

                reached_states.add(next_q)

                observation = next_observation
                q = next_q
                success = success or automaton_step.succeeded
                failed = failed or automaton_step.failed

                if render:
                    time.sleep(0.02)

            # Store episode-level metrics and count every DFA state reached at least once.
            successes += int(success)
            failures += int(failed)
            completed_cycles.append(episode_completed_cycles)
            for reached_q in reached_states:
                state_reach_counts[reached_q] += 1
            task_returns.append(float(goal_reward) * (episode_completed_cycles if automaton.is_continuing else int(success)))
            environment_returns.append(environment_return)
            episode_lengths.append(steps)
            if tracing:
                grid_traces.append(cell_trace)
    finally:
        env.close()

    return {
        "policy": policy_name,
        "path": str(policy_path),
        "task_returns": task_returns,
        "environment_returns": environment_returns,
        "episode_lengths": episode_lengths,
        "successes": successes,
        "failures": failures,
        "completed_cycles": completed_cycles,
        "state_reach_counts": state_reach_counts,
        "grid_traces": grid_traces,
        "trace_frames": trace_frames,
        "trace_geometries": trace_geometries,
    }


# ==============================
# Plotting helpers
# ==============================

def _safe_stem(name):
    """Create a filesystem-safe plot stem from a checkpoint filename."""
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in Path(name).stem)


def plot_policy(result, window_size, output_dir):
    """Plot Gym returns for one policy."""
    returns = result["environment_returns"]
    smooth = moving_average(returns, window_size)

    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    episodes = np.arange(1, len(returns) + 1)
    axis.plot(episodes, returns, alpha=0.28, color=RAW_DATA_COLOR, linewidth=0.8, label="Raw Gym return")
    start = window_size - 1 if len(returns) >= window_size else 0
    axis.plot(np.arange(start + 1, start + len(smooth) + 1), smooth, color=LEARNING_REWARD_COLOR, linewidth=1.7, label=f"Trailing mean (N={window_size})")
    axis.set_xlabel("#Episode")
    axis.set_ylabel("Gym return")
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    output_path = output_dir / f"eval_{_safe_stem(result['policy'])}.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_comparison(results, window_size, output_dir):
    """Plot smoothed Gym returns for multiple policies."""
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for index, result in enumerate(results):
        returns = result["environment_returns"]
        smooth = moving_average(returns, window_size)
        start = window_size - 1 if len(returns) >= window_size else 0
        axis.plot(np.arange(start + 1, start + len(smooth) + 1), smooth, color=SERIES_COLORS[index % len(SERIES_COLORS)], linewidth=1.7, label=result["policy"])
    axis.set_xlabel("#Episode")
    axis.set_ylabel("Gym return")
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=min(len(results), 3), frameon=False)
    output_path = output_dir / "policy_comparison.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_grid_traces(result, regions, grid_w, grid_h, output_dir):
    """Save one abstract-grid path image for every recorded episode."""
    output_paths = []
    trace_data = zip( result["grid_traces"], result["trace_frames"], result["trace_geometries"], )
    for episode_index, (cells, frame, geometry) in enumerate(trace_data, start=1):
        figure = draw_abstract_grid( frame=frame, geometry=geometry, grid_w=grid_w, grid_h=grid_h, regions=regions, title=f"Agent Abstract-Cell Trace — Episode {episode_index}", )
        axis = figure.axes[0]
        points = [
            abstract_cell_to_pixel(x, y, grid_w, grid_h, geometry)
            for x, y in cells
        ]
        if points:
            pixel_x, pixel_y = zip(*points)
            axis.plot( pixel_x, pixel_y, color="#00e5ff", linewidth=2.8, marker="o", markersize=5, label="Visited-cell path", zorder=4, )
            for change_index, ((cell_x, cell_y), (point_x, point_y)) in enumerate( zip(cells, points) ):
                axis.annotate( str(change_index), (point_x, point_y), ha="center", va="center", fontsize=7, fontweight="bold", color="black", zorder=7, )
        axis.legend( loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True, )
        figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        output_path = output_dir / (
            f"grid_trace_{_safe_stem(result['policy'])}_episode_{episode_index}.png"
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def format_region_trace(cells, regions, grid_w, grid_h):
    """Report first contact with each rasterized region in a cell trace."""
    first_visit = {}
    for index, cell in enumerate(cells):
        first_visit.setdefault(tuple(cell), index)
    region_cells = rasterize_regions(regions, grid_w, grid_h)
    statuses = []
    for name, occupied_cells in region_cells.items():
        visits = [first_visit[cell] for cell in occupied_cells if cell in first_visit]
        statuses.append(f"{name}=cell-contact@{min(visits)}" if visits else f"{name}=missed")
    return ", ".join(statuses)


# ==============================
# Command-line interface
# ==============================

def _positive_int(value):
    """Parse and validate a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _select_files_graphically(policy_dir, config_path):
    """Select policy checkpoints and the experiment configuration with native dialogs."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError( "The graphical selector requires tkinter. Install python3-tk or pass " "the policy paths and --config from the command line." ) from error

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError( "The graphical selector could not be opened. Make sure a desktop " "session is available, or use the command-line arguments." ) from error
    root.withdraw()
    root.update()

    try:
        initial_directory = EXPERIMENTS_DIR if EXPERIMENTS_DIR.is_dir() else SCRIPT_DIR
        policies = filedialog.askopenfilenames( parent=root, title="Select one or more policy files", initialdir=str(initial_directory), filetypes=[ ("Policy checkpoints", "*.pt *.pth *.ckpt *.pkl"), ("All files", "*"), ], )
        if not policies:
            raise RuntimeError("No policy file was selected.")

        config = filedialog.askopenfilename( parent=root, title="Select trajectory.json", initialdir=str(initial_directory), initialfile=Path(config_path).name, filetypes=[ ("JSON files", "*.json"), ("All files", "*"), ], )
        if not config:
            raise RuntimeError("No trajectory configuration was selected.")
    finally:
        root.destroy()

    return list(policies), Path(config)


def parse_args():
    """Build and parse the evaluator command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LTLf-guided neural or tabular policies for LunarLander.")
    parser.add_argument( "policies", nargs="*", help="Checkpoint filenames or explicit checkpoint paths. If omitted, graphical file selectors are opened.", )
    parser.add_argument("--config", type=Path, default=FRAMEWORK_DIR / "config" / "trajectory.json", help="Experiment JSON configuration.")
    parser.add_argument( "--abstraction-config", type=Path, default=FRAMEWORK_DIR / "config" / "abstraction.json", help="Grid hierarchy; evaluation uses its level1 dimensions.", )
    parser.add_argument("--policy-dir", type=Path, default=FRAMEWORK_DIR / "results", help="Directory used to resolve checkpoint filenames.")
    parser.add_argument("--gui", action="store_true", help="Select policies and trajectory.json using graphical dialogs.")
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--window", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-limit", action="store_true", help="Increase the environment episode limit to 5000 steps.")
    parser.add_argument( "--trace-grid", action="store_true", help="Save the sequence of abstract cells visited during evaluation.", )
    parser.add_argument( "--trace-episodes", type=_positive_int, default=1, help="Number of episodes to trace when --trace-grid is enabled (default: 1).", )
    parser.add_argument("--output-dir", type=Path, default=FRAMEWORK_DIR / "results" / "evaluation")
    parser.add_argument( "--network-type", choices=["standard", "dueling"], default="standard", help="Q-network architecture used by the checkpoint.", )
    return parser.parse_args()


# ==============================
# Main program
# ==============================

def main():
    """Load the configuration, evaluate the policies, and generate the plots."""
    args = parse_args()
    if args.render and args.trace_grid:
        raise SystemExit( "--render and --trace-grid cannot be used together because Gymnasium " "requires a single render mode. Run them as separate evaluations." )

    # Open native file dialogs when requested or when no policy was supplied.
    if args.gui or not args.policies:
        try:
            args.policies, args.config = _select_files_graphically(args.policy_dir, args.config)
        except RuntimeError as error:
            raise SystemExit(f"Selection cancelled: {error}") from error

    # Load the LTLf task shared with the trainer.
    with args.config.expanduser().open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    abstraction_config = AbstractionConfig.load(args.abstraction_config.expanduser())

    regions, _, task_propositions = load_task_propositions(config.get("regions"), config.get("predicates"))
    grid_w = abstraction_config.primary.width
    grid_h = abstraction_config.primary.height
    goal_reward = float(config.get("goal_reward", 10000.0))

    # Evaluate policies one at a time to keep rendering and output deterministic.
    results = []
    for policy in args.policies:
        traced_episodes = min(args.trace_episodes, args.episodes) if args.trace_grid else 0
        result = evaluate_policy( policy, args.policy_dir, args.episodes, args.render, config, task_propositions, goal_reward, grid_w, grid_h, args.seed, trace_episodes=traced_episodes, network_type=args.network_type, no_limit=args.no_limit, )
        results.append(result)

    # Print the summary and create one plot for each evaluated policy.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        success_rate = result["successes"] / args.episodes
        failure_rate = result["failures"] / args.episodes
        mean_gym_return = np.mean(result["environment_returns"])
        mean_length = np.mean(result["episode_lengths"])
        cycle_summary = f", mean cycles={np.mean(result['completed_cycles']):.2f}" if "waypoint_cycle" in config or config.get("task_type") == "cyclic_waypoints" else ""
        reached = ", ".join(f"q={q}: {count}/{args.episodes}" for q, count in result["state_reach_counts"].items())
        print(f"[{result['policy']}] success={success_rate:.1%}, failure={failure_rate:.1%}{cycle_summary}, mean Gym return={mean_gym_return:.2f}, mean length={mean_length:.1f} | reached: {reached}")
        print(f"Plot saved to: {plot_policy(result, args.window, args.output_dir)}")
        if args.trace_grid:
            trace_paths = plot_grid_traces( result, regions, grid_w, grid_h, args.output_dir )
            for episode_index, (cells, trace_path) in enumerate( zip(result["grid_traces"], trace_paths), start=1 ):
                region_status = format_region_trace(cells, regions, grid_w, grid_h)
                print( f"Grid trace episode {episode_index}: {region_status} | " f"saved to: {trace_path}" )

    # Add a combined comparison when more than one policy was requested.
    if len(results) > 1:
        print(f"Comparison saved to: {plot_comparison(results, args.window, args.output_dir)}")


if __name__ == "__main__":
    main()
