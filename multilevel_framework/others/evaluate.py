"""Evaluate LTLf or cyclic-waypoint LunarLander policies."""

# ==============================
# Standard library imports
# ==============================

import argparse
import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = SCRIPT_DIR.parent
SRC_DIR = FRAMEWORK_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from abstract_mdps import LTLfWaypointMDP, build_task_automaton
from agent import DuelingQNetwork, QNetwork, TabularQLearner
from spatial_regions import load_task_propositions
from utils import (
    LEARNING_REWARD_COLOR,
    RAW_DATA_COLOR,
    SERIES_COLORS,
)


# ==============================
# Paths and generic helpers
# ==============================

EXPERIMENTS_DIR = FRAMEWORK_DIR / "results"
SEEDED_POLICY_RE = re.compile(
    r"^(best|last)_policy(?:_seed_(-?\d+))?\.(?:pt|pth|ckpt|pkl)$",
    re.IGNORECASE,
)


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


# ==============================
# Policy evaluation
# ==============================

def evaluate_policy(policy, policy_dir, episodes, render, task_config, regions, goal_reward, seed, network_type="standard", no_limit=False, verbose=True):
    """Load and evaluate one policy using the same task semantics as training."""
    # Rebuild the same automaton and abstract MDP used during training.
    policy_path = _resolve_policy_path(policy, policy_dir)
    policy_name = policy_path.name
    automaton = build_task_automaton(task_config)
    abstract_mdp = LTLfWaypointMDP(regions=regions, ltlf_automaton=automaton, goal_reward=goal_reward)
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}

    # Create the environment and a network with one extra feature per DFA state.
    render_mode = "human" if render else None
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

    # Run every requested episode sequentially.
    try:
        for episode in range(episodes):
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)
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
                truth_assignment = abstract_mdp.get_environment_truth_assignment(next_observation)
                automaton_step = automaton.advance(q, truth_assignment)
                next_q = automaton_step.next_state
                if next_q not in state_to_index:
                    raise RuntimeError(f"DFA returned unknown state {next_q!r}")

                # Report every effective DFA transition during evaluation.
                if verbose and next_q != q:
                    if automaton_step.accepted:
                        print(f"[{policy_name} | Episode {episode + 1}] DFA transition {q} -> {next_q}: final goal reached.")
                    elif automaton_step.failed:
                        print(f"[{policy_name} | Episode {episode + 1}] DFA transition {q} -> {next_q}: irreversible task failure.")
                    else:
                        print(f"[{policy_name} | Episode {episode + 1}] DFA transition {q} -> {next_q}: intermediate waypoint reached.")
                if automaton_step.completed_cycle:
                    episode_completed_cycles += 1
                    if verbose:
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
    finally:
        env.close()

    return {
        "policy": policy_name,
        "path": str(policy_path),
        "task_description": automaton.formula_str,
        "is_continuing": automaton.is_continuing,
        "task_returns": task_returns,
        "environment_returns": environment_returns,
        "episode_lengths": episode_lengths,
        "successes": successes,
        "failures": failures,
        "completed_cycles": completed_cycles,
        "state_reach_counts": state_reach_counts,
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


def print_best_last_summary(results, task_description):
    """Print the task-appropriate aggregate metric across training seeds."""
    groups = {"best": [], "last": []}
    seeds = {"best": set(), "last": set()}
    is_continuing = bool(results[0]["is_continuing"])
    if any(bool(result["is_continuing"]) != is_continuing for result in results):
        raise ValueError("All policies in one aggregate evaluation must use the same task type")
    for result in results:
        match = SEEDED_POLICY_RE.fullmatch(result["policy"])
        if match is None:
            continue
        group, seed_text = match.groups()
        group = group.lower()
        episode_count = len(result["task_returns"])
        metric = (
            float(np.mean(result["completed_cycles"]))
            if is_continuing
            else float(result["successes"]) / episode_count
        )
        groups[group].append(metric)
        if seed_text is not None:
            seeds[group].add(int(seed_text))

    if not groups["best"] or not groups["last"]:
        return
    if len(groups["best"]) != len(groups["last"]):
        raise ValueError(
            "Best and last aggregate evaluation must contain the same number of policies: "
            f"best={len(groups['best'])}, last={len(groups['last'])}"
        )
    if seeds["best"] or seeds["last"]:
        if seeds["best"] != seeds["last"]:
            raise ValueError(
                "Best and last aggregate evaluation must contain the same training seeds: "
                f"best={sorted(seeds['best'])}, last={sorted(seeds['last'])}"
            )

    def aggregate(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.mean(values)), float(np.std(values))

    best_mean, best_std = aggregate(groups["best"])
    last_mean, last_std = aggregate(groups["last"])
    run_count = len(groups["best"])
    table_task = task_description.replace("|", "\\|")
    metric_label = "mean cycles per episode" if is_continuing else "success rate"
    if is_continuing:
        best_value = f"{best_mean:.3f} ± {best_std:.3f}"
        last_value = f"{last_mean:.3f} ± {last_std:.3f}"
    else:
        best_value = f"{best_mean:.2%} ± {best_std:.2%}"
        last_value = f"{last_mean:.2%} ± {last_std:.2%}"
    print("\n=== AGGREGATE POLICY EVALUATION ===")
    print(
        f"| Task | Training seeds | Best policy {metric_label} | "
        f"Last policy {metric_label} |"
    )
    print("|---|---:|---:|---:|")
    print(
        f"| {table_task} | {run_count} | "
        f"{best_value} | {last_value} |"
    )


# ==============================
# Command-line interface
# ==============================

def _positive_int(value):
    """Parse and validate a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _select_experiment_graphically():
    """Select an experiment and choose aggregate or individual-policy evaluation."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as error:
        raise RuntimeError(
            "The graphical selector requires tkinter. Install python3-tk or pass "
            "--experiment from the command line."
        ) from error

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError( "The graphical selector could not be opened. Make sure a desktop " "session is available, or use the command-line arguments." ) from error
    root.withdraw()
    root.update()

    try:
        initial_directory = EXPERIMENTS_DIR if EXPERIMENTS_DIR.is_dir() else SCRIPT_DIR
        experiment = filedialog.askdirectory(
            parent=root,
            title="Select an experiment directory",
            initialdir=str(initial_directory),
            mustexist=True,
        )
        if not experiment:
            raise RuntimeError("No experiment directory was selected.")
        aggregate = messagebox.askyesnocancel(
            "Evaluation mode",
            "Evaluate all best and last policies in aggregate?\n\n"
            "Yes: aggregate evaluation\n"
            "No: select one or more individual policies",
            parent=root,
        )
        if aggregate is None:
            raise RuntimeError("No evaluation mode was selected.")
        policies = []
        if not aggregate:
            experiment_policy_dir = Path(experiment) / "policy"
            initial_policy_dir = (
                experiment_policy_dir
                if experiment_policy_dir.is_dir()
                else Path(experiment)
            )
            policies = list(
                filedialog.askopenfilenames(
                    parent=root,
                    title="Select one or more policies",
                    initialdir=str(initial_policy_dir),
                    filetypes=(
                        ("Policy checkpoints", "*.pt *.pth *.ckpt *.pkl"),
                        ("All files", "*"),
                    ),
                )
            )
            if not policies:
                raise RuntimeError("No policy was selected.")
    finally:
        root.destroy()

    return Path(experiment), [Path(policy) for policy in policies]


def _resolve_experiment_directory(experiment):
    """Resolve either an experiment name below results/ or an explicit directory."""
    supplied = Path(experiment).expanduser()
    candidates = (supplied, EXPERIMENTS_DIR / supplied)
    for candidate in candidates:
        if candidate.is_dir():
            directory = candidate.resolve()
            if not (directory / "trajectory.json").is_file():
                raise FileNotFoundError(
                    f"Selected experiment does not contain trajectory.json: {directory}"
                )
            return directory
    raise FileNotFoundError(f"Experiment not found: {experiment}")


def _discover_policies(experiment_dir):
    """Return all best and last checkpoints stored by the trainer."""
    extensions = {".pt", ".pth", ".ckpt", ".pkl"}
    policies = []
    for category in ("best", "last"):
        directory = experiment_dir / "policy" / category
        if directory.is_dir():
            policies.extend(
                sorted(
                    path for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in extensions
                )
            )
    if not policies:
        raise FileNotFoundError(f"No best/last policies found under {experiment_dir / 'policy'}")
    return policies


def _is_aggregate_policy_selection(policies):
    """Return true when every selected checkpoint belongs to best/last groups."""
    groups = set()
    for policy in policies:
        match = SEEDED_POLICY_RE.fullmatch(Path(policy).name)
        if match is None:
            return False
        groups.add(match.group(1).lower())
    return groups == {"best", "last"}


def parse_args():
    """Build and parse the evaluator command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LTLf-guided neural or tabular policies for LunarLander.")
    parser.add_argument("policies", nargs="*", help="Optional checkpoint names or paths; by default all best/last policies are evaluated.")
    parser.add_argument("--experiment", help="Experiment name below results/ or an explicit experiment directory.")
    parser.add_argument("--gui", action="store_true", help="Select an experiment directory graphically.")
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--window", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-limit", action="store_true", help="Increase the environment episode limit to 5000 steps.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument( "--network-type", choices=["standard", "dueling"], default="standard", help="Q-network architecture used by the checkpoint.", )
    return parser.parse_args()


# ==============================
# Main program
# ==============================

def main():
    """Load the configuration, evaluate the policies, and generate the plots."""
    args = parse_args()
    if args.gui or (args.experiment is None and not args.policies):
        try:
            selected_experiment, selected_policies = _select_experiment_graphically()
        except RuntimeError as error:
            raise SystemExit(f"Selection cancelled: {error}") from error
    elif args.experiment is None:
        raise SystemExit("--experiment is required when policy paths are supplied from the command line")
    else:
        selected_experiment = args.experiment
        selected_policies = args.policies

    experiment_dir = _resolve_experiment_directory(selected_experiment)
    config_path = experiment_dir / "trajectory.json"
    policy_dir = experiment_dir / "policy"
    policies = selected_policies or _discover_policies(experiment_dir)
    aggregate_mode = _is_aggregate_policy_selection(policies)
    output_dir = args.output_dir.expanduser() if args.output_dir else experiment_dir / "evaluation"

    # Load the LTLf task shared with the trainer.
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    regions, _, task_propositions = load_task_propositions(config.get("regions"), config.get("predicates"))
    goal_reward = float(config.get("goal_reward", 10000.0))

    # Evaluate policies one at a time to keep rendering and output deterministic.
    results = []
    for policy in policies:
        result = evaluate_policy( policy, policy_dir, args.episodes, args.render, config, task_propositions, goal_reward, args.seed, network_type=args.network_type, no_limit=args.no_limit, verbose=not aggregate_mode, )
        results.append(result)

    # Print the summary and create one plot for each evaluated policy.
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        success_rate = result["successes"] / args.episodes
        failure_rate = result["failures"] / args.episodes
        mean_gym_return = np.mean(result["environment_returns"])
        mean_length = np.mean(result["episode_lengths"])
        cycle_summary = f", mean cycles={np.mean(result['completed_cycles']):.2f}" if "waypoint_cycle" in config or config.get("task_type") == "cyclic_waypoints" else ""
        reached = ", ".join(f"q={q}: {count}/{args.episodes}" for q, count in result["state_reach_counts"].items())
        if not aggregate_mode:
            print(f"[{result['policy']}] success={success_rate:.1%}, failure={failure_rate:.1%}{cycle_summary}, mean Gym return={mean_gym_return:.2f}, mean length={mean_length:.1f} | reached: {reached}")
        policy_plot = plot_policy(result, args.window, output_dir)
        if not aggregate_mode:
            print(f"Plot saved to: {policy_plot}")

    # Add a combined comparison when more than one policy was requested.
    if len(results) > 1:
        comparison_plot = plot_comparison(results, args.window, output_dir)
        if not aggregate_mode:
            print(f"Comparison saved to: {comparison_plot}")
    print_best_last_summary(results, results[0]["task_description"])


if __name__ == "__main__":
    main()
