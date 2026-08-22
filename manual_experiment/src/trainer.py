# ==============================
# Standard library imports
# ==============================

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import numpy as np
import torch

from abstract_mdps import ManualWaypointMDP
from agent import HierarchicalDQNLearner
from manual_automaton import CyclicWaypointsAutomaton
from spatial_regions import load_regions
from utils import phi_mapping_sequential, plot_buffer_fractions, plot_buffer_variance, plot_shaping_reward_breakdown, plot_training_variance, save_sequential_heatmaps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == "src"
    else SCRIPT_DIR
)


# ==============================
# Data and state helpers
# ==============================

def _positive_int(value):
    """Parse a strictly positive command-line integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _experiment_name(value):
    """Validate a safe single-directory experiment name."""
    name = str(value).strip()
    if len(name) > 100 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise argparse.ArgumentTypeError("must start with a letter or digit and contain only letters, digits, '.', '_' or '-'")
    return name


def _resolve_config_path(requested_path, default_filename, experiment_dir, post_process):
    """Resolve a config, preferring the experiment snapshot during post-processing."""
    requested = Path(requested_path).expanduser()
    framework_default = Path(SCRIPT_DIR) / default_filename
    uses_default = (
        str(requested_path) == default_filename
        or requested.resolve() == framework_default.resolve()
    )
    candidates = []
    if post_process and uses_default:
        candidates.extend([Path(experiment_dir) / default_filename, Path(experiment_dir) / "results" / default_filename])
    candidates.append(requested)
    if not requested.is_absolute():
        candidates.append(Path(SCRIPT_DIR) / requested)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Configuration file not found. Checked:\n  - {checked}")


def _archive_config(config_path, experiment_dir, filename):
    """Store the exact training configuration beside the experiment outputs."""
    destination = Path(experiment_dir) / filename
    if Path(config_path).resolve() != destination.resolve():
        shutil.copy2(config_path, destination)


def _resolve_metrics_path(experiment_dir, filename):
    """Find metrics in the results subfolder or the legacy experiment root."""
    candidates = [
        Path(experiment_dir) / "results" / filename,
        Path(experiment_dir) / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Training data not found. Checked:\n  - {checked}")


def _organize_policy_files(policy_dir):
    """Move checkpoints from legacy layouts into policy/best and policy/last."""
    policy_root = Path(policy_dir)
    destinations = {
        "best": policy_root / "best",
        "last": policy_root / "last",
    }
    for destination in destinations.values():
        destination.mkdir(parents=True, exist_ok=True)

    for category, destination in destinations.items():
        for source in policy_root.glob(f"{category}_policy*"):
            if source.is_file() and not (destination / source.name).exists():
                shutil.move(str(source), destination / source.name)


def _organize_legacy_seed_plots(image_dir):
    """Move legacy per-seed plots from img/ into img/seed_<seed>/ folders."""
    pattern = re.compile(r"^((?:reward_breakdown|buffer_fractions)_.+)_seed_(-?\d+)\.png$")
    for source in Path(image_dir).glob("*.png"):
        match = pattern.fullmatch(source.name)
        if not match:
            continue
        destination_dir = Path(image_dir) / f"seed_{match.group(2)}"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{match.group(1)}.png"
        if not destination.exists():
            shutil.move(str(source), destination)


def save_training_data(filename, **kwargs):
    """Convert training metrics to arrays and save them in a compressed NPZ file."""
    # Preserve numeric dtypes and rectangular shapes for direct plotting.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np_data = {key: np.asarray(value) for key, value in kwargs.items()}
    if any(array.dtype == object for array in np_data.values()):
        raise ValueError("Training metrics must be rectangular numeric arrays")
    np.savez_compressed(filename, **np_data)
    print(f"\nTraining data saved to: {filename}")


def _set_training_seed(seed, env=None):
    """Seed every random generator used by DDQN and LunarLander."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.action_space.seed(seed)


def _aggregate_seed_metrics(seed_metrics, seeds):
    """Keep first-run compatibility and add every metric stacked by seed."""
    if not seed_metrics:
        raise ValueError("At least one seed run is required")
    aggregated = dict(seed_metrics[0])
    aggregated["seeds"] = np.asarray(seeds, dtype=np.int64)
    for key in seed_metrics[0]:
        if key == "automaton_states":
            continue
        try:
            aggregated[f"{key}_runs"] = np.stack([np.asarray(metrics[key]) for metrics in seed_metrics])
        except ValueError as error:
            raise ValueError(f"Metric {key!r} has inconsistent shapes across seeds") from error
    for key in ("task_rewards", "learning_rewards", "shaping_rewards"):
        runs = aggregated[f"{key}_runs"]
        aggregated[f"{key}_mean"] = np.mean(runs, axis=0)
        aggregated[f"{key}_variance"] = np.var(runs, axis=0)
    return aggregated


def _abstract_position(observation, abstract_mdp):
    """Map a raw environment observation to its abstract spatial coordinates."""
    x, y, _ = phi_mapping_sequential(observation, 0, abstract_mdp.width, abstract_mdp.height)
    return x, y


def _augment_state(observation, q, state_to_index):
    """Append a one-hot encoding of the current automaton state."""
    one_hot = np.zeros(len(state_to_index), dtype=np.float32)
    one_hot[state_to_index[q]] = 1.0
    return np.concatenate((observation, one_hot)).astype(np.float32)


def _evaluate_initial_automaton_state(observation, abstract_mdp):
    """Consume the initial observation and return the first active automaton state."""
    initial_truth_assignment = abstract_mdp.get_environment_truth_assignment(observation)
    pre_trace_q = abstract_mdp.automaton.get_initial_q()
    return abstract_mdp.automaton.advance(pre_trace_q, initial_truth_assignment).next_state


def _format_counter(counter):
    """Convert an automaton transition counter into a readable string."""
    if not counter:
        return "none"
    return ", ".join(f"{source}->{destination}: {count}" for (source, destination), count in sorted(counter.items()))


# ==============================
# Logging and checkpoint helpers
# ==============================

def _write_log(message, log_handle=None):
    """Print a message and optionally append it to the active log file."""
    print(message)
    if log_handle:
        log_handle.write(message)
        log_handle.flush()


def _write_run_header(log_handle, episodes, use_shaping, goal_reward, abstract_mdp, automaton_states, training_shaping_gamma, eval_interval, eval_episodes, eval_seed):
    """Write the configuration and automaton metadata for a training run."""
    if not log_handle:
        return
    automaton = abstract_mdp.automaton
    shaping_formula = (
        "gamma*Phi(next)-Phi(state)"
        if training_shaping_gamma
        else "Phi(next)-Phi(state)"
    )
    header = (
        "\n=== NEW RUN ===\n"
        f"episodes={episodes}, shaping={use_shaping}, goal_reward={goal_reward}, gamma={abstract_mdp.gamma}\n"
        f"training_shaping_gamma={training_shaping_gamma}, shaping_formula={shaping_formula}\n"
        f"eval_interval={eval_interval}, eval_episodes={eval_episodes}, eval_seed={eval_seed}\n"
        f"regions={{{', '.join(f'{name}: {region.as_dict()}' for name, region in abstract_mdp.regions.items())}}}\n"
        f"automaton_states={automaton_states}, initial={automaton.get_initial_q()}, accepting={sorted(automaton.accepting_states)}\n"
    )
    log_handle.write(header)
    log_handle.flush()


def _should_log(episode, episodes, log_interval):
    """Return whether the current episode requires a periodic training report."""
    return episode == 0 or episode + 1 == episodes or (episode + 1) % log_interval == 0


def _is_evaluation_due(episode, episodes, eval_interval):
    """Return whether greedy evaluation is due for the current episode."""
    return episode + 1 == episodes or (episode + 1) % eval_interval == 0


def _build_training_log(episode, episodes, log_interval, automaton_states, agent, histories, cumulative_counters):
    """Build a report containing recent metrics and automaton counters."""
    window = min(log_interval, episode + 1)
    recent_slice = slice(-window, None)
    recent_transitions = Counter()
    for transitions in histories["transition_counters"][-window:]:
        recent_transitions.update(transitions)

    recent_state_visits = np.asarray(histories["state_visits"], dtype=np.int64)[:, -window:].sum(axis=1)
    recent_state_entries = np.asarray(histories["state_entries"], dtype=np.int64)[:, -window:].sum(axis=1)
    buffer_details = ", ".join(f"{q}: {agent.memory.q_fraction_onehot(index, len(automaton_states)):.1%}" for index, q in enumerate(automaton_states))
    recent_visits_details = ", ".join(f"{q}: {recent_state_visits[index]}" for index, q in enumerate(automaton_states))
    recent_entries_details = ", ".join(f"{q}: {recent_state_entries[index]}" for index, q in enumerate(automaton_states))
    cumulative_visits_details = ", ".join(f"{q}: {cumulative_counters['state_visits'][q]}" for q in automaton_states)
    cumulative_entries_details = ", ".join(f"{q}: {cumulative_counters['state_entries'][q]}" for q in automaton_states)

    return (
        "\n"
        f"[Episode {episode + 1}/{episodes} | last {window}]\n"
        f"success rate                : {np.mean(histories['successes'][recent_slice]):.1%} (cumulative {np.mean(histories['successes']):.1%})\n"
        f"synthetic task reward       : {np.mean(histories['task_rewards'][recent_slice]):.3f}\n"
        f"shaping reward              : {np.mean(histories['shaping_rewards'][recent_slice]):.3f}\n"
        f"learning reward             : {np.mean(histories['learning_rewards'][recent_slice]):.3f}\n"
        f"episode length              : {np.mean(histories['episode_lengths'][recent_slice]):.1f}\n"
        f"abstract changes / episode  : {np.mean(histories['abstract_changes'][recent_slice]):.1f}\n"
        f"completed cycles / episode   : {np.mean(histories['completed_cycles'][recent_slice]):.2f}\n"
        f"automaton changes / episode  : {np.mean(histories['automaton_transitions'][recent_slice]):.2f}\n"
        f"automaton changes in window  : {_format_counter(recent_transitions)}\n"
        f"epsilon (next episode)       : {histories['epsilons'][-1]:.5f}\n"
        f"replay buffer                : {len(agent.memory)} samples [{buffer_details}]\n"
        f"state visits in window        : {recent_visits_details}\n"
        f"state visits cumulative       : {cumulative_visits_details}\n"
        f"state entries in window       : {recent_entries_details}\n"
        f"state entries cumulative      : {cumulative_entries_details}\n"
        f"transitions cumulative       : {_format_counter(cumulative_counters['transitions'])}\n"
        f"Gym endings cumulative       : terminated={cumulative_counters['env_terminated']}, truncated={cumulative_counters['env_truncated']}\n"
    )


def _save_named_policy(agent, policy_name):
    """Save the current policy using a stable descriptive filename."""
    category = "best" if policy_name.startswith("best_policy") else "last"
    os.makedirs(os.path.join(agent.policy_dir, category), exist_ok=True)
    agent.policy_name = os.path.join(category, policy_name)
    agent._save_policy()


def _monitoring_average(values, episode, log_interval):
    """Return the mean over the active monitoring window."""
    window = min(log_interval, episode + 1)
    return float(np.mean(values[-window:]))


def _greedy_action(agent, augmented_state):
    """Select an action from the policy network without exploration."""
    with torch.inference_mode():
        state_tensor = torch.as_tensor(augmented_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        return agent.policy_net(state_tensor).argmax(dim=1).item()


def _evaluate_agent_greedily(agent, abstract_mdp, episodes, goal_reward, seed):
    """Evaluate the policy without exploration, replay writes, or updates."""
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.active_states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    successful_episodes = 0
    task_rewards = []
    episode_lengths = []
    completed_cycles = []
    transition_counts = Counter()
    evaluation_env = gym.make("LunarLander-v3", continuous=False)
    was_training = agent.policy_net.training
    agent.policy_net.eval()
    try:
        for evaluation_episode in range(episodes):
            raw_state, _ = evaluation_env.reset(seed=seed + evaluation_episode)
            q = _evaluate_initial_automaton_state(raw_state, abstract_mdp)
            terminated = truncated = False
            steps = 0
            cycles = 0
            while not (terminated or truncated):
                augmented_state = _augment_state(raw_state, q, state_to_index)
                action = _greedy_action(agent, augmented_state)
                next_raw_state, _ignored_reward, terminated, truncated, _ = evaluation_env.step(action)
                previous_q = q
                automaton_step = automaton.advance(previous_q, abstract_mdp.get_environment_truth_assignment(next_raw_state))
                q = automaton_step.next_state
                if q not in state_to_index:
                    raise RuntimeError(f"Automaton returned unknown evaluation state {q!r}")
                if q != previous_q:
                    transition_counts[(previous_q, q)] += 1
                cycles += int(automaton_step.completed_cycle)
                raw_state = next_raw_state
                steps += 1
            successful_episodes += int(cycles > 0)
            task_rewards.append(float(goal_reward) * cycles)
            episode_lengths.append(steps)
            completed_cycles.append(cycles)
    finally:
        if was_training:
            agent.policy_net.train()
        evaluation_env.close()
    return {"success_rate": successful_episodes / episodes, "mean_task_reward": float(np.mean(task_rewards)), "mean_episode_length": float(np.mean(episode_lengths)), "mean_completed_cycles": float(np.mean(completed_cycles)), "transition_counts": transition_counts}


def _evaluation_score(metrics):
    """Order evaluations by success, task reward, then shorter episodes."""
    return (metrics["success_rate"], metrics["mean_task_reward"], -metrics["mean_episode_length"])


def _validate_training_setup(automaton, state_to_index, episodes, log_interval, eval_interval, eval_episodes):
    """Validate automaton consistency and numeric training parameters."""
    if automaton.get_initial_q() not in state_to_index:
        raise ValueError("The initial state is missing from automaton.states")
    if set(state_to_index) != set(automaton.active_states):
        raise ValueError("Network phases must match the stable automaton states")
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if log_interval <= 0:
        raise ValueError("log_interval must be greater than zero")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be greater than zero")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be greater than zero")


def _build_training_results(histories, buffer_histories, automaton_states, best_mean_reward, best_policy_episode):
    """Select and name the numeric histories returned by the training loop."""
    return {
        "task_rewards": histories["task_rewards"],
        "learning_rewards": histories["learning_rewards"],
        "shaping_rewards": histories["shaping_rewards"],
        "epsilon_history": histories["epsilons"],
        "buffer_histories": buffer_histories,
        "state_visit_histories": histories["state_visits"],
        "state_entry_histories": histories["state_entries"],
        "successes": histories["successes"],
        "completed_cycles": histories["completed_cycles"],
        "episode_lengths": histories["episode_lengths"],
        "abstract_changes": histories["abstract_changes"],
        "automaton_transitions": histories["automaton_transitions"],
        "automaton_states": automaton_states,
        "best_mean_learning_reward": best_mean_reward,
        "best_mean_eval_task_reward": best_mean_reward,
        "best_policy_episode": best_policy_episode,
        "evaluation_steps": histories["evaluation_steps"],
        "eval_success_rates": histories["eval_success_rates"],
        "eval_task_rewards": histories["eval_task_rewards"],
        "eval_episode_lengths": histories["eval_episode_lengths"],
        "eval_completed_cycles": histories["eval_completed_cycles"],
    }


# ==============================
# Training loop
# ==============================

def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, log_file=None, log_interval=100, eval_interval=1000, eval_episodes=50, eval_seed=100000, training_shaping_gamma=True, seed=None, policy_suffix=""):
    """
    Train the DDQN agent with the manual automaton and one global epsilon.

    The Gym reward is deliberately discarded. The learning reward is the
    synthetic goal reward plus potential-based shaping. Shaping is evaluated
    only when the complete abstract state (x, y, q) changes.
    """
    # Build a stable mapping between automaton states and network features.
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.active_states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    num_states = len(automaton_states)

    # Fail early if the automaton or training parameters are inconsistent.
    _validate_training_setup(automaton, state_to_index, episodes, log_interval, eval_interval, eval_episodes)

    # Store episode-level metrics for plots and post-processing.
    task_reward_history = []
    learning_reward_history = []
    shaping_reward_history = []
    epsilon_history = []
    episode_length_history = []
    success_history = []
    completed_cycle_history = []
    abstract_change_history = []
    automaton_transition_history = []
    transition_counter_history = []
    buffer_histories = [[] for _ in automaton_states]
    state_visit_histories = [[] for _ in automaton_states]
    state_entry_histories = [[] for _ in automaton_states]
    histories = {
        "task_rewards": task_reward_history,
        "learning_rewards": learning_reward_history,
        "shaping_rewards": shaping_reward_history,
        "epsilons": epsilon_history,
        "episode_lengths": episode_length_history,
        "successes": success_history,
        "completed_cycles": completed_cycle_history,
        "abstract_changes": abstract_change_history,
        "automaton_transitions": automaton_transition_history,
        "transition_counters": transition_counter_history,
        "state_visits": state_visit_histories,
        "state_entries": state_entry_histories,
        "evaluation_steps": [],
        "eval_success_rates": [],
        "eval_task_rewards": [],
        "eval_episode_lengths": [],
        "eval_completed_cycles": [],
    }

    # Keep cumulative counters for diagnostics shown during training.
    cumulative_state_visits = Counter()
    cumulative_state_entries = Counter()
    cumulative_transitions = Counter()
    cumulative_env_terminated = 0
    cumulative_env_truncated = 0
    best_mean_reward = -np.inf
    best_policy_episode = 0
    best_evaluation_score = None

    # Open one append-only log file for the complete run.
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
    _write_run_header(log_handle, episodes, use_shaping, goal_reward, abstract_mdp, automaton_states, training_shaping_gamma, eval_interval, eval_episodes, eval_seed)

    try:
        for episode in range(episodes):
            evaluation_due = _is_evaluation_due(episode, episodes, eval_interval)
            # Reset the environment and consume s0 before selecting an action.
            raw_state, _ = env.reset(seed=seed if episode == 0 else None)
            q = _evaluate_initial_automaton_state(raw_state, abstract_mdp)
            if q not in state_to_index:
                raise RuntimeError(f"Automaton returned unknown initial state {q!r}")
            augmented_state = _augment_state(raw_state, q, state_to_index)

            # Reset counters local to the current episode.
            succeeded = False
            episode_done = False
            episode_steps = 0
            episode_task_reward = 0.0
            episode_shaping_reward = 0.0
            episode_abstract_changes = 0
            episode_automaton_transitions = 0
            episode_completed_cycles = 0
            episode_state_visits = [0] * num_states
            episode_state_visits[state_to_index[q]] = 1
            # Count s0 as an entry from the virtual pre-episode state.
            episode_state_entries = [0] * num_states
            episode_state_entries[state_to_index[q]] = 1
            episode_transitions = Counter()
            cumulative_state_visits[q] += 1
            cumulative_state_entries[q] += 1

            while not episode_done:
                # Select an action using the single global epsilon.
                agent.eps = epsilon_history[-1] if epsilon_history else agent.eps
                action = agent.select_action(augmented_state)

                # The environment reward is intentionally not part of training.
                next_raw_state, _ignored_env_reward, env_terminated, env_truncated, _ = env.step(action)

                # Map the transition to abstract spatial states.
                x, y = _abstract_position(raw_state, abstract_mdp)
                next_x, next_y = _abstract_position(next_raw_state, abstract_mdp)
                abstract_state = (x, y, q)

                # Advance the automaton using propositions true on arrival.
                truth_assignment = abstract_mdp.get_environment_truth_assignment(next_raw_state)
                automaton_step = automaton.advance(q, truth_assignment)
                next_q = automaton_step.next_state
                if next_q not in state_to_index:
                    raise RuntimeError(f"Automaton returned unknown state {next_q!r} from state {q!r}")

                # Count every arrival in an automaton state, including self-loops.
                episode_state_visits[state_to_index[next_q]] += 1
                cumulative_state_visits[next_q] += 1

                # Track physical abstraction and automaton changes separately.
                abstract_next_state = (next_x, next_y, next_q)
                abstract_changed = abstract_state != abstract_next_state
                automaton_changed = next_q != q

                if abstract_changed:
                    episode_abstract_changes += 1
                if automaton_changed:
                    transition = (q, next_q)
                    episode_automaton_transitions += 1
                    episode_state_entries[state_to_index[next_q]] += 1
                    episode_transitions[transition] += 1
                    cumulative_state_entries[next_q] += 1
                    cumulative_transitions[transition] += 1

                # Reward the final waypoint once per completed cycle.
                completed_cycle = automaton_step.completed_cycle
                synthetic_goal_reward = (
                    float(goal_reward) if completed_cycle else 0.0
                )
                if completed_cycle:
                    succeeded = True
                    episode_completed_cycles += 1

                # Acceptance does not end an episode. Only Gymnasium can do so.
                # A truncation (for example Gym's time limit) ends data
                # collection, but it is not an MDP terminal state: DDQN must
                # still bootstrap from its final observation.
                episode_done = env_terminated or env_truncated
                bootstrap_terminal = env_terminated
                next_augmented_state = _augment_state(next_raw_state, next_q, state_to_index)

                # Evaluate shaping only when the complete abstract state changes.
                shaping_signal = 0.0
                if use_shaping and abstract_changed:
                    phi_state = abstract_mdp.v_star.get(abstract_state, 0.0)
                    phi_next_state = abstract_mdp.v_star.get(abstract_next_state, 0.0)
                    training_discount = abstract_mdp.gamma if training_shaping_gamma else 1.0
                    shaping_signal = training_discount * phi_next_state - phi_state

                # Store the transition and perform one DDQN optimization step.
                learning_reward = synthetic_goal_reward + shaping_signal
                agent.memory.push(augmented_state, action, learning_reward, next_augmented_state, bootstrap_terminal)
                agent.optimize_model()

                # Update the episode totals and move to the next state.
                episode_steps += 1
                episode_task_reward += synthetic_goal_reward
                episode_shaping_reward += shaping_signal
                raw_state = next_raw_state
                augmented_state = next_augmented_state
                q = next_q

                # Count Gym endings for diagnostics without using its reward.
                if env_terminated:
                    cumulative_env_terminated += 1
                if env_truncated:
                    cumulative_env_truncated += 1

            # Decay the single epsilon once at the end of the episode.
            next_epsilon = max(agent.eps_min, agent.eps * agent.eps_decay)
            agent.eps = next_epsilon

            # Save the metrics collected for this episode.
            episode_learning_reward = episode_task_reward + episode_shaping_reward
            task_reward_history.append(episode_task_reward)
            shaping_reward_history.append(episode_shaping_reward)
            learning_reward_history.append(episode_learning_reward)
            epsilon_history.append(next_epsilon)
            episode_length_history.append(episode_steps)
            success_history.append(int(succeeded))
            completed_cycle_history.append(episode_completed_cycles)
            abstract_change_history.append(episode_abstract_changes)
            automaton_transition_history.append(episode_automaton_transitions)
            transition_counter_history.append(episode_transitions)

            # Record replay-buffer composition, state visits, and entries from other states.
            for index in range(num_states):
                buffer_histories[index].append(agent.memory.q_fraction_onehot(index, num_states))
                state_visit_histories[index].append(episode_state_visits[index])
                state_entry_histories[index].append(episode_state_entries[index])

            # Print recent and cumulative diagnostics at the requested interval.
            if _should_log(episode, episodes, log_interval) or evaluation_due:
                cumulative_counters = {"state_visits": cumulative_state_visits, "state_entries": cumulative_state_entries, "transitions": cumulative_transitions, "env_terminated": cumulative_env_terminated, "env_truncated": cumulative_env_truncated}
                _write_log(_build_training_log(episode, episodes, log_interval, automaton_states, agent, histories, cumulative_counters), log_handle)

            if evaluation_due:
                _write_log(f"\nStarting autonomous greedy evaluation at episode {episode + 1} ({eval_episodes} fixed-seed episodes)...\n", log_handle)
                evaluation = _evaluate_agent_greedily(agent, abstract_mdp, eval_episodes, goal_reward, eval_seed)
                histories["evaluation_steps"].append(episode + 1)
                histories["eval_success_rates"].append(evaluation["success_rate"])
                histories["eval_task_rewards"].append(evaluation["mean_task_reward"])
                histories["eval_episode_lengths"].append(evaluation["mean_episode_length"])
                histories["eval_completed_cycles"].append(evaluation["mean_completed_cycles"])
                _write_log(f"[Greedy evaluation at episode {episode + 1} | {eval_episodes} fixed-seed episodes]\nsuccess={evaluation['success_rate']:.1%}, task reward={evaluation['mean_task_reward']:.3f}, cycles={evaluation['mean_completed_cycles']:.3f}, length={evaluation['mean_episode_length']:.1f}\nautomaton transitions: {_format_counter(evaluation['transition_counts'])}\n", log_handle)
                score = _evaluation_score(evaluation)
                if best_evaluation_score is None or score > best_evaluation_score:
                    best_evaluation_score = score
                    best_mean_reward = evaluation["mean_task_reward"]
                    best_policy_episode = episode + 1
                    if save_policy:
                        _save_named_policy(agent, f"best_policy{policy_suffix}.pth")
                    _write_log(f"Best policy updated from autonomous greedy evaluation at episode {best_policy_episode}.\n", log_handle)

        # Save the final policy independently from its monitored performance.
        if save_policy:
            _save_named_policy(agent, f"last_policy{policy_suffix}.pth")
            _write_log(f"Last policy saved after episode {episodes}. Best greedy evaluation: episode {best_policy_episode}, mean task reward={best_mean_reward:.3f}\n", log_handle)
    finally:
        # Always close the log, including when training raises an exception.
        if log_handle:
            log_handle.close()

    # Return named histories to avoid ambiguous tuple positions.
    return _build_training_results(histories, buffer_histories, automaton_states, best_mean_reward, best_policy_episode)


# ==============================
# Experiment setup and outputs
# ==============================

def main(args):
    """Configure the experiment, run or load training, and generate diagnostic plots."""
    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be greater than zero")
    # Keep every artifact isolated under results/<experiment-name>/.
    experiment_dir = os.path.join(FRAMEWORK_DIR, "results", args.experiment_name)
    if args.post_process and not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    data_dir = os.path.join(experiment_dir, "results")
    image_dir = os.path.join(experiment_dir, "img")
    log_dir = os.path.join(experiment_dir, "logs")
    policy_dir = os.path.join(experiment_dir, "policy")
    best_policy_dir = os.path.join(policy_dir, "best")
    last_policy_dir = os.path.join(policy_dir, "last")
    for directory in (
        data_dir,
        image_dir,
        log_dir,
        best_policy_dir,
        last_policy_dir,
    ):
        os.makedirs(directory, exist_ok=True)
    _organize_policy_files(policy_dir)
    _organize_legacy_seed_plots(image_dir)
    plot_dir = image_dir
    print(f"Experiment outputs: {experiment_dir}")

    # Load the manual task and optional training parameters.
    config_path = _resolve_config_path(args.config, "trajectory.json", experiment_dir, args.post_process)
    print(f"Configuration: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not args.post_process:
        _archive_config(config_path, experiment_dir, "trajectory.json")

    regions = load_regions(config.get("regions"))
    gamma = float(config.get("gamma", 0.99))
    goal_reward = float(config.get("goal_reward", 10000))
    grid_w = int(config.get("grid_w", 12))
    grid_h = int(config.get("grid_h", 12))

    # The cycle order is explicit; when omitted, JSON region insertion order
    # provides a convenient backwards-compatible default.
    waypoint_cycle = config.get("waypoint_cycle", list(regions))
    automaton = CyclicWaypointsAutomaton(waypoint_cycle)
    automaton.validate_regions(regions)
    print("=== MANUAL AUTOMATON TRAINING (single epsilon) ===\n" f"Regions: { {name: region.as_dict() for name, region in regions.items()} }\n" f"Automaton: states={automaton.states}, stable={automaton.active_states}, " f"initial={automaton.initial_state}, " f"accepting={sorted(automaton.accepting_states)}\n" f"Cycle: {automaton.describe_cycle()}\n" "Gym reward is ignored; acceptance does not end an episode.")

    if not args.post_process:
        automaton.render_graph(directory=image_dir)

    # Heatmaps depend only on the saved task configuration, not on agent training.
    abstract_mdp = ManualWaypointMDP(regions=regions, automaton=automaton, width=grid_w, height=grid_h, gamma=gamma, goal_reward=goal_reward)
    abstract_mdp.value_iteration()
    save_sequential_heatmaps(abstract_mdp, filename_prefix="single_epsilon_exp", output_dir=os.path.join(image_dir, "heatmaps"))

    if not args.post_process:
        # Create LunarLander only when agent training is requested.
        seeds = [args.seed + index for index in range(args.num_seeds)]
        seed_metrics = []
        for run_index, run_seed in enumerate(seeds, start=1):
            print(f"\n=== SEED RUN {run_index}/{args.num_seeds}: seed={run_seed} ===")
            _set_training_seed(run_seed)
            env = gym.make("LunarLander-v3", continuous=False)
            try:
                _set_training_seed(run_seed, env)
                agent = HierarchicalDQNLearner(env=env, max_episodes=args.episodes, eps_decay=args.eps_decay, gamma=gamma, extra_state_dims=len(automaton.active_states), use_polyak=args.polyak, tau=args.polyak_tau, target_update_freq=args.target_update_freq, network_type=args.network_type, policy_dir=policy_dir)
                policy_suffix = "" if args.num_seeds == 1 else f"_seed_{run_seed}"
                metrics = run_sequential_training(env=env, agent=agent, abstract_mdp=abstract_mdp, episodes=args.episodes, goal_reward=goal_reward, use_shaping=not args.no_shaping, log_file=f"{log_dir}/single_epsilon_training_seed_{run_seed}.log", log_interval=args.log_interval, eval_interval=args.eval_interval, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed, training_shaping_gamma=args.training_shaping_gamma, seed=run_seed, policy_suffix=policy_suffix)
                seed_metrics.append(metrics)
                save_training_data(f"{data_dir}/single_epsilon_data_seed_{run_seed}.npz", **metrics)
            finally:
                env.close()
        save_training_data(f"{data_dir}/single_epsilon_data.npz", **_aggregate_seed_metrics(seed_metrics, seeds))

    # Load saved metrics and generate the final diagnostic plots.
    data_path = (
        _resolve_metrics_path(experiment_dir, "single_epsilon_data.npz")
        if args.post_process
        else Path(data_dir) / "single_epsilon_data.npz"
    )
    print(f"Training data: {data_path}")
    data = np.load(data_path, allow_pickle=False)
    task_reward_runs = data["task_rewards_runs"] if "task_rewards_runs" in data else data["task_rewards"][np.newaxis, :]
    learning_reward_runs = data["learning_rewards_runs"] if "learning_rewards_runs" in data else data["learning_rewards"][np.newaxis, :]
    epsilon_runs = data["epsilon_history_runs"] if "epsilon_history_runs" in data else data["epsilon_history"][np.newaxis, ...]
    buffer_runs = data["buffer_histories_runs"] if "buffer_histories_runs" in data else data["buffer_histories"][np.newaxis, ...]
    seed_values = data["seeds"] if "seeds" in data else np.asarray([args.seed])
    for obsolete_name in ("buffer_fractions_single_epsilon.png", "reward_breakdown_single_epsilon.png"):
        (Path(plot_dir) / obsolete_name).unlink(missing_ok=True)
    for run_index, (run_seed, task_rewards, learning_rewards, epsilon_history) in enumerate(zip(seed_values, task_reward_runs, learning_reward_runs, epsilon_runs)):
        seed_plot_dir = os.path.join(plot_dir, f"seed_{int(run_seed)}")
        os.makedirs(seed_plot_dir, exist_ok=True)
        plot_shaping_reward_breakdown(task_rewards, learning_rewards, epsilon_history, window_size=args.plot_window, filename=f"{seed_plot_dir}/reward_breakdown_single_epsilon.png", title=f"Reward Breakdown — Seed {int(run_seed)}")
        if run_index < len(buffer_runs):
            plot_buffer_fractions(buffer_runs[run_index], filename=f"{seed_plot_dir}/buffer_fractions_single_epsilon.png", window_size=args.plot_window, state_labels=data["automaton_states"], title=f"Replay Buffer Composition — Seed {int(run_seed)}")
    plot_training_variance(learning_reward_runs, window_size=args.plot_window, filename=f"{plot_dir}/training_variance_single_epsilon.png", epsilon_histories=epsilon_runs)
    plot_buffer_variance(buffer_runs, window_size=args.plot_window, filename=f"{plot_dir}/buffer_variance_single_epsilon.png", state_labels=data["automaton_states"])
    print("\nFinished.")


# ==============================
# Command-line entry point
# ==============================

if __name__ == "__main__":
    # Expose the main training and post-processing options.
    parser = argparse.ArgumentParser(description="Manual-automaton DDQN training with one global epsilon.")
    parser.add_argument("--experiment-name", type=_experiment_name, required=True, help="Output directory name under results/.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-seeds", type=_positive_int, default=1, help="Number of training runs with consecutive seeds.")
    parser.add_argument("--seed", type=int, default=42, help="First training seed.")
    parser.add_argument("--config", default="trajectory.json")
    parser.add_argument("--eps-decay", type=float, default=0.9996)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=_positive_int, default=1000, help="Run autonomous greedy evaluation every N training episodes.")
    parser.add_argument("--eval-episodes", type=_positive_int, default=50, help="Number of fixed-seed episodes used at each greedy evaluation.")
    parser.add_argument("--eval-seed", type=int, default=100000, help="First held-out seed reused at every greedy evaluation.")
    parser.add_argument("--plot-window", type=int, default=500)
    parser.add_argument("--polyak", action=argparse.BooleanOptionalAction, default=True, help="Use Polyak target updates (disable with --no-polyak).")
    parser.add_argument("--polyak-tau", type=float, default=0.005)
    parser.add_argument("--target-update-freq", type=int, default=1000, help="Hard target-network update interval used with --no-polyak.")
    parser.add_argument("--network-type", choices=["standard", "dueling"], default="standard", help="Q-network architecture: standard MLP or dueling value/advantage streams.")
    parser.add_argument("--training-shaping-gamma", action=argparse.BooleanOptionalAction, default=True, help="Use gamma*Phi(next)-Phi(state) during training; disable to use Phi(next)-Phi(state).")
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--post-process", action="store_true")
    main(parser.parse_args())
