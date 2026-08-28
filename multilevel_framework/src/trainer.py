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

from abstraction import AbstractionConfig
from abstract_mdps import LTLfAutomaton, MultiLevelWaypointMDP
from agent import HierarchicalDQNLearner, TabularQLearner
from automaton_validator import validate_automaton
from spatial_regions import load_task_propositions
from utils import (
    phi_mapping_sequential,
    plot_buffer_fractions,
    plot_buffer_variance,
    plot_evaluation_performance,
    plot_shaping_reward_breakdown,
    plot_tabular_training_diagnostics,
    plot_training_variance,
    save_abstract_learning_curves,
    save_multilevel_heatmaps,
    save_multilevel_value_functions,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == "src"
    else SCRIPT_DIR
)
CONFIG_DIR = os.path.join(FRAMEWORK_DIR, "config")


# ==============================
# Data and state helpers
# ==============================

def _positive_int(value):
    """Parse a strictly positive command-line integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _discount_factor(value):
    """Parse a valid discounted-return factor."""
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("must be in the interval (0, 1]")
    return number


def _learning_rate(value):
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("must be in the interval (0, 1]")
    return number


def _experiment_name(value):
    """Validate a safe single-directory experiment name."""
    name = str(value).strip()
    if len(name) > 100 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise argparse.ArgumentTypeError( "must start with a letter or digit and contain only letters, digits, '.', '_' or '-'" )
    return name


def _resolve_config_path(requested_path, default_filename, experiment_dir, post_process):
    """Resolve a config, preferring the experiment snapshot during post-processing."""
    requested = Path(requested_path).expanduser()
    framework_default = Path(CONFIG_DIR) / default_filename
    uses_default = (
        str(requested_path) == default_filename
        or requested.resolve() == framework_default.resolve()
    )
    candidates = []
    if post_process and uses_default:
        candidates.extend( [ Path(experiment_dir) / default_filename, Path(experiment_dir) / "results" / default_filename, ] )
    candidates.append(requested)
    if not requested.is_absolute():
        candidates.append(Path(CONFIG_DIR) / requested)

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
            aggregated[f"{key}_runs"] = np.stack( [np.asarray(metrics[key]) for metrics in seed_metrics] )
        except ValueError as error:
            raise ValueError(f"Metric {key!r} has inconsistent shapes across seeds") from error
    for key in ("task_rewards", "learning_rewards", "shaping_rewards"):
        runs = aggregated[f"{key}_runs"]
        aggregated[f"{key}_mean"] = np.mean(runs, axis=0)
        aggregated[f"{key}_variance"] = np.var(runs, axis=0)
    return aggregated


def _abstract_position(observation, abstract_mdp):
    """Map a raw environment observation to its abstract spatial coordinates."""
    x, y, _ = phi_mapping_sequential( observation, 0, abstract_mdp.width, abstract_mdp.height )
    return x, y


def _augment_state(observation, q, state_to_index):
    """Append a one-hot encoding of the current DFA state to an observation."""
    one_hot = np.zeros(len(state_to_index), dtype=np.float32)
    one_hot[state_to_index[q]] = 1.0
    return np.concatenate((observation, one_hot)).astype(np.float32)


def _evaluate_initial_automaton_state(observation, abstract_mdp):
    """Consume the initial observation from the DFA pre-trace state and return the first active state."""
    initial_truth_assignment = abstract_mdp.get_environment_truth_assignment(observation)
    pre_trace_q = abstract_mdp.automaton.get_initial_q()
    return abstract_mdp.automaton.get_next_q(pre_trace_q, initial_truth_assignment)


def _format_counter(counter):
    """Convert a DFA transition counter into a compact human-readable string."""
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


def _write_run_header(log_handle, episodes, use_shaping, goal_reward, abstract_mdp, automaton_states, gamma_shaping, eval_interval, eval_episodes, eval_seed):
    """Write the configuration and DFA metadata at the beginning of a training run."""
    if not log_handle:
        return
    automaton = abstract_mdp.automaton
    shaping_formula = "gamma_shaping*Phi(next)-Phi(state)"
    header = (
        "\n=== NEW RUN ===\n"
        f"episodes={episodes}, shaping={use_shaping}, goal_reward={goal_reward}, gamma={abstract_mdp.gamma}\n"
        f"gamma_shaping={gamma_shaping}, shaping_formula={shaping_formula}\n"
        f"eval_interval={eval_interval}, eval_episodes={eval_episodes}, eval_seed={eval_seed}\n"
        f"inter_level_shaping={abstract_mdp.upper_level_mdp is not None}, "
        "inter_level_formula=gamma*Phi(next)-Phi(state)\n"
        f"formula={automaton.formula_str}\n"
        f"regions={{{', '.join(f'{name}: {region.as_dict()}' for name, region in abstract_mdp.regions.items())}}}\n"
        f"dfa_states={automaton_states}, pre_trace={automaton.get_initial_q()}, accepting={sorted(automaton.accepting_states)}, failure={sorted(automaton.failure_states)}\n"
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
    """Build a report containing recent metrics and cumulative DFA counters."""
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
    tabular_metrics = agent.metrics_snapshot() if isinstance(agent, TabularQLearner) else None
    tabular_diagnostics = agent.consume_diagnostics() if isinstance(agent, TabularQLearner) else None
    tabular_line = f"tabular Q-table             : {tabular_metrics['table_size']} states, {tabular_metrics['updated_state_actions']} updated pairs, coverage={tabular_metrics['state_action_coverage']:.3%}, positive updates={tabular_metrics['positive_updates']}\ntabular updates in window   : {tabular_diagnostics['updates']}, |TD error| mean/max={tabular_diagnostics['mean_abs_td_error']:.4g}/{tabular_diagnostics['max_abs_td_error']:.4g}, positive={tabular_diagnostics['positive_update_fraction']:.3%}\n" if tabular_metrics is not None else ""

    return (
        "\n"
        f"[Episode {episode + 1}/{episodes} | last {window}]\n"
        f"success rate                : {np.mean(histories['successes'][recent_slice]):.1%} (cumulative {np.mean(histories['successes']):.1%})\n"
        f"failure rate                : {np.mean(histories['failures'][recent_slice]):.1%} (cumulative {np.mean(histories['failures']):.1%})\n"
        f"synthetic task reward       : {np.mean(histories['task_rewards'][recent_slice]):.3f}\n"
        f"shaping reward              : {np.mean(histories['shaping_rewards'][recent_slice]):.3f}\n"
        f"learning reward             : {np.mean(histories['learning_rewards'][recent_slice]):.3f}\n"
        f"episode length              : {np.mean(histories['episode_lengths'][recent_slice]):.1f}\n"
        f"abstract changes / episode  : {np.mean(histories['abstract_changes'][recent_slice]):.1f}\n"
        f"DFA transitions / episode   : {np.mean(histories['dfa_transitions'][recent_slice]):.2f}\n"
        f"DFA transitions in window   : {_format_counter(recent_transitions)}\n"
        f"epsilon (next episode)       : {histories['epsilons'][-1]:.5f}\n"
        f"replay buffer                : {len(agent.memory)} samples [{buffer_details}]\n"
        f"{tabular_line}"
        f"DFA state visits in window   : {recent_visits_details}\n"
        f"DFA state visits cumulative  : {cumulative_visits_details}\n"
        f"DFA state entries in window  : {recent_entries_details}\n"
        f"DFA state entries cumulative : {cumulative_entries_details}\n"
        f"transitions cumulative       : {_format_counter(cumulative_counters['transitions'])}\n"
        f"accepted directly from s0    : {cumulative_counters['initial_acceptances']}\n"
        f"Gym endings cumulative       : terminated={cumulative_counters['env_terminated']}, truncated={cumulative_counters['env_truncated']}\n"
    )


def _save_named_policy(agent, policy_name):
    """Save the current policy using a stable descriptive filename."""
    category = "best" if policy_name.startswith("best_policy") else "last"
    os.makedirs(os.path.join(agent.policy_dir, category), exist_ok=True)
    agent.policy_name = os.path.join(category, policy_name)
    agent._save_policy()


def _policy_extension(agent):
    return "pkl" if isinstance(agent, TabularQLearner) else "pth"


def _monitoring_average(values, episode, log_interval):
    """Return the mean over the active monitoring window."""
    window = min(log_interval, episode + 1)
    return float(np.mean(values[-window:]))


def _greedy_action(agent, augmented_state, return_known=False):
    """Select a greedy action without changing the learner exploration state."""
    if isinstance(agent, TabularQLearner):
        return agent.greedy_action(augmented_state, return_known=return_known)
    with torch.inference_mode():
        state_tensor = torch.as_tensor(augmented_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        action = agent.policy_net(state_tensor).argmax(dim=1).item()
    return (action, True) if return_known else action


def _evaluate_agent_greedily(agent, abstract_mdp, episodes, goal_reward, seed):
    """Evaluate one learner without exploration, replay writes, or updates."""
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    successes = 0
    failures = 0
    task_rewards = []
    episode_lengths = []
    transition_counts = Counter()
    evaluation_env = gym.make("LunarLander-v3", continuous=False)
    is_neural = hasattr(agent, "policy_net")
    tabular_rng_state = agent.random_rng.getstate() if isinstance(agent, TabularQLearner) else None
    if isinstance(agent, TabularQLearner):
        agent.random_rng.seed(seed)
    was_training = agent.policy_net.training if is_neural else False
    if is_neural:
        agent.policy_net.eval()
    known_states = 0
    evaluated_states = 0
    try:
        for evaluation_episode in range(episodes):
            raw_state, _ = evaluation_env.reset(seed=seed + evaluation_episode)
            q = _evaluate_initial_automaton_state(raw_state, abstract_mdp)
            succeeded = automaton.is_goal_reached(q)
            failed = automaton.is_failure(q)
            terminated = truncated = False
            steps = 0
            while not (succeeded or failed or terminated or truncated):
                augmented_state = _augment_state(raw_state, q, state_to_index)
                action, known = _greedy_action(agent, augmented_state, return_known=True)
                known_states += int(known)
                evaluated_states += 1
                next_raw_state, _ignored_reward, terminated, truncated, _ = evaluation_env.step(action)
                previous_q = q
                q = automaton.get_next_q(previous_q, abstract_mdp.get_environment_truth_assignment(next_raw_state))
                if q not in state_to_index:
                    raise RuntimeError(f"DFA returned unknown evaluation state {q!r}")
                if q != previous_q:
                    transition_counts[(previous_q, q)] += 1
                succeeded = automaton.is_goal_reached(q)
                failed = automaton.is_failure(q)
                raw_state = next_raw_state
                steps += 1
            successes += int(succeeded)
            failures += int(failed)
            task_rewards.append(float(goal_reward) if succeeded else 0.0)
            episode_lengths.append(steps)
    finally:
        if is_neural and was_training:
            agent.policy_net.train()
        if tabular_rng_state is not None:
            agent.random_rng.setstate(tabular_rng_state)
        evaluation_env.close()
    return {"success_rate": successes / episodes, "failure_rate": failures / episodes, "mean_task_reward": float(np.mean(task_rewards)), "mean_episode_length": float(np.mean(episode_lengths)), "transition_counts": transition_counts, "known_state_fraction": known_states / evaluated_states if evaluated_states else 1.0}


def _evaluation_score(metrics):
    """Order evaluations by success, task reward, then shorter episodes."""
    return (metrics["success_rate"], metrics["mean_task_reward"], -metrics["mean_episode_length"])


def _validate_training_setup(automaton, state_to_index, episodes, log_interval, eval_interval, eval_episodes):
    """Validate DFA consistency and the numeric parameters required by training."""
    if automaton.get_initial_q() not in state_to_index:
        raise ValueError("The DFA initial state is missing from automaton.states")
    if not automaton.accepting_states.issubset(state_to_index):
        raise ValueError("At least one accepting DFA state is missing from automaton.states")
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if log_interval <= 0:
        raise ValueError("log_interval must be greater than zero")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be greater than zero")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be greater than zero")


def _build_training_results(histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode, gamma_shaping):
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
        "failures": histories["failures"],
        "initial_acceptances": initial_acceptance_history,
        "episode_lengths": histories["episode_lengths"],
        "abstract_changes": histories["abstract_changes"],
        "dfa_transitions": histories["dfa_transitions"],
        "automaton_states": automaton_states,
        "best_mean_learning_reward": best_mean_reward,
        "best_mean_eval_task_reward": best_mean_reward,
        "best_policy_episode": best_policy_episode,
        "gamma_shaping": gamma_shaping,
        "evaluation_steps": histories["evaluation_steps"],
        "eval_success_rates": histories["eval_success_rates"],
        "eval_task_rewards": histories["eval_task_rewards"],
        "eval_episode_lengths": histories["eval_episode_lengths"],
        "eval_known_state_fractions": histories["eval_known_state_fractions"],
        "tabular_table_sizes": histories["tabular_table_sizes"],
        "tabular_visited_states": histories["tabular_visited_states"],
        "tabular_updated_state_actions": histories["tabular_updated_state_actions"],
        "tabular_state_action_coverage": histories["tabular_state_action_coverage"],
        "tabular_positive_updates": histories["tabular_positive_updates"],
    }


# ==============================
# Training loop
# ==============================

def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, gamma_shaping=1.0, log_file=None, log_interval=100, eval_interval=1000, eval_episodes=50, eval_seed=100000, seed=None, policy_suffix=""):
    """
    Train one DDQN or tabular agent with the LTLf automaton and one epsilon.

    The Gym reward is deliberately discarded. The learning reward is the
    synthetic goal reward plus potential-based shaping. Shaping is evaluated
    on every transition, including terminal success and failure transitions.
    """
    # Build a stable mapping between DFA states and learner features.
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    num_states = len(automaton_states)

    # Fail early if the DFA or training parameters are inconsistent.
    _validate_training_setup(automaton, state_to_index, episodes, log_interval, eval_interval, eval_episodes)
    if not 0.0 < gamma_shaping <= 1.0:
        raise ValueError("gamma_shaping must be in the interval (0, 1]")

    # Store episode-level metrics for plots and post-processing.
    task_reward_history = []
    learning_reward_history = []
    shaping_reward_history = []
    epsilon_history = []
    episode_length_history = []
    success_history = []
    failure_history = []
    initial_acceptance_history = []
    abstract_change_history = []
    dfa_transition_history = []
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
        "failures": failure_history,
        "abstract_changes": abstract_change_history,
        "dfa_transitions": dfa_transition_history,
        "transition_counters": transition_counter_history,
        "state_visits": state_visit_histories,
        "state_entries": state_entry_histories,
        "tabular_table_sizes": [],
        "tabular_visited_states": [],
        "tabular_updated_state_actions": [],
        "tabular_state_action_coverage": [],
        "tabular_positive_updates": [],
        "evaluation_steps": [],
        "eval_success_rates": [],
        "eval_task_rewards": [],
        "eval_episode_lengths": [],
        "eval_known_state_fractions": [],
    }

    # Keep cumulative counters for diagnostics shown during training.
    cumulative_state_visits = Counter()
    cumulative_state_entries = Counter()
    cumulative_transitions = Counter()
    cumulative_env_terminated = 0
    cumulative_env_truncated = 0
    cumulative_initial_acceptances = 0
    best_mean_reward = -np.inf
    best_policy_episode = 0
    best_evaluation_score = None

    # Open one append-only log file for the complete run.
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
    _write_run_header(log_handle, episodes, use_shaping, goal_reward, abstract_mdp, automaton_states, gamma_shaping, eval_interval, eval_episodes, eval_seed)

    try:
        for episode in range(episodes):
            evaluation_due = _is_evaluation_due(episode, episodes, eval_interval)
            # Reset the environment and consume s0 before selecting the first action.
            raw_state, _ = env.reset(seed=seed if episode == 0 else None)
            q = _evaluate_initial_automaton_state(raw_state, abstract_mdp)
            if q not in state_to_index:
                raise RuntimeError(f"DFA returned unknown initial state {q!r} after evaluating s0")
            augmented_state = _augment_state(raw_state, q, state_to_index)

            # Reset counters local to the current episode.
            succeeded = automaton.is_goal_reached(q)
            failed = automaton.is_failure(q)
            episode_done = succeeded or failed
            episode_steps = 0
            episode_task_reward = float(goal_reward) if succeeded else 0.0
            episode_shaping_reward = 0.0
            episode_abstract_changes = 0
            episode_dfa_transitions = 0
            episode_state_visits = [0] * num_states
            episode_state_visits[state_to_index[q]] = 1
            # Count s0 as an entry from the virtual pre-trace state.
            episode_state_entries = [0] * num_states
            episode_state_entries[state_to_index[q]] = 1
            episode_transitions = Counter()
            if succeeded:
                cumulative_initial_acceptances += 1
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

                # Advance the DFA using propositions true in the arrival state.
                truth_assignment = abstract_mdp.get_environment_truth_assignment(next_raw_state)
                next_q = automaton.get_next_q(q, truth_assignment)
                if next_q not in state_to_index:
                    raise RuntimeError(f"DFA returned unknown state {next_q!r} from state {q!r}")

                # Count every arrival in a DFA state, including self-transitions.
                episode_state_visits[state_to_index[next_q]] += 1
                cumulative_state_visits[next_q] += 1

                # Track physical abstraction changes separately from DFA changes.
                abstract_next_state = (next_x, next_y, next_q)
                abstract_changed = abstract_state != abstract_next_state
                dfa_changed = next_q != q

                if abstract_changed:
                    episode_abstract_changes += 1
                if dfa_changed:
                    transition = (q, next_q)
                    episode_dfa_transitions += 1
                    episode_state_entries[state_to_index[next_q]] += 1
                    episode_transitions[transition] += 1
                    cumulative_state_entries[next_q] += 1
                    cumulative_transitions[transition] += 1

                # Assign the synthetic task reward only on DFA acceptance.
                synthetic_goal_reward = 0.0
                if automaton.is_goal_reached(next_q):
                    synthetic_goal_reward = float(goal_reward)
                    succeeded = True
                failed = automaton.is_failure(next_q)

                # Stop data collection on any Gym ending, DFA success, or
                # irreversible DFA failure.
                # A truncation (for example Gym's time limit) ends data
                # collection, but it is not an MDP terminal state: the learner must
                # still bootstrap from its final observation.
                episode_done = env_terminated or env_truncated or succeeded or failed
                bootstrap_terminal = env_terminated or succeeded or failed
                next_augmented_state = _augment_state(next_raw_state, next_q, state_to_index)

                # Evaluate the potential difference on every environment step.
                # With gamma_shaping=1, unchanged abstract states yield zero,
                # reproducing the previous cell-change heuristic exactly.
                shaping_signal = 0.0
                if use_shaping:
                    phi_state = abstract_mdp.v_star.get(abstract_state, 0.0)
                    phi_next_state = abstract_mdp.v_star.get(abstract_next_state, 0.0)
                    shaping_signal = gamma_shaping * phi_next_state - phi_state

                # Store the transition and perform one learner update.
                learning_reward = synthetic_goal_reward + shaping_signal
                agent.memory.push(augmented_state, action, learning_reward, next_augmented_state, bootstrap_terminal)
                if isinstance(agent, TabularQLearner):
                    agent.update(augmented_state, action, learning_reward, next_augmented_state, bootstrap_terminal)
                else:
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
            failure_history.append(int(failed))
            initial_acceptance_history.append(int(episode_steps == 0 and succeeded))
            abstract_change_history.append(episode_abstract_changes)
            dfa_transition_history.append(episode_dfa_transitions)
            transition_counter_history.append(episode_transitions)
            tabular_metrics = agent.metrics_snapshot() if isinstance(agent, TabularQLearner) else {"table_size": np.nan, "visited_states": np.nan, "updated_state_actions": np.nan, "state_action_coverage": np.nan, "positive_updates": np.nan}
            histories["tabular_table_sizes"].append(tabular_metrics["table_size"])
            histories["tabular_visited_states"].append(tabular_metrics["visited_states"])
            histories["tabular_updated_state_actions"].append(tabular_metrics["updated_state_actions"])
            histories["tabular_state_action_coverage"].append(tabular_metrics["state_action_coverage"])
            histories["tabular_positive_updates"].append(tabular_metrics["positive_updates"])

            # Record replay-buffer composition, state visits, and entries from other states.
            for index in range(num_states):
                buffer_histories[index].append(agent.memory.q_fraction_onehot(index, num_states))
                state_visit_histories[index].append(episode_state_visits[index])
                state_entry_histories[index].append(episode_state_entries[index])

            # Print recent and cumulative diagnostics at the requested interval.
            if _should_log(episode, episodes, log_interval) or evaluation_due:
                cumulative_counters = {"state_visits": cumulative_state_visits, "state_entries": cumulative_state_entries, "transitions": cumulative_transitions, "initial_acceptances": cumulative_initial_acceptances, "env_terminated": cumulative_env_terminated, "env_truncated": cumulative_env_truncated}
                _write_log(_build_training_log(episode, episodes, log_interval, automaton_states, agent, histories, cumulative_counters), log_handle)

            if evaluation_due:
                _write_log(f"\nStarting autonomous greedy evaluation at episode {episode + 1} ({eval_episodes} fixed-seed episodes)...\n", log_handle)
                evaluation = _evaluate_agent_greedily(agent, abstract_mdp, eval_episodes, goal_reward, eval_seed)
                histories["evaluation_steps"].append(episode + 1)
                histories["eval_success_rates"].append(evaluation["success_rate"])
                histories["eval_task_rewards"].append(evaluation["mean_task_reward"])
                histories["eval_episode_lengths"].append(evaluation["mean_episode_length"])
                histories["eval_known_state_fractions"].append(evaluation["known_state_fraction"])
                known_line = f", known states={evaluation['known_state_fraction']:.1%}" if isinstance(agent, TabularQLearner) else ""
                _write_log(f"[Greedy evaluation at episode {episode + 1} | {eval_episodes} fixed-seed episodes]\nsuccess={evaluation['success_rate']:.1%}, failure={evaluation['failure_rate']:.1%}, task reward={evaluation['mean_task_reward']:.3f}, length={evaluation['mean_episode_length']:.1f}{known_line}\nDFA transitions: {_format_counter(evaluation['transition_counts'])}\n", log_handle)
                score = _evaluation_score(evaluation)
                if best_evaluation_score is None or score > best_evaluation_score:
                    best_evaluation_score = score
                    best_mean_reward = evaluation["mean_task_reward"]
                    best_policy_episode = episode + 1
                    if save_policy:
                        _save_named_policy(agent, f"best_policy{policy_suffix}.{_policy_extension(agent)}")
                    _write_log(f"Best policy updated from autonomous greedy evaluation at episode {best_policy_episode}.\n", log_handle)

        # Save the final policy independently from its monitored performance.
        if save_policy:
            _save_named_policy(agent, f"last_policy{policy_suffix}.{_policy_extension(agent)}")
            _write_log(f"Last policy saved after episode {episodes}. Best greedy evaluation: episode {best_policy_episode}, mean task reward={best_mean_reward:.3f}\n", log_handle)
    finally:
        # Always close the log, including when training raises an exception.
        if log_handle:
            log_handle.close()

    # Return named histories to avoid ambiguous tuple positions.
    return _build_training_results(histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode, gamma_shaping)


# ==============================
# Experiment setup and outputs
# ==============================

def main(args):
    """Configure the experiment, run or load training, and generate diagnostic plots."""
    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be greater than zero")
    if args.learner == "tabular" and args.stochastic_bellman_update:
        raise ValueError("--stochastic-bellman-update is only valid with --learner ddqn; use --tabular-alpha for tabular Q-learning")
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

    # Load the temporal task and optional training parameters.
    config_path = _resolve_config_path( args.config, "trajectory.json", experiment_dir, args.post_process )
    abstraction_config_path = _resolve_config_path( args.abstraction_config, "abstraction.json", experiment_dir, args.post_process, )
    print(f"Task configuration: {config_path}")
    print(f"Abstraction configuration: {abstraction_config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    abstraction_config = AbstractionConfig.load(abstraction_config_path)
    if not args.post_process:
        _archive_config(config_path, experiment_dir, "trajectory.json")
        _archive_config(abstraction_config_path, experiment_dir, "abstraction.json")

    formula = config.get("formula", "F(goal)")
    regions, spatial_predicates, task_propositions = load_task_propositions(config.get("regions"), config.get("predicates"))
    gamma = float(config.get("gamma", 0.99))
    goal_reward = float(config.get("goal_reward", 10000))
    # Build the DFA once for both training and post-processing.
    automaton = LTLfAutomaton(formula)
    validation_report = validate_automaton( automaton, task_propositions, )
    level_summary = ", ".join(
        f"{index}:{level.name}={level.width}x{level.height}[checkpoint={level.checkpoint}]"
        if level.checkpoint is not None
        else f"{index}:{level.name}={level.width}x{level.height}[{abstraction_config.algorithm_for_index(index - 1)}, V={level.value_function_method}]"
        for index, level in enumerate(abstraction_config.levels, start=1)
    )
    bellman_summary = "not applicable" if args.learner == "tabular" else str(args.stochastic_bellman_update)
    if args.learner == "ddqn" and args.stochastic_bellman_update:
        bellman_summary += f" (alpha={args.bellman_alpha})"
    print( "=== LTLf TRAINING (single epsilon) ===\n" f"Learner: {args.learner}\n" f"Formula: {formula}\n" f"Regions: { {name: region.as_dict() for name, region in regions.items()} }\n" f"Predicates: { {name: predicate.as_dict() for name, predicate in spatial_predicates.items()} }\n" f"Abstractions: {level_summary}\n" "Inter-level shaping: gamma*Phi(next)-Phi(state)\n" f"Training gamma_shaping: {args.gamma_shaping}\n" f"Stochastic Bellman update: {bellman_summary}\n" "Automaton coordinates and training potential: level1\n" f"DFA: states={automaton.states}, pre-trace={automaton.initial_state}, " f"accepting={sorted(automaton.accepting_states)}, failure={sorted(automaton.failure_states)}\n" "Gym reward is ignored by design.\n" f"{validation_report.format()}" )

    if not args.post_process:
        automaton.render_graph(directory=image_dir)

    # Heatmaps depend only on the saved task configuration, not on agent training.
    multilevel_mdp = MultiLevelWaypointMDP( regions=task_propositions, ltlf_automaton=automaton, abstraction_config=abstraction_config, gamma=gamma, goal_reward=goal_reward, )
    multilevel_mdp.compute_value_functions(learning_log_dir=os.path.join(log_dir, "abstract_learning"))
    save_multilevel_value_functions(multilevel_mdp, output_root=os.path.join(data_dir, "abstract_value_functions"))
    if not args.no_heatmaps:
        save_multilevel_heatmaps( multilevel_mdp, filename_prefix="single_epsilon_exp", output_root=os.path.join(image_dir, "heatmaps"), annotate_cells=args.heatmap_annotation, )
    save_abstract_learning_curves(multilevel_mdp, output_root=os.path.join(image_dir, "abstract_learning"), smoothing_window=args.plot_window)
    abstract_mdp = multilevel_mdp.primary_mdp

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
                if args.learner == "tabular":
                    agent = TabularQLearner(env=env, num_phases=len(automaton.states), eps_decay=args.eps_decay, gamma=gamma, alpha=args.tabular_alpha, policy_dir=policy_dir, random_seed=run_seed)
                else:
                    agent = HierarchicalDQNLearner(env=env, max_episodes=args.episodes, eps_decay=args.eps_decay, gamma=gamma, extra_state_dims=len(automaton.states), use_polyak=args.polyak, tau=args.polyak_tau, target_update_freq=args.target_update_freq, network_type=args.network_type, policy_dir=policy_dir, stochastic_bellman_update=args.stochastic_bellman_update, bellman_alpha=args.bellman_alpha)
                policy_suffix = "" if args.num_seeds == 1 else f"_seed_{run_seed}"
                metrics = run_sequential_training(env=env, agent=agent, abstract_mdp=abstract_mdp, episodes=args.episodes, goal_reward=goal_reward, use_shaping=not args.no_shaping, gamma_shaping=args.gamma_shaping, log_file=f"{log_dir}/single_epsilon_training_seed_{run_seed}.log", log_interval=args.log_interval, eval_interval=args.eval_interval, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed, seed=run_seed, policy_suffix=policy_suffix)
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
    has_evaluations = all(
        key in data
        for key in (
            "evaluation_steps",
            "eval_success_rates",
            "eval_task_rewards",
            "eval_episode_lengths",
        )
    )
    if has_evaluations:
        evaluation_steps_runs = data["evaluation_steps_runs"] if "evaluation_steps_runs" in data else data["evaluation_steps"][np.newaxis, ...]
        eval_success_runs = data["eval_success_rates_runs"] if "eval_success_rates_runs" in data else data["eval_success_rates"][np.newaxis, ...]
        eval_reward_runs = data["eval_task_rewards_runs"] if "eval_task_rewards_runs" in data else data["eval_task_rewards"][np.newaxis, ...]
        eval_length_runs = data["eval_episode_lengths_runs"] if "eval_episode_lengths_runs" in data else data["eval_episode_lengths"][np.newaxis, ...]
    for obsolete_name in ("buffer_fractions_single_epsilon.png", "reward_breakdown_single_epsilon.png"):
        (Path(plot_dir) / obsolete_name).unlink(missing_ok=True)
    for run_index, (run_seed, task_rewards, learning_rewards, epsilon_history) in enumerate(zip(seed_values, task_reward_runs, learning_reward_runs, epsilon_runs)):
        seed_plot_dir = os.path.join(plot_dir, f"seed_{int(run_seed)}")
        os.makedirs(seed_plot_dir, exist_ok=True)
        plot_shaping_reward_breakdown(task_rewards, learning_rewards, epsilon_history, window_size=args.plot_window, filename=f"{seed_plot_dir}/reward_breakdown_single_epsilon.png", title=f"Reward Breakdown — Seed {int(run_seed)}")
        if has_evaluations:
            plot_evaluation_performance(evaluation_steps_runs[run_index], eval_success_runs[run_index], eval_reward_runs[run_index], eval_length_runs[run_index], filename=f"{seed_plot_dir}/evaluation_performance.png", title=f"Greedy Evaluation — Seed {int(run_seed)}")
        if run_index < len(buffer_runs):
            plot_buffer_fractions(buffer_runs[run_index], filename=f"{seed_plot_dir}/buffer_fractions_single_epsilon.png", window_size=args.plot_window, state_labels=data["automaton_states"], title=f"Replay Buffer Composition — Seed {int(run_seed)}")
    plot_training_variance( learning_reward_runs, window_size=args.plot_window, filename=f"{plot_dir}/training_variance_single_epsilon.png", epsilon_histories=epsilon_runs, )
    if has_evaluations:
        plot_evaluation_performance(evaluation_steps_runs, eval_success_runs, eval_reward_runs, eval_length_runs, filename=f"{plot_dir}/evaluation_performance.png", title="Greedy Evaluation Across Seeds")
    plot_buffer_variance(buffer_runs, window_size=args.plot_window, filename=f"{plot_dir}/buffer_variance_single_epsilon.png", state_labels=data["automaton_states"])
    tabular_table_runs = data["tabular_table_sizes_runs"] if "tabular_table_sizes_runs" in data else data["tabular_table_sizes"][np.newaxis, ...] if "tabular_table_sizes" in data else None
    if tabular_table_runs is not None and np.isfinite(tabular_table_runs).any():
        tabular_pair_runs = data["tabular_updated_state_actions_runs"] if "tabular_updated_state_actions_runs" in data else data["tabular_updated_state_actions"][np.newaxis, ...]
        tabular_coverage_runs = data["tabular_state_action_coverage_runs"] if "tabular_state_action_coverage_runs" in data else data["tabular_state_action_coverage"][np.newaxis, ...]
        tabular_positive_runs = data["tabular_positive_updates_runs"] if "tabular_positive_updates_runs" in data else data["tabular_positive_updates"][np.newaxis, ...]
        plot_tabular_training_diagnostics(tabular_table_runs, tabular_pair_runs, tabular_coverage_runs, tabular_positive_runs, filename=f"{plot_dir}/tabular_training_diagnostics.png")
    print("\nFinished.")


# ==============================
# Command-line entry point
# ==============================

if __name__ == "__main__":
    # Expose the main training and post-processing options.
    parser = argparse.ArgumentParser(description="LTLf DDQN or tabular Q-learning with one global epsilon.")
    parser.add_argument("--experiment-name", type=_experiment_name, required=True, help="Output directory name under results/.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-seeds", type=_positive_int, default=1, help="Number of training runs with consecutive seeds.")
    parser.add_argument("--seed", type=int, default=42, help="First training seed.")
    parser.add_argument("--config", default="trajectory.json")
    parser.add_argument( "--abstraction-config", default="abstraction.json", help="Ordered grid hierarchy (level1 defines automaton coordinates).", )
    parser.add_argument("--eps-decay", type=float, default=0.9996)
    parser.add_argument("--learner", choices=["ddqn", "tabular"], default="ddqn", help="Ground learner used for action selection and updates.")
    parser.add_argument("--tabular-alpha", type=_learning_rate, default=0.1, help="Learning rate used when --learner tabular is selected.")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=_positive_int, default=1000, help="Run autonomous greedy evaluation every N training episodes.")
    parser.add_argument("--eval-episodes", type=_positive_int, default=50, help="Number of fixed-seed episodes used at each greedy evaluation.")
    parser.add_argument("--eval-seed", type=int, default=100000, help="First held-out seed reused at every greedy evaluation.")
    parser.add_argument("--plot-window", type=int, default=500)
    parser.add_argument( "--polyak", action=argparse.BooleanOptionalAction, default=True, help="Use Polyak target updates (disable with --no-polyak).", )
    parser.add_argument("--polyak-tau", type=float, default=0.005)
    parser.add_argument( "--target-update-freq", type=int, default=1000, help="Hard target-network update interval used with --no-polyak.", )
    parser.add_argument( "--network-type", choices=["standard", "dueling"], default="standard", help="Q-network architecture: standard MLP or dueling value/advantage streams.", )
    parser.add_argument( "--stochastic-bellman-update", action=argparse.BooleanOptionalAction, default=False, help=( "Use Q <- Q + alpha*(target_DDQN-Q) as the regression target " "(disabled by default)." ), )
    parser.add_argument( "--bellman-alpha", type=float, default=0.1, help="Alpha used by --stochastic-bellman-update (default: 0.1).", )
    parser.add_argument( "--gamma-shaping", type=_discount_factor, default=1.0, help="Discount used in Phi shaping (default: 1.0).", )
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--no-heatmaps", action="store_true", help="Skip abstract-potential heatmap generation.")
    parser.add_argument("--heatmap-annotation", action="store_true", help="Annotate heatmap cells with V-function values and DFA state changes (default: disabled).")
    parser.add_argument("--post-process", action="store_true")
    main(parser.parse_args())
