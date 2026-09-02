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
from dataclasses import dataclass
from pathlib import Path

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import numpy as np
import torch

from abstraction import AbstractionConfig
from abstract_mdps import MultiLevelWaypointMDP, build_task_automaton
from agent import HierarchicalDQNLearner, TabularQLearner
from automaton_validator import validate_automaton
from spatial_regions import load_task_propositions
from utils import (
    load_multilevel_postprocess_data,
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
    """Append a one-hot encoding of the current automaton state."""
    one_hot = np.zeros(len(state_to_index), dtype=np.float32)
    one_hot[state_to_index[q]] = 1.0
    return np.concatenate((observation, one_hot)).astype(np.float32)


def _evaluate_initial_automaton_state(observation, abstract_mdp):
    """Consume the initial observation and return the first active task state."""
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


def _write_run_header(log_handle, episodes, use_shaping, goal_reward, abstract_mdp, automaton_states, gamma_shaping, eval_interval, eval_episodes, eval_seed, has_unbiased_learner):
    """Write the configuration and DFA metadata at the beginning of a training run."""
    if not log_handle:
        return
    automaton = abstract_mdp.automaton
    shaping_formula = "gamma_shaping*Phi(next)-Phi(state)"
    header = (
        "\n=== NEW RUN ===\n"
        f"episodes={episodes}, shaping={use_shaping}, goal_reward={goal_reward}, gamma={abstract_mdp.gamma}\n"
        f"gamma_shaping={gamma_shaping}, shaping_formula={shaping_formula}\n"
        f"ground_unbiased_learner={has_unbiased_learner}\n"
        f"eval_interval={eval_interval}, eval_episodes={eval_episodes}, eval_seed={eval_seed}\n"
        f"inter_level_shaping={abstract_mdp.upper_level_mdp is not None}, "
        "inter_level_formula=gamma*Phi(next)-Phi(state)\n"
        f"task_type={automaton.task_type}, task={automaton.formula_str}\n"
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
        f"completed cycles / episode  : {np.mean(histories['completed_cycles'][recent_slice]):.3f}\n"
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
    category = "best" if policy_name.startswith("best_") else "last"
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
    completed_cycles = []
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
            episode_cycles = 0
            while not (failed or terminated or truncated or (succeeded and not automaton.is_continuing)):
                augmented_state = _augment_state(raw_state, q, state_to_index)
                action, known = _greedy_action(agent, augmented_state, return_known=True)
                known_states += int(known)
                evaluated_states += 1
                next_raw_state, _ignored_reward, terminated, truncated, _ = evaluation_env.step(action)
                previous_q = q
                automaton_step = automaton.advance(previous_q, abstract_mdp.get_environment_truth_assignment(next_raw_state))
                q = automaton_step.next_state
                if q not in state_to_index:
                    raise RuntimeError(f"DFA returned unknown evaluation state {q!r}")
                if q != previous_q:
                    transition_counts[(previous_q, q)] += 1
                succeeded = succeeded or automaton_step.succeeded
                failed = failed or automaton_step.failed
                episode_cycles += int(automaton_step.completed_cycle)
                raw_state = next_raw_state
                steps += 1
            successes += int(succeeded)
            failures += int(failed)
            completed_cycles.append(episode_cycles)
            task_rewards.append(float(goal_reward) * (episode_cycles if automaton.is_continuing else int(succeeded)))
            episode_lengths.append(steps)
    finally:
        if is_neural and was_training:
            agent.policy_net.train()
        if tabular_rng_state is not None:
            agent.random_rng.setstate(tabular_rng_state)
        evaluation_env.close()
    return {"success_rate": successes / episodes, "failure_rate": failures / episodes, "mean_task_reward": float(np.mean(task_rewards)), "mean_episode_length": float(np.mean(episode_lengths)), "mean_completed_cycles": float(np.mean(completed_cycles)), "transition_counts": transition_counts, "known_state_fraction": known_states / evaluated_states if evaluated_states else 1.0}


def _evaluation_score(metrics):
    """Order evaluations by success, task reward, then shorter episodes."""
    return (metrics["success_rate"], metrics["mean_task_reward"], -metrics["mean_episode_length"])


def _validate_training_setup(automaton, state_to_index, episodes, log_interval, eval_interval, eval_episodes):
    """Validate automaton consistency and numeric training parameters."""
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


def _build_training_results(histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode, gamma_shaping, include_unbiased=False, best_unbiased_mean_reward=-np.inf, best_unbiased_policy_episode=0):
    """Select and name the numeric histories returned by the training loop."""
    results = {
        "task_rewards": histories["task_rewards"],
        "learning_rewards": histories["learning_rewards"],
        "shaping_rewards": histories["shaping_rewards"],
        "epsilon_history": histories["epsilons"],
        "buffer_histories": buffer_histories,
        "state_visit_histories": histories["state_visits"],
        "state_entry_histories": histories["state_entries"],
        "successes": histories["successes"],
        "failures": histories["failures"],
        "completed_cycles": histories["completed_cycles"],
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
        "eval_completed_cycles": histories["eval_completed_cycles"],
        "eval_known_state_fractions": histories["eval_known_state_fractions"],
        "tabular_table_sizes": histories["tabular_table_sizes"],
        "tabular_visited_states": histories["tabular_visited_states"],
        "tabular_updated_state_actions": histories["tabular_updated_state_actions"],
        "tabular_state_action_coverage": histories["tabular_state_action_coverage"],
        "tabular_positive_updates": histories["tabular_positive_updates"],
    }
    if include_unbiased:
        results.update({
            "best_unbiased_mean_eval_task_reward": best_unbiased_mean_reward,
            "best_unbiased_policy_episode": best_unbiased_policy_episode,
            "unbiased_eval_success_rates": histories["unbiased_eval_success_rates"],
            "unbiased_eval_task_rewards": histories["unbiased_eval_task_rewards"],
            "unbiased_eval_episode_lengths": histories["unbiased_eval_episode_lengths"],
            "unbiased_eval_completed_cycles": histories["unbiased_eval_completed_cycles"],
            "unbiased_eval_known_state_fractions": histories["unbiased_eval_known_state_fractions"],
        })
    return results


# ==============================
# Training loop
# ==============================

@dataclass
class TrainingContext:
    """Mutable state shared by the small phases of one complete training run."""

    env: object
    agent: object
    unbiased_agent: object
    abstract_mdp: object
    automaton: object
    automaton_states: list
    state_to_index: dict
    episodes: int
    goal_reward: float
    save_policy: bool
    use_shaping: bool
    gamma_shaping: float
    log_interval: int
    eval_interval: int
    eval_episodes: int
    eval_seed: int
    policy_suffix: str
    histories: dict
    initial_acceptance_history: list
    buffer_histories: list
    cumulative_state_visits: Counter
    cumulative_state_entries: Counter
    cumulative_transitions: Counter
    log_handle: object = None
    cumulative_env_terminated: int = 0
    cumulative_env_truncated: int = 0
    cumulative_initial_acceptances: int = 0
    best_evaluation_score: tuple | None = None
    best_mean_reward: float = -np.inf
    best_policy_episode: int = 0
    best_unbiased_evaluation_score: tuple | None = None
    best_unbiased_mean_reward: float = -np.inf
    best_unbiased_policy_episode: int = 0


@dataclass(frozen=True)
class TrainingStepResult:
    """All state changes and diagnostics produced by one environment step."""

    next_raw_state: object
    next_augmented_state: object
    next_q: object
    synthetic_goal_reward: float
    shaping_signal: float
    episode_done: bool
    succeeded: bool
    failed: bool
    completed_cycle: bool
    env_terminated: bool
    env_truncated: bool
    abstract_changed: bool
    dfa_changed: bool


@dataclass(frozen=True)
class TrainingEpisodeResult:
    """Episode-level measurements committed to the global histories together."""

    task_reward: float
    shaping_reward: float
    length: int
    succeeded: bool
    failed: bool
    completed_cycles: int
    abstract_changes: int
    dfa_transitions: int
    state_visits: list
    state_entries: list
    transitions: Counter
    next_epsilon: float


def _create_training_histories(automaton_states):
    """Create every rectangular history consumed by logging and post-processing."""
    state_visit_histories = [[] for _ in automaton_states]
    state_entry_histories = [[] for _ in automaton_states]
    histories = {
        "task_rewards": [], "learning_rewards": [], "shaping_rewards": [], "epsilons": [],
        "episode_lengths": [], "successes": [], "failures": [], "completed_cycles": [], "abstract_changes": [],
        "dfa_transitions": [], "transition_counters": [], "state_visits": state_visit_histories,
        "state_entries": state_entry_histories, "tabular_table_sizes": [], "tabular_visited_states": [],
        "tabular_updated_state_actions": [], "tabular_state_action_coverage": [], "tabular_positive_updates": [],
        "evaluation_steps": [], "eval_success_rates": [], "eval_task_rewards": [],
        "eval_episode_lengths": [], "eval_completed_cycles": [], "eval_known_state_fractions": [],
        "unbiased_eval_success_rates": [], "unbiased_eval_task_rewards": [],
        "unbiased_eval_episode_lengths": [], "unbiased_eval_completed_cycles": [],
        "unbiased_eval_known_state_fractions": [],
    }
    return histories, [], [[] for _ in automaton_states]


def _initialize_training_context(env, agent, abstract_mdp, episodes, goal_reward, save_policy, use_shaping, gamma_shaping, log_file, log_interval, eval_interval, eval_episodes, eval_seed, policy_suffix, unbiased_agent):
    """Validate one run and allocate the state shared by all training phases."""
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    _validate_training_setup(automaton, state_to_index, episodes, log_interval, eval_interval, eval_episodes)
    if not 0.0 < gamma_shaping <= 1.0:
        raise ValueError("gamma_shaping must be in the interval (0, 1]")
    histories, initial_acceptance_history, buffer_histories = _create_training_histories(automaton_states)
    if unbiased_agent is not None:
        if not use_shaping:
            raise ValueError("An unbiased observer is redundant when the primary learner does not use shaping")
        if isinstance(agent, TabularQLearner) != isinstance(unbiased_agent, TabularQLearner):
            raise TypeError("The biased and unbiased ground learners must use the same algorithm")
        if not isinstance(agent, TabularQLearner) and agent.batch_size != unbiased_agent.batch_size:
            raise ValueError("The biased and unbiased ground learners must use the same batch size")
        unbiased_agent.memory = agent.memory
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
    context = TrainingContext(env=env, agent=agent, unbiased_agent=unbiased_agent, abstract_mdp=abstract_mdp, automaton=automaton, automaton_states=automaton_states, state_to_index=state_to_index, episodes=episodes, goal_reward=float(goal_reward), save_policy=save_policy, use_shaping=use_shaping, gamma_shaping=gamma_shaping, log_interval=log_interval, eval_interval=eval_interval, eval_episodes=eval_episodes, eval_seed=eval_seed, policy_suffix=policy_suffix, histories=histories, initial_acceptance_history=initial_acceptance_history, buffer_histories=buffer_histories, cumulative_state_visits=Counter(), cumulative_state_entries=Counter(), cumulative_transitions=Counter(), log_handle=log_handle)
    _write_run_header(log_handle, episodes, use_shaping, goal_reward, abstract_mdp, automaton_states, gamma_shaping, eval_interval, eval_episodes, eval_seed, unbiased_agent is not None)
    return context


def _perform_training_step(context, raw_state, augmented_state, q):
    """Execute one action, advance the DFA, compute rewards, and update the learner."""
    action = context.agent.select_action(augmented_state)

    # Gym's reward is deliberately discarded: task completion is defined only
    # by the DFA, while the abstract V-function supplies the shaping signal.
    next_raw_state, _ignored_env_reward, env_terminated, env_truncated, _ = context.env.step(action)
    x, y = _abstract_position(raw_state, context.abstract_mdp)
    next_x, next_y = _abstract_position(next_raw_state, context.abstract_mdp)
    abstract_state = (x, y, q)
    abstract_next_state_without_q = (next_x, next_y)

    truth_assignment = context.abstract_mdp.get_environment_truth_assignment(next_raw_state)
    automaton_step = context.automaton.advance(q, truth_assignment)
    next_q = automaton_step.next_state
    if next_q not in context.state_to_index:
        raise RuntimeError(f"DFA returned unknown state {next_q!r} from state {q!r}")
    abstract_next_state = (*abstract_next_state_without_q, next_q)

    succeeded = automaton_step.succeeded
    failed = automaton_step.failed
    synthetic_goal_reward = context.goal_reward if succeeded else 0.0

    # A Gym truncation stops data collection but still permits bootstrap. True
    # environment terminals and DFA terminals do not have a continuation value.
    episode_done = env_terminated or env_truncated or automaton_step.terminal
    bootstrap_terminal = env_terminated or automaton_step.terminal
    next_augmented_state = _augment_state(next_raw_state, next_q, context.state_to_index)

    # Potential-based shaping is evaluated even on the last collected step.
    # With gamma_shaping=1 an unchanged product state contributes exactly zero.
    shaping_signal = 0.0
    if context.use_shaping:
        phi_state = context.abstract_mdp.v_star.get(abstract_state, 0.0)
        phi_next_state = context.abstract_mdp.v_star.get(abstract_next_state, 0.0)
        shaping_signal = context.gamma_shaping * phi_next_state - phi_state

    learning_reward = synthetic_goal_reward + shaping_signal
    context.agent.memory.push(
        augmented_state,
        action,
        learning_reward,
        next_augmented_state,
        bootstrap_terminal,
        task_reward=synthetic_goal_reward,
    )
    if isinstance(context.agent, TabularQLearner):
        context.agent.update(augmented_state, action, learning_reward, next_augmented_state, bootstrap_terminal)
        if context.unbiased_agent is not None:
            context.unbiased_agent.update(augmented_state, action, synthetic_goal_reward, next_augmented_state, bootstrap_terminal)
    elif context.unbiased_agent is not None:
        if len(context.agent.memory) >= context.agent.batch_size:
            states, actions, biased_rewards, task_rewards, next_states, dones = context.agent.memory.sample_dual(context.agent.batch_size)
            context.agent.optimize_model((states, actions, biased_rewards, next_states, dones))
            context.unbiased_agent.optimize_model((states, actions, task_rewards, next_states, dones))
    else:
        context.agent.optimize_model()

    return TrainingStepResult(next_raw_state=next_raw_state, next_augmented_state=next_augmented_state, next_q=next_q, synthetic_goal_reward=synthetic_goal_reward, shaping_signal=shaping_signal, episode_done=episode_done, succeeded=succeeded, failed=failed, completed_cycle=automaton_step.completed_cycle, env_terminated=env_terminated, env_truncated=env_truncated, abstract_changed=abstract_state != abstract_next_state, dfa_changed=q != next_q)


def _run_training_episode(context, reset_seed=None):
    """Run one complete episode and return metrics without writing global histories."""
    raw_state, _ = context.env.reset(seed=reset_seed)
    q = _evaluate_initial_automaton_state(raw_state, context.abstract_mdp)
    if q not in context.state_to_index:
        raise RuntimeError(f"DFA returned unknown initial state {q!r} after evaluating s0")
    augmented_state = _augment_state(raw_state, q, context.state_to_index)
    context.agent.eps = context.histories["epsilons"][-1] if context.histories["epsilons"] else context.agent.eps

    succeeded = context.automaton.is_goal_reached(q)
    failed = context.automaton.is_failure(q)
    episode_done = succeeded or failed
    task_reward = context.goal_reward if succeeded else 0.0
    shaping_reward = 0.0
    steps = 0
    abstract_changes = 0
    dfa_transitions = 0
    completed_cycles = 0
    state_visits = [0] * len(context.automaton_states)
    state_entries = [0] * len(context.automaton_states)
    state_visits[context.state_to_index[q]] = 1
    state_entries[context.state_to_index[q]] = 1
    transitions = Counter()

    # The initial observation is an entry from the virtual pre-trace state.
    context.cumulative_state_visits[q] += 1
    context.cumulative_state_entries[q] += 1
    if succeeded:
        context.cumulative_initial_acceptances += 1

    while not episode_done:
        previous_q = q
        step = _perform_training_step(context, raw_state, augmented_state, previous_q)
        state_visits[context.state_to_index[step.next_q]] += 1
        context.cumulative_state_visits[step.next_q] += 1
        abstract_changes += int(step.abstract_changed)

        if step.dfa_changed:
            transition = (previous_q, step.next_q)
            dfa_transitions += 1
            state_entries[context.state_to_index[step.next_q]] += 1
            transitions[transition] += 1
            context.cumulative_state_entries[step.next_q] += 1
            context.cumulative_transitions[transition] += 1

        task_reward += step.synthetic_goal_reward
        shaping_reward += step.shaping_signal
        steps += 1
        succeeded = succeeded or step.succeeded
        failed = failed or step.failed
        completed_cycles += int(step.completed_cycle)
        episode_done = step.episode_done
        raw_state = step.next_raw_state
        augmented_state = step.next_augmented_state
        q = step.next_q
        context.cumulative_env_terminated += int(step.env_terminated)
        context.cumulative_env_truncated += int(step.env_truncated)

    # Exploration decays exactly once, so every transition in an episode uses
    # the same epsilon and the stored value applies to the following episode.
    next_epsilon = max(context.agent.eps_min, context.agent.eps * context.agent.eps_decay)
    context.agent.eps = next_epsilon
    return TrainingEpisodeResult(task_reward=task_reward, shaping_reward=shaping_reward, length=steps, succeeded=succeeded, failed=failed, completed_cycles=completed_cycles, abstract_changes=abstract_changes, dfa_transitions=dfa_transitions, state_visits=state_visits, state_entries=state_entries, transitions=transitions, next_epsilon=next_epsilon)


def _record_training_episode(context, result):
    """Commit one completed episode atomically to every monitoring history."""
    histories = context.histories
    histories["task_rewards"].append(result.task_reward)
    histories["shaping_rewards"].append(result.shaping_reward)
    histories["learning_rewards"].append(result.task_reward + result.shaping_reward)
    histories["epsilons"].append(result.next_epsilon)
    histories["episode_lengths"].append(result.length)
    histories["successes"].append(int(result.succeeded))
    histories["failures"].append(int(result.failed))
    histories["completed_cycles"].append(result.completed_cycles)
    histories["abstract_changes"].append(result.abstract_changes)
    histories["dfa_transitions"].append(result.dfa_transitions)
    histories["transition_counters"].append(result.transitions)
    context.initial_acceptance_history.append(int(result.length == 0 and result.succeeded))

    tabular_metrics = context.agent.metrics_snapshot() if isinstance(context.agent, TabularQLearner) else {"table_size": np.nan, "visited_states": np.nan, "updated_state_actions": np.nan, "state_action_coverage": np.nan, "positive_updates": np.nan}
    histories["tabular_table_sizes"].append(tabular_metrics["table_size"])
    histories["tabular_visited_states"].append(tabular_metrics["visited_states"])
    histories["tabular_updated_state_actions"].append(tabular_metrics["updated_state_actions"])
    histories["tabular_state_action_coverage"].append(tabular_metrics["state_action_coverage"])
    histories["tabular_positive_updates"].append(tabular_metrics["positive_updates"])

    for index in range(len(context.automaton_states)):
        context.buffer_histories[index].append(context.agent.memory.q_fraction_onehot(index, len(context.automaton_states)))
        histories["state_visits"][index].append(result.state_visits[index])
        histories["state_entries"][index].append(result.state_entries[index])


def _training_cumulative_counters(context):
    """Expose cumulative diagnostics in the format expected by the logger."""
    return {"state_visits": context.cumulative_state_visits, "state_entries": context.cumulative_state_entries, "transitions": context.cumulative_transitions, "initial_acceptances": context.cumulative_initial_acceptances, "env_terminated": context.cumulative_env_terminated, "env_truncated": context.cumulative_env_truncated}


def _run_periodic_training_evaluation(context, episode):
    """Evaluate greedily, append metrics, and replace the best policy if needed."""
    _write_log(f"\nStarting autonomous greedy evaluation at episode {episode + 1} ({context.eval_episodes} fixed-seed episodes)...\n", context.log_handle)
    evaluation = _evaluate_agent_greedily(context.agent, context.abstract_mdp, context.eval_episodes, context.goal_reward, context.eval_seed)
    context.histories["evaluation_steps"].append(episode + 1)
    context.histories["eval_success_rates"].append(evaluation["success_rate"])
    context.histories["eval_task_rewards"].append(evaluation["mean_task_reward"])
    context.histories["eval_episode_lengths"].append(evaluation["mean_episode_length"])
    context.histories["eval_completed_cycles"].append(evaluation["mean_completed_cycles"])
    context.histories["eval_known_state_fractions"].append(evaluation["known_state_fraction"])
    known_line = f", known states={evaluation['known_state_fraction']:.1%}" if isinstance(context.agent, TabularQLearner) else ""
    cycle_line = f", cycles={evaluation['mean_completed_cycles']:.3f}" if context.automaton.is_continuing else ""
    _write_log(f"[Greedy evaluation at episode {episode + 1} | {context.eval_episodes} fixed-seed episodes]\nsuccess={evaluation['success_rate']:.1%}, failure={evaluation['failure_rate']:.1%}, task reward={evaluation['mean_task_reward']:.3f}{cycle_line}, length={evaluation['mean_episode_length']:.1f}{known_line}\nDFA transitions: {_format_counter(evaluation['transition_counts'])}\n", context.log_handle)

    score = _evaluation_score(evaluation)
    if context.best_evaluation_score is None or score > context.best_evaluation_score:
        context.best_evaluation_score = score
        context.best_mean_reward = evaluation["mean_task_reward"]
        context.best_policy_episode = episode + 1
        if context.save_policy:
            _save_named_policy(context.agent, f"best_policy{context.policy_suffix}.{_policy_extension(context.agent)}")
        _write_log(f"Best policy updated from autonomous greedy evaluation at episode {context.best_policy_episode}.\n", context.log_handle)

    if context.unbiased_agent is not None:
        unbiased_evaluation = _evaluate_agent_greedily(
            context.unbiased_agent,
            context.abstract_mdp,
            context.eval_episodes,
            context.goal_reward,
            context.eval_seed,
        )
        context.histories["unbiased_eval_success_rates"].append(unbiased_evaluation["success_rate"])
        context.histories["unbiased_eval_task_rewards"].append(unbiased_evaluation["mean_task_reward"])
        context.histories["unbiased_eval_episode_lengths"].append(unbiased_evaluation["mean_episode_length"])
        context.histories["unbiased_eval_completed_cycles"].append(unbiased_evaluation["mean_completed_cycles"])
        context.histories["unbiased_eval_known_state_fractions"].append(unbiased_evaluation["known_state_fraction"])
        unbiased_known_line = f", known states={unbiased_evaluation['known_state_fraction']:.1%}" if isinstance(context.unbiased_agent, TabularQLearner) else ""
        unbiased_cycle_line = f", cycles={unbiased_evaluation['mean_completed_cycles']:.3f}" if context.automaton.is_continuing else ""
        _write_log(
            f"[Unbiased greedy evaluation at episode {episode + 1} | {context.eval_episodes} fixed-seed episodes]\n"
            f"success={unbiased_evaluation['success_rate']:.1%}, failure={unbiased_evaluation['failure_rate']:.1%}, "
            f"task reward={unbiased_evaluation['mean_task_reward']:.3f}{unbiased_cycle_line}, "
            f"length={unbiased_evaluation['mean_episode_length']:.1f}{unbiased_known_line}\n"
            f"DFA transitions: {_format_counter(unbiased_evaluation['transition_counts'])}\n",
            context.log_handle,
        )
        unbiased_score = _evaluation_score(unbiased_evaluation)
        if context.best_unbiased_evaluation_score is None or unbiased_score > context.best_unbiased_evaluation_score:
            context.best_unbiased_evaluation_score = unbiased_score
            context.best_unbiased_mean_reward = unbiased_evaluation["mean_task_reward"]
            context.best_unbiased_policy_episode = episode + 1
            if context.save_policy:
                _save_named_policy(
                    context.unbiased_agent,
                    f"best_unbiased_policy{context.policy_suffix}.{_policy_extension(context.unbiased_agent)}",
                )
            _write_log(
                f"Best unbiased policy updated at episode {context.best_unbiased_policy_episode}.\n",
                context.log_handle,
            )


def _finalize_training(context):
    """Save the last policy and return the numeric histories exposed by the API."""
    if context.save_policy:
        _save_named_policy(context.agent, f"last_policy{context.policy_suffix}.{_policy_extension(context.agent)}")
        _write_log(f"Last policy saved after episode {context.episodes}. Best greedy evaluation: episode {context.best_policy_episode}, mean task reward={context.best_mean_reward:.3f}\n", context.log_handle)
        if context.unbiased_agent is not None:
            _save_named_policy(
                context.unbiased_agent,
                f"last_unbiased_policy{context.policy_suffix}.{_policy_extension(context.unbiased_agent)}",
            )
            _write_log(
                f"Last unbiased policy saved after episode {context.episodes}. "
                f"Best unbiased evaluation: episode {context.best_unbiased_policy_episode}, "
                f"mean task reward={context.best_unbiased_mean_reward:.3f}\n",
                context.log_handle,
            )
    return _build_training_results(
        context.histories,
        context.initial_acceptance_history,
        context.buffer_histories,
        context.automaton_states,
        context.best_mean_reward,
        context.best_policy_episode,
        context.gamma_shaping,
        include_unbiased=context.unbiased_agent is not None,
        best_unbiased_mean_reward=context.best_unbiased_mean_reward,
        best_unbiased_policy_episode=context.best_unbiased_policy_episode,
    )


def train(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, gamma_shaping=1.0, log_file=None, log_interval=100, eval_interval=1000, eval_episodes=1000, eval_seed=100000, seed=None, policy_suffix="", unbiased_agent=None):
    """Train one learner while delegating each lifecycle phase to a focused helper."""
    context = _initialize_training_context(env, agent, abstract_mdp, episodes, goal_reward, save_policy, use_shaping, gamma_shaping, log_file, log_interval, eval_interval, eval_episodes, eval_seed, policy_suffix, unbiased_agent)
    try:
        for episode in range(episodes):
            evaluation_due = _is_evaluation_due(episode, episodes, eval_interval)
            result = _run_training_episode(context, reset_seed=seed if episode == 0 else None)
            _record_training_episode(context, result)
            if _should_log(episode, episodes, log_interval) or evaluation_due:
                _write_log(_build_training_log(episode, episodes, log_interval, context.automaton_states, agent, context.histories, _training_cumulative_counters(context)), context.log_handle)
            if evaluation_due:
                _run_periodic_training_evaluation(context, episode)
        return _finalize_training(context)
    finally:
        # The log must be closed even when the environment or learner raises.
        if context.log_handle:
            context.log_handle.close()


# ==============================
# Experiment setup and outputs
# ==============================

def main(args):
    """Configure the experiment, run or load training, and generate diagnostic plots."""
    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be greater than zero")
    if args.learner == "tabular" and args.stochastic_bellman_update:
        raise ValueError("--stochastic-bellman-update is only valid with --learner ddqn; use --tabular-alpha for tabular Q-learning")
    if args.ground_unbiased_learner and args.no_shaping:
        raise ValueError("--ground-unbiased-learner requires shaping; without shaping the primary learner is already unbiased")
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

    regions, spatial_predicates, task_propositions = load_task_propositions(config.get("regions"), config.get("predicates"))
    gamma = float(config.get("gamma", 0.99))
    goal_reward = float(config.get("goal_reward", 10000))
    # Build the DFA once for both training and post-processing.
    automaton = build_task_automaton(config)
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
    print( "=== TEMPORAL TASK TRAINING (single epsilon) ===\n" f"Learner: {args.learner}\n" f"Ground unbiased learner: {args.ground_unbiased_learner}\n" f"Task type: {automaton.task_type}\n" f"Task: {automaton.formula_str}\n" f"Regions: { {name: region.as_dict() for name, region in regions.items()} }\n" f"Predicates: { {name: predicate.as_dict() for name, predicate in spatial_predicates.items()} }\n" f"Abstractions: {level_summary}\n" "Inter-level shaping: gamma*Phi(next)-Phi(state)\n" f"Training gamma_shaping: {args.gamma_shaping}\n" f"Stochastic Bellman update: {bellman_summary}\n" "Automaton coordinates and training potential: level1\n" f"Automaton: states={automaton.states}, initial={automaton.initial_state}, " f"accepting={sorted(automaton.accepting_states)}, failure={sorted(automaton.failure_states)}, continuing={automaton.is_continuing}\n" "Gym reward is ignored by design.\n" f"{validation_report.format()}" )

    automaton.render_graph(directory=image_dir)

    # Build the hierarchy, then either solve it or restore its saved outputs.
    multilevel_mdp = MultiLevelWaypointMDP( regions=task_propositions, ltlf_automaton=automaton, abstraction_config=abstraction_config, gamma=gamma, goal_reward=goal_reward, )
    if args.post_process:
        load_multilevel_postprocess_data(
            multilevel_mdp,
            value_root=os.path.join(data_dir, "abstract_value_functions"),
            learning_root=os.path.join(data_dir, "abstract_learning"),
        )
    else:
        multilevel_mdp.compute_value_functions(learning_log_dir=os.path.join(log_dir, "abstract_learning"))
        save_multilevel_value_functions(multilevel_mdp, output_root=os.path.join(data_dir, "abstract_value_functions"))
    save_abstract_learning_curves(
        multilevel_mdp,
        output_root=os.path.join(image_dir, "abstract_learning"),
        data_root=os.path.join(data_dir, "abstract_learning"),
        smoothing_window=args.plot_window,
        save_data=not args.post_process,
    )
    if not args.no_heatmaps:
        save_multilevel_heatmaps( multilevel_mdp, filename_prefix="single_epsilon_exp", output_root=os.path.join(image_dir, "heatmaps"), annotate_cells=args.heatmap_annotation, )
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
                    unbiased_agent = TabularQLearner(env=env, num_phases=len(automaton.states), eps_decay=args.eps_decay, gamma=gamma, alpha=args.tabular_alpha, policy_dir=os.path.join(policy_dir, "unbiased"), random_seed=run_seed) if args.ground_unbiased_learner else None
                else:
                    agent = HierarchicalDQNLearner(env=env, max_episodes=args.episodes, eps_decay=args.eps_decay, gamma=gamma, extra_state_dims=len(automaton.states), use_polyak=args.polyak, tau=args.polyak_tau, target_update_freq=args.target_update_freq, network_type=args.network_type, policy_dir=policy_dir, stochastic_bellman_update=args.stochastic_bellman_update, bellman_alpha=args.bellman_alpha)
                    unbiased_agent = HierarchicalDQNLearner(env=env, max_episodes=args.episodes, eps_decay=args.eps_decay, gamma=gamma, extra_state_dims=len(automaton.states), use_polyak=args.polyak, tau=args.polyak_tau, target_update_freq=args.target_update_freq, network_type=args.network_type, policy_dir=os.path.join(policy_dir, "unbiased"), stochastic_bellman_update=args.stochastic_bellman_update, bellman_alpha=args.bellman_alpha) if args.ground_unbiased_learner else None
                policy_suffix = "" if args.num_seeds == 1 else f"_seed_{run_seed}"
                metrics = train(env=env, agent=agent, abstract_mdp=abstract_mdp, episodes=args.episodes, goal_reward=goal_reward, use_shaping=not args.no_shaping, gamma_shaping=args.gamma_shaping, log_file=f"{log_dir}/single_epsilon_training_seed_{run_seed}.log", log_interval=args.log_interval, eval_interval=args.eval_interval, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed, seed=run_seed, policy_suffix=policy_suffix, unbiased_agent=unbiased_agent)
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
            "eval_task_rewards",
        )
    )
    if has_evaluations:
        evaluation_steps_runs = data["evaluation_steps_runs"] if "evaluation_steps_runs" in data else data["evaluation_steps"][np.newaxis, ...]
        eval_reward_runs = data["eval_task_rewards_runs"] if "eval_task_rewards_runs" in data else data["eval_task_rewards"][np.newaxis, ...]
        has_unbiased_evaluations = (
            "unbiased_eval_task_rewards_runs" in data
            and data["unbiased_eval_task_rewards_runs"].shape == eval_reward_runs.shape
            and data["unbiased_eval_task_rewards_runs"].shape[-1] > 0
        )
        unbiased_eval_reward_runs = data["unbiased_eval_task_rewards_runs"] if has_unbiased_evaluations else None
    for obsolete_name in ("buffer_fractions_single_epsilon.png", "reward_breakdown_single_epsilon.png"):
        (Path(plot_dir) / obsolete_name).unlink(missing_ok=True)
    for run_index, (run_seed, task_rewards, learning_rewards, epsilon_history) in enumerate(zip(seed_values, task_reward_runs, learning_reward_runs, epsilon_runs)):
        seed_plot_dir = os.path.join(plot_dir, f"seed_{int(run_seed)}")
        os.makedirs(seed_plot_dir, exist_ok=True)
        plot_shaping_reward_breakdown(task_rewards, learning_rewards, epsilon_history, window_size=args.plot_window, filename=f"{seed_plot_dir}/reward_breakdown_single_epsilon.png", title=f"Reward Breakdown — Seed {int(run_seed)}")
        if has_evaluations:
            plot_evaluation_performance(evaluation_steps_runs[run_index], eval_reward_runs[run_index], unbiased_task_rewards=unbiased_eval_reward_runs[run_index] if has_unbiased_evaluations else None, filename=f"{seed_plot_dir}/evaluation_performance.png", title=f"Greedy Evaluation — Seed {int(run_seed)}")
        if run_index < len(buffer_runs):
            plot_buffer_fractions(buffer_runs[run_index], filename=f"{seed_plot_dir}/buffer_fractions_single_epsilon.png", window_size=args.plot_window, state_labels=data["automaton_states"], title=f"Replay Buffer Composition — Seed {int(run_seed)}")
    plot_training_variance( learning_reward_runs, window_size=args.plot_window, filename=f"{plot_dir}/training_variance_single_epsilon.png", epsilon_histories=epsilon_runs, )
    if has_evaluations:
        plot_evaluation_performance(evaluation_steps_runs, eval_reward_runs, unbiased_task_rewards=unbiased_eval_reward_runs, filename=f"{plot_dir}/evaluation_performance.png", title="Greedy Evaluation Across Seeds")
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
    parser.add_argument("--eval-episodes", type=_positive_int, default=1000, help="Number of fixed-seed episodes used at each greedy evaluation.")
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
    parser.add_argument("--ground-unbiased-learner", action="store_true", help="Train an additional original-task-reward learner on the same batches collected by the shaped behavior policy.")
    parser.add_argument("--no-heatmaps", action="store_true", help="Skip abstract-potential heatmap generation.")
    parser.add_argument("--heatmap-annotation", action="store_true", help="Annotate heatmap cells with V-function values and DFA state changes (default: disabled).")
    parser.add_argument("--post-process", action="store_true")
    main(parser.parse_args())
