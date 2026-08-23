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
from agent import DualReplayBuffer, HierarchicalDQNLearner, TabularQLearner
from automaton_validator import validate_automaton
from spatial_regions import load_regions
from utils import (
    phi_mapping_sequential,
    plot_buffer_fractions,
    plot_buffer_variance,
    plot_dual_learner_evaluation,
    plot_tabular_learning_diagnostics,
    plot_shaping_reward_breakdown,
    plot_training_variance,
    save_multilevel_heatmaps,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == "src"
    else SCRIPT_DIR
)
HEAVY_DIAGNOSTICS_INTERVAL = 1000


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


def _positive_float(value):
    """Parse a strictly positive floating-point value."""
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _learning_rate(value):
    """Parse a learning rate in the tabular update interval."""
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("must be in the interval (0, 1]")
    return number


def _probability(value):
    """Parse a probability including the interval endpoints."""
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be in the interval [0, 1]")
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
    framework_default = Path(SCRIPT_DIR) / default_filename
    uses_default = (
        str(requested_path) == default_filename
        or requested.resolve() == framework_default.resolve()
    )
    candidates = []
    if post_process and uses_default:
        candidates.extend( [ Path(experiment_dir) / default_filename, Path(experiment_dir) / "results" / default_filename, ] )
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


def _save_fine_tuning_config(experiment_dir, checkpoint_path, args, unbiased_gamma, goal_reward):
    """Persist the complete fine-tuning configuration beside the experiment."""
    configuration = {"source_biased_policy": str(checkpoint_path), "episodes": args.episodes, "network_type": args.network_type, "unbiased_gamma": unbiased_gamma, "base_goal_reward": goal_reward, "unbiased_reward_scale": args.unbiased_reward_scale, "effective_goal_reward": goal_reward * args.unbiased_reward_scale, "epsilon_start": args.fine_tune_eps_start, "epsilon_min": args.fine_tune_eps_min, "epsilon_decay": args.fine_tune_eps_decay, "learning_rate": args.fine_tune_learning_rate, "batch_size": args.fine_tune_batch_size, "replay_capacity": args.fine_tune_replay_capacity, "polyak": args.polyak, "polyak_tau": args.polyak_tau, "target_update_freq": args.target_update_freq, "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes, "eval_seed": args.eval_seed, "training_seed": args.seed, "num_seeds": args.num_seeds}
    destination = Path(experiment_dir) / "fine_tuning.json"
    destination.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")


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


def _resolve_policy_path(requested_path):
    """Resolve and validate a fine-tuning source checkpoint."""
    requested = Path(requested_path).expanduser()
    candidates = [requested, Path(FRAMEWORK_DIR) / requested] if not requested.is_absolute() else [requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Biased policy checkpoint not found. Checked:\n  - {checked}")


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


def _create_seeded_learner(*, initialization_seed, random_seed, **learner_kwargs):
    """Build a learner without advancing the process-wide PyTorch RNG."""
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(initialization_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(initialization_seed)
        return HierarchicalDQNLearner( random_seed=random_seed, **learner_kwargs, )


def _load_neural_policy(agent, checkpoint_path, learning_rate=None):
    """Load a policy unchanged and reset its target network and optimizer."""
    state_dict = torch.load(checkpoint_path, map_location=agent.device, weights_only=True)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.target_net.eval()
    if learning_rate is not None:
        agent.lr = learning_rate
    agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=agent.lr)
    agent.optimization_steps = 0
    agent.reset_diagnostics()


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
    for key in (
        "task_rewards",
        "learning_rewards",
        "biased_learning_rewards",
        "unbiased_learning_rewards",
        "shaping_rewards",
    ):
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
    initial_truth_assignment = abstract_mdp.get_environment_truth_assignment( observation )
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


def _write_run_header(log_handle, episodes, use_shaping, goal_reward, abstract_mdp, automaton_states, gamma_shaping, eval_interval, eval_episodes, eval_seed, biased_gamma, unbiased_gamma, unbiased_reward_scale, zero_init_unbiased_output, fine_tune_unbiased):
    """Write the configuration and DFA metadata at the beginning of a training run."""
    if not log_handle:
        return
    automaton = abstract_mdp.automaton
    shaping_formula = "gamma_shaping*Phi(next)-Phi(state)"
    header = (
        "\n=== NEW RUN ===\n"
        "ground_learners=biased_behavior(task+shaping), unbiased_output(task_only)\n"
        "replay_buffer=shared_with_biased_and_unbiased_reward_channels\n"
        f"episodes={episodes}, shaping={use_shaping}, goal_reward={goal_reward}\n"
        f"abstract_gamma={abstract_mdp.gamma}, biased_gamma={biased_gamma}, unbiased_gamma={unbiased_gamma}\n"
        f"unbiased_reward_scale={unbiased_reward_scale}\n"
        f"zero_init_unbiased_output={zero_init_unbiased_output}\n"
        f"fine_tune_unbiased={fine_tune_unbiased}\n"
        f"eval_interval={eval_interval}, eval_episodes={eval_episodes}, eval_seed={eval_seed}\n"
        f"heavy_diagnostics_interval={HEAVY_DIAGNOSTICS_INTERVAL}\n"
        f"gamma_shaping={gamma_shaping}, shaping_formula={shaping_formula}\n"
        f"inter_level_shaping={abstract_mdp.upper_level_mdp is not None}, "
        "inter_level_formula=gamma*Phi(next)-Phi(state)\n"
        f"formula={automaton.formula_str}\n"
        f"regions={{{', '.join(f'{name}: {region.as_dict()}' for name, region in abstract_mdp.regions.items())}}}\n"
        f"dfa_states={automaton_states}, pre_trace={automaton.get_initial_q()}, accepting={sorted(automaton.accepting_states)}\n"
    )
    log_handle.write(header)
    log_handle.flush()


def _should_log(episode, episodes, log_interval):
    """Return whether the current episode requires a periodic training report."""
    return episode == 0 or episode + 1 == episodes or (episode + 1) % log_interval == 0


def _is_evaluation_due(episode, episodes, eval_interval):
    """Return whether greedy evaluation is due for the current episode."""
    return episode + 1 == episodes or (episode + 1) % eval_interval == 0


def _is_heavy_diagnostics_due(episode, episodes):
    """Return whether expensive optimizer and policy diagnostics are due."""
    return (
        episode + 1 == episodes
        or (episode + 1) % HEAVY_DIAGNOSTICS_INTERVAL == 0
    )


def _build_training_log( episode, episodes, log_interval, automaton_states, biased_agent, unbiased_agent, histories, cumulative_counters, include_detailed_diagnostics=False, ):
    """Build a report containing recent metrics and cumulative DFA counters."""
    window = min(log_interval, episode + 1)
    recent_slice = slice(-window, None)
    recent_transitions = Counter()
    for transitions in histories["transition_counters"][-window:]:
        recent_transitions.update(transitions)

    recent_state_visits = np.asarray(histories["state_visits"], dtype=np.int64)[:, -window:].sum(axis=1)
    recent_state_entries = np.asarray(histories["state_entries"], dtype=np.int64)[:, -window:].sum(axis=1)
    buffer_details = ", ".join(f"{q}: {biased_agent.memory.q_fraction_onehot(index, len(automaton_states)):.1%}" for index, q in enumerate(automaton_states))
    recent_visits_details = ", ".join(f"{q}: {recent_state_visits[index]}" for index, q in enumerate(automaton_states))
    recent_entries_details = ", ".join(f"{q}: {recent_state_entries[index]}" for index, q in enumerate(automaton_states))
    cumulative_visits_details = ", ".join(f"{q}: {cumulative_counters['state_visits'][q]}" for q in automaton_states)
    cumulative_entries_details = ", ".join(f"{q}: {cumulative_counters['state_entries'][q]}" for q in automaton_states)

    empty_diagnostics = {
        "updates": 0, "stats_updates": 0, "mean_loss": 0.0, "max_loss": 0.0,
        "mean_abs_q": 0.0, "max_abs_q": 0.0,
        "mean_abs_target": 0.0, "max_abs_target": 0.0,
        "mean_gradient_norm": 0.0, "max_gradient_norm": 0.0,
        "positive_sample_fraction": 0.0, "positive_batch_fraction": 0.0,
    }
    biased_diagnostics = (
        biased_agent.consume_diagnostics()
        if hasattr(biased_agent, "consume_diagnostics")
        else empty_diagnostics
    )
    unbiased_diagnostics = (
        unbiased_agent.consume_diagnostics()
        if hasattr(unbiased_agent, "consume_diagnostics")
        else empty_diagnostics
    )

    # Compare both greedy policies on the same recent replay states without
    # consuming either learner's random stream or modifying the replay buffer.
    replay_transitions = getattr(biased_agent.memory, "buffer", [])
    can_compare_policies = all( hasattr(agent, attribute) for agent in (biased_agent, unbiased_agent) for attribute in ("policy_net", "device") )
    comparison_size = (
        min(1024, len(replay_transitions))
        if include_detailed_diagnostics and can_compare_policies
        else 0
    )
    greedy_agreement = float("nan")
    if comparison_size:
        comparison_states = np.asarray( [transition[0] for transition in replay_transitions[-comparison_size:]], dtype=np.float32, )
        state_tensor = torch.as_tensor(comparison_states, device=biased_agent.device)
        with torch.no_grad():
            biased_actions = biased_agent.policy_net(state_tensor).argmax(dim=1).cpu()
            unbiased_actions = unbiased_agent.policy_net( state_tensor.to(unbiased_agent.device) ).argmax(dim=1).cpu()
        greedy_agreement = float((biased_actions == unbiased_actions).float().mean().item())

    def format_learner_diagnostics(name, diagnostics):
        if "mean_abs_td_error" in diagnostics:
            return f"{name} tabular ({diagnostics['updates']} updates): |TD error| mean/max={diagnostics['mean_abs_td_error']:.4g}/{diagnostics['max_abs_td_error']:.4g}, positive updates={diagnostics['positive_sample_fraction']:.3%}\n"
        if not include_detailed_diagnostics:
            return (
                f"{name} optimizer ({diagnostics['updates']} updates): "
                f"positive samples={diagnostics['positive_sample_fraction']:.3%}, "
                f"positive batches={diagnostics['positive_batch_fraction']:.1%}\n"
            )
        return (
            f"{name} optimizer ({diagnostics['updates']} updates, {diagnostics['stats_updates']} sampled): "
            f"loss mean/max={diagnostics['mean_loss']:.4g}/{diagnostics['max_loss']:.4g}, "
            f"|Q| mean/max={diagnostics['mean_abs_q']:.4g}/{diagnostics['max_abs_q']:.4g}, "
            f"|target| mean/max={diagnostics['mean_abs_target']:.4g}/{diagnostics['max_abs_target']:.4g}, "
            f"grad norm mean/max={diagnostics['mean_gradient_norm']:.4g}/{diagnostics['max_gradient_norm']:.4g}, "
            f"positive samples={diagnostics['positive_sample_fraction']:.3%}, "
            f"positive batches={diagnostics['positive_batch_fraction']:.1%}\n"
        )

    agreement_line = f"greedy action agreement      : {greedy_agreement:.1%} on {comparison_size} buffer states\n" if comparison_size else ""
    tabular_snapshot = unbiased_agent.metrics_snapshot() if isinstance(unbiased_agent, TabularQLearner) else None
    tabular_line = f"tabular Q-table             : {tabular_snapshot['table_size']} states, {tabular_snapshot['updated_state_actions']} updated pairs, coverage={tabular_snapshot['state_action_coverage']:.3%}, positive updates={tabular_snapshot['positive_updates']}\n" if tabular_snapshot is not None else ""

    return (
        "\n"
        f"[Episode {episode + 1}/{episodes} | last {window}]\n"
        f"success rate                : {np.mean(histories['successes'][recent_slice]):.1%} (cumulative {np.mean(histories['successes']):.1%})\n"
        f"synthetic task reward       : {np.mean(histories['task_rewards'][recent_slice]):.3f}\n"
        f"shaping reward              : {np.mean(histories['shaping_rewards'][recent_slice]):.3f}\n"
        f"biased learning reward      : {np.mean(histories['biased_learning_rewards'][recent_slice]):.3f}\n"
        f"unbiased learning reward    : {np.mean(histories['unbiased_learning_rewards'][recent_slice]):.3f}\n"
        f"episode length              : {np.mean(histories['episode_lengths'][recent_slice]):.1f}\n"
        f"abstract changes / episode  : {np.mean(histories['abstract_changes'][recent_slice]):.1f}\n"
        f"DFA transitions / episode   : {np.mean(histories['dfa_transitions'][recent_slice]):.2f}\n"
        f"DFA transitions in window   : {_format_counter(recent_transitions)}\n"
        f"behavior epsilon (next ep.)   : {histories['epsilons'][-1]:.5f}\n"
        f"shared dual replay buffer    : {len(biased_agent.memory)} samples [{buffer_details}]\n"
        f"{tabular_line}"
        f"{agreement_line}"
        f"{format_learner_diagnostics('biased  ', biased_diagnostics)}"
        f"{format_learner_diagnostics('unbiased', unbiased_diagnostics)}"
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


def _greedy_action(agent, augmented_state, return_known=False):
    """Select a greedy action through the learner-specific representation."""
    if isinstance(agent, TabularQLearner):
        return agent.greedy_action(augmented_state, return_known=return_known)
    with torch.inference_mode():
        state_tensor = torch.as_tensor(augmented_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        action = agent.policy_net(state_tensor).argmax(dim=1).item()
    return (action, True) if return_known else action


def _monitoring_average(values, episode, log_interval):
    """Return the mean over the active monitoring window."""
    window = min(log_interval, episode + 1)
    return float(np.mean(values[-window:]))


def _evaluate_agent_greedily(agent, abstract_mdp, episodes, goal_reward, seed):
    """Evaluate one learner without exploration, replay writes, or updates."""
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    successes = 0
    task_rewards = []
    episode_lengths = []
    transition_counts = Counter()
    env = gym.make("LunarLander-v3", continuous=False)
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
            raw_state, _ = env.reset(seed=seed + evaluation_episode)
            q = _evaluate_initial_automaton_state(raw_state, abstract_mdp)
            succeeded = automaton.is_goal_reached(q)
            terminated = truncated = False
            steps = 0

            while not (succeeded or terminated or truncated):
                augmented_state = _augment_state(raw_state, q, state_to_index)
                action, known = _greedy_action(agent, augmented_state, return_known=True)
                known_states += int(known)
                evaluated_states += 1

                next_raw_state, _ignored_reward, terminated, truncated, _ = env.step(action)
                truth_assignment = abstract_mdp.get_environment_truth_assignment( next_raw_state )
                previous_q = q
                q = automaton.get_next_q(previous_q, truth_assignment)
                if q not in state_to_index:
                    raise RuntimeError(f"DFA returned unknown evaluation state {q!r}")
                if q != previous_q:
                    transition_counts[(previous_q, q)] += 1
                succeeded = automaton.is_goal_reached(q)
                raw_state = next_raw_state
                steps += 1

            successes += int(succeeded)
            task_rewards.append(float(goal_reward) if succeeded else 0.0)
            episode_lengths.append(steps)
    finally:
        if is_neural and was_training:
            agent.policy_net.train()
        if tabular_rng_state is not None:
            agent.random_rng.setstate(tabular_rng_state)
        env.close()

    return {
        "success_rate": successes / episodes,
        "mean_task_reward": float(np.mean(task_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "transition_counts": transition_counts,
        "known_state_fraction": known_states / evaluated_states if evaluated_states else 1.0,
    }


def _evaluation_score(metrics):
    """Order evaluation results by success, reward, then shorter episodes."""
    return (
        metrics["success_rate"],
        metrics["mean_task_reward"],
        -metrics["mean_episode_length"],
    )


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


def _build_training_results(histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode, abstract_gamma, biased_gamma, unbiased_gamma, gamma_shaping, unbiased_reward_scale, zero_init_unbiased_output):
    """Select and name the numeric histories returned by the training loop."""
    return {
        "task_rewards": histories["task_rewards"],
        "learning_rewards": histories["learning_rewards"],
        "biased_learning_rewards": histories["biased_learning_rewards"],
        "unbiased_learning_rewards": histories["unbiased_learning_rewards"],
        "shaping_rewards": histories["shaping_rewards"],
        "epsilon_history": histories["epsilons"],
        "buffer_histories": buffer_histories,
        "state_visit_histories": histories["state_visits"],
        "state_entry_histories": histories["state_entries"],
        "successes": histories["successes"],
        "initial_acceptances": initial_acceptance_history,
        "episode_lengths": histories["episode_lengths"],
        "abstract_changes": histories["abstract_changes"],
        "dfa_transitions": histories["dfa_transitions"],
        "automaton_states": automaton_states,
        "best_mean_task_reward": best_mean_reward,
        "best_policy_episode": best_policy_episode,
        "abstract_gamma": abstract_gamma,
        "biased_gamma": biased_gamma,
        "unbiased_gamma": unbiased_gamma,
        "gamma_shaping": gamma_shaping,
        "unbiased_reward_scale": unbiased_reward_scale,
        "shared_dual_replay_buffer": True,
        "zero_init_unbiased_output": zero_init_unbiased_output,
        "best_biased_eval_success_rate": max(histories["biased_eval_success_rates"]),
        "best_unbiased_eval_success_rate": max(histories["unbiased_eval_success_rates"]),
        "evaluation_steps": histories["evaluation_steps"],
        "biased_eval_success_rates": histories["biased_eval_success_rates"],
        "biased_eval_task_rewards": histories["biased_eval_task_rewards"],
        "biased_eval_episode_lengths": histories["biased_eval_episode_lengths"],
        "unbiased_eval_success_rates": histories["unbiased_eval_success_rates"],
        "unbiased_eval_task_rewards": histories["unbiased_eval_task_rewards"],
        "unbiased_eval_episode_lengths": histories["unbiased_eval_episode_lengths"],
        "unbiased_eval_known_state_fractions": histories["unbiased_eval_known_state_fractions"],
        "tabular_table_sizes": histories["tabular_table_sizes"],
        "tabular_visited_states": histories["tabular_visited_states"],
        "tabular_updated_state_actions": histories["tabular_updated_state_actions"],
        "tabular_state_action_coverage": histories["tabular_state_action_coverage"],
        "tabular_positive_updates": histories["tabular_positive_updates"],
    }


# ==============================
# Training loop
# ==============================

def run_sequential_training(env, biased_agent, unbiased_agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, gamma_shaping=None, unbiased_reward_scale=1.0, log_file=None, log_interval=100, eval_interval=1000, eval_episodes=50, eval_seed=100000, seed=None, policy_suffix="", fine_tune_unbiased=False, fine_tune_replay_capacity=300000):
    """
    Train paired off-policy DDQN learners with one shared environment stream.

    The biased learner selects every action and learns from task reward plus
    potential-based shaping.  The unbiased learner sees the same transitions
    but learns from task reward alone.  Gym reward is deliberately discarded.
    """
    # Build a stable mapping between DFA states and neural-network features.
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    num_states = len(automaton_states)

    # Fail early if the DFA or training parameters are inconsistent.
    _validate_training_setup( automaton, state_to_index, episodes, log_interval, eval_interval, eval_episodes, )
    unbiased_is_tabular = isinstance(unbiased_agent, TabularQLearner)
    if fine_tune_unbiased and unbiased_is_tabular:
        raise ValueError("Fine-tuning from a biased neural checkpoint requires --unbiased-learner ddqn")
    if not fine_tune_unbiased and not unbiased_is_tabular and biased_agent.batch_size != unbiased_agent.batch_size:
        raise ValueError("biased and unbiased learners must use the same batch size")
    if fine_tune_replay_capacity <= 0:
        raise ValueError("fine_tune_replay_capacity must be greater than zero")
    if gamma_shaping is None:
        gamma_shaping = biased_agent.gamma
    if not 0.0 < gamma_shaping <= 1.0:
        raise ValueError("gamma_shaping must be in the interval (0, 1]")
    if unbiased_reward_scale <= 0.0:
        raise ValueError("unbiased_reward_scale must be greater than zero")
    if not unbiased_is_tabular and biased_agent.device != unbiased_agent.device:
        raise ValueError("biased and unbiased learners must use the same device")
    minibatch_seed = None if seed is None else seed + 4_000_003
    minibatch_rng = random.Random(minibatch_seed)

    # Store both reward views once and materialize each shared minibatch once.
    memory_capacity = fine_tune_replay_capacity if fine_tune_unbiased else min(getattr(biased_agent.memory, "capacity", 300000), getattr(unbiased_agent.memory, "capacity", 300000)) if not unbiased_is_tabular else getattr(biased_agent.memory, "capacity", 300000)
    shared_memory = DualReplayBuffer(memory_capacity, num_states)
    biased_agent.memory = shared_memory
    if not unbiased_is_tabular:
        unbiased_agent.memory = shared_memory
    behavior_agent = unbiased_agent if fine_tune_unbiased else biased_agent

    # Store episode-level metrics for plots and post-processing.
    task_reward_history = []
    learning_reward_history = []
    shaping_reward_history = []
    unbiased_learning_reward_history = []
    epsilon_history = []
    episode_length_history = []
    success_history = []
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
        "biased_learning_rewards": learning_reward_history,
        "unbiased_learning_rewards": unbiased_learning_reward_history,
        "shaping_rewards": shaping_reward_history,
        "epsilons": epsilon_history,
        "episode_lengths": episode_length_history,
        "successes": success_history,
        "abstract_changes": abstract_change_history,
        "dfa_transitions": dfa_transition_history,
        "transition_counters": transition_counter_history,
        "state_visits": state_visit_histories,
        "state_entries": state_entry_histories,
        "evaluation_steps": [],
        "biased_eval_success_rates": [],
        "biased_eval_task_rewards": [],
        "biased_eval_episode_lengths": [],
        "unbiased_eval_success_rates": [],
        "unbiased_eval_task_rewards": [],
        "unbiased_eval_episode_lengths": [],
        "unbiased_eval_known_state_fractions": [],
        "tabular_table_sizes": [],
        "tabular_visited_states": [],
        "tabular_updated_state_actions": [],
        "tabular_state_action_coverage": [],
        "tabular_positive_updates": [],
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
    best_evaluation_scores = {"biased": None, "unbiased": None}

    # Open one append-only log file for the complete run.
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
    zero_init_unbiased_output = bool( getattr(unbiased_agent, "output_layer_zero_initialized", False) )
    _write_run_header(log_handle, episodes, use_shaping, goal_reward, abstract_mdp, automaton_states, gamma_shaping, eval_interval, eval_episodes, eval_seed, biased_agent.gamma, unbiased_agent.gamma, unbiased_reward_scale, zero_init_unbiased_output, fine_tune_unbiased)

    try:
        for episode in range(episodes):
            evaluation_due = _is_evaluation_due(episode, episodes, eval_interval)
            heavy_diagnostics_due = _is_heavy_diagnostics_due(episode, episodes)
            biased_agent.collect_detailed_diagnostics = heavy_diagnostics_due
            if hasattr(unbiased_agent, "collect_detailed_diagnostics"):
                unbiased_agent.collect_detailed_diagnostics = heavy_diagnostics_due

            # Reset the environment and consume s0 before selecting the first action.
            raw_state, _ = env.reset(seed=seed if episode == 0 else None)
            q = _evaluate_initial_automaton_state(raw_state, abstract_mdp)
            if q not in state_to_index:
                raise RuntimeError(f"DFA returned unknown initial state {q!r} after evaluating s0")
            augmented_state = _augment_state(raw_state, q, state_to_index)

            # Reset counters local to the current episode.
            succeeded = automaton.is_goal_reached(q)
            episode_done = succeeded
            episode_steps = 0
            episode_task_reward = float(goal_reward) if succeeded else 0.0
            episode_shaping_reward = 0.0
            episode_unbiased_learning_reward = (
                episode_task_reward * unbiased_reward_scale
            )
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
                behavior_agent.eps = epsilon_history[-1] if epsilon_history else behavior_agent.eps
                action = behavior_agent.select_action(augmented_state)

                # The environment reward is intentionally not part of training.
                next_raw_state, _ignored_env_reward, env_terminated, env_truncated, _ = env.step(action)

                # Map the transition to abstract spatial states.
                x, y = _abstract_position(raw_state, abstract_mdp)
                next_x, next_y = _abstract_position(next_raw_state, abstract_mdp)
                abstract_state = (x, y, q)

                # Advance the DFA using propositions true in the arrival state.
                truth_assignment = abstract_mdp.get_environment_truth_assignment( next_raw_state )
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

                # Stop data collection on any Gym ending or DFA success.
                # A truncation (for example Gym's time limit) ends data
                # collection, but it is not an MDP terminal state: DDQN must
                # still bootstrap from its final observation.
                episode_done = env_terminated or env_truncated or succeeded
                bootstrap_terminal = env_terminated or succeeded
                next_augmented_state = _augment_state(next_raw_state, next_q, state_to_index)

                # Evaluate the potential difference on every environment step.
                # With gamma_shaping=1, unchanged abstract states yield zero,
                # reproducing the previous cell-change heuristic exactly.
                shaping_signal = 0.0
                if use_shaping:
                    phi_state = abstract_mdp.v_star.get(abstract_state, 0.0)
                    phi_next_state = abstract_mdp.v_star.get(abstract_next_state, 0.0)
                    shaping_signal = gamma_shaping * phi_next_state - phi_state

                # The behavior learner is biased by shaping.  The off-policy
                # output learner receives the same sample with task reward only.
                biased_learning_reward = synthetic_goal_reward + shaping_signal
                unbiased_learning_reward = (
                    synthetic_goal_reward * unbiased_reward_scale
                )
                shared_memory.push(augmented_state, action, biased_learning_reward, unbiased_learning_reward, next_augmented_state, bootstrap_terminal)
                if unbiased_is_tabular:
                    unbiased_agent.update(augmented_state, action, unbiased_learning_reward, next_augmented_state, bootstrap_terminal)
                if len(shared_memory) >= behavior_agent.batch_size:
                    batch_indices = shared_memory.sample_indices(behavior_agent.batch_size, minibatch_rng)
                    if not fine_tune_unbiased:
                        biased_agent.optimize_model(batch_indices, reward_channel="biased")
                    if not unbiased_is_tabular:
                        unbiased_agent.optimize_model(batch_indices, reward_channel="unbiased")

                # Update the episode totals and move to the next state.
                episode_steps += 1
                episode_task_reward += synthetic_goal_reward
                episode_shaping_reward += shaping_signal
                episode_unbiased_learning_reward += unbiased_learning_reward
                raw_state = next_raw_state
                augmented_state = next_augmented_state
                q = next_q

                # Count Gym endings for diagnostics without using its reward.
                if env_terminated:
                    cumulative_env_terminated += 1
                if env_truncated:
                    cumulative_env_truncated += 1

            # Decay the single epsilon once at the end of the episode.
            next_epsilon = max(behavior_agent.eps_min, behavior_agent.eps * behavior_agent.eps_decay)
            behavior_agent.eps = next_epsilon

            # Save the metrics collected for this episode.
            episode_learning_reward = episode_task_reward + episode_shaping_reward
            task_reward_history.append(episode_task_reward)
            shaping_reward_history.append(episode_shaping_reward)
            unbiased_learning_reward_history.append( episode_unbiased_learning_reward )
            learning_reward_history.append(episode_learning_reward)
            epsilon_history.append(next_epsilon)
            episode_length_history.append(episode_steps)
            success_history.append(int(succeeded))
            initial_acceptance_history.append(int(episode_steps == 0 and succeeded))
            abstract_change_history.append(episode_abstract_changes)
            dfa_transition_history.append(episode_dfa_transitions)
            transition_counter_history.append(episode_transitions)
            tabular_metrics = unbiased_agent.metrics_snapshot() if unbiased_is_tabular else {"table_size": np.nan, "visited_states": np.nan, "updated_state_actions": np.nan, "state_action_coverage": np.nan, "positive_updates": np.nan}
            histories["tabular_table_sizes"].append(tabular_metrics["table_size"])
            histories["tabular_visited_states"].append(tabular_metrics["visited_states"])
            histories["tabular_updated_state_actions"].append(tabular_metrics["updated_state_actions"])
            histories["tabular_state_action_coverage"].append(tabular_metrics["state_action_coverage"])
            histories["tabular_positive_updates"].append(tabular_metrics["positive_updates"])

            # Record replay-buffer composition, state visits, and entries from other states.
            for index in range(num_states):
                buffer_histories[index].append(biased_agent.memory.q_fraction_onehot(index, num_states))
                state_visit_histories[index].append(episode_state_visits[index])
                state_entry_histories[index].append(episode_state_entries[index])

            # Print recent and cumulative diagnostics at the requested interval.
            if (
                _should_log(episode, episodes, log_interval)
                or evaluation_due
                or heavy_diagnostics_due
            ):
                cumulative_counters = {"state_visits": cumulative_state_visits, "state_entries": cumulative_state_entries, "transitions": cumulative_transitions, "initial_acceptances": cumulative_initial_acceptances, "env_terminated": cumulative_env_terminated, "env_truncated": cumulative_env_truncated}
                _write_log( _build_training_log( episode, episodes, log_interval, automaton_states, biased_agent, unbiased_agent, histories, cumulative_counters, include_detailed_diagnostics=heavy_diagnostics_due, ), log_handle, )

            # Evaluate both policies greedily on identical held-out episodes.
            # The final training episode is always evaluated even when it is
            # not an exact multiple of eval_interval.
            if evaluation_due:
                _write_log( f"\nStarting autonomous greedy evaluation at episode {episode + 1} " f"({eval_episodes} episodes per learner)...\n", log_handle, )
                biased_evaluation = _evaluate_agent_greedily( biased_agent, abstract_mdp, eval_episodes, goal_reward, eval_seed, )
                _write_log("Biased learner evaluation completed; evaluating unbiased learner...\n", log_handle)
                evaluation_results = {
                    "biased": biased_evaluation,
                    "unbiased": _evaluate_agent_greedily( unbiased_agent, abstract_mdp, eval_episodes, goal_reward, eval_seed, ),
                }
                histories["evaluation_steps"].append(episode + 1)
                for learner_name, result in evaluation_results.items():
                    histories[f"{learner_name}_eval_success_rates"].append(result["success_rate"])
                    histories[f"{learner_name}_eval_task_rewards"].append(result["mean_task_reward"])
                    histories[f"{learner_name}_eval_episode_lengths"].append(result["mean_episode_length"])
                histories["unbiased_eval_known_state_fractions"].append(evaluation_results["unbiased"]["known_state_fraction"])

                # Keep recoverable snapshots of the networks evaluated at this
                # checkpoint.  Unlike the best policies, these are overwritten
                # at every evaluation and therefore always represent the most
                # recently evaluated learners.
                if save_policy:
                    _save_named_policy(biased_agent, f"last_policy_biased{policy_suffix}.{_policy_extension(biased_agent)}")
                    _save_named_policy(unbiased_agent, f"last_policy_unbiased{policy_suffix}.{_policy_extension(unbiased_agent)}")
                    _write_log( f"Last learner snapshots updated at evaluation episode {episode + 1}.\n", log_handle, )

                _write_log( "\n" f"[Greedy evaluation at episode {episode + 1} | {eval_episodes} fixed-seed episodes]\n" f"biased   : success={evaluation_results['biased']['success_rate']:.1%}, " f"task reward={evaluation_results['biased']['mean_task_reward']:.3f}, " f"length={evaluation_results['biased']['mean_episode_length']:.1f}\n" f"biased DFA transitions   : {_format_counter(evaluation_results['biased'].get('transition_counts', Counter()))}\n" f"unbiased : success={evaluation_results['unbiased']['success_rate']:.1%}, " f"task reward={evaluation_results['unbiased']['mean_task_reward']:.3f}, " f"length={evaluation_results['unbiased']['mean_episode_length']:.1f}, " f"known states={evaluation_results['unbiased']['known_state_fraction']:.1%}\n" f"unbiased DFA transitions : {_format_counter(evaluation_results['unbiased'].get('transition_counts', Counter()))}\n", log_handle, )

                # Each best checkpoint is selected from that learner's own
                # autonomous evaluation, never from behavior trajectories.
                for learner_name, learner in (
                    ("biased", biased_agent),
                    ("unbiased", unbiased_agent),
                ):
                    score = _evaluation_score(evaluation_results[learner_name])
                    is_new_best = (
                        best_evaluation_scores[learner_name] is None
                        or score > best_evaluation_scores[learner_name]
                    )
                    if is_new_best:
                        best_evaluation_scores[learner_name] = score
                        if save_policy:
                            _save_named_policy(learner, f"best_policy_{learner_name}{policy_suffix}.{_policy_extension(learner)}")
                        _write_log( f"Best {learner_name} policy updated from autonomous evaluation at episode {episode + 1}.\n", log_handle, )
                        if learner_name == "unbiased":
                            best_mean_reward = evaluation_results[learner_name]["mean_task_reward"]
                            best_policy_episode = episode + 1

        # Save the final policy independently from its monitored performance.
        if save_policy:
            _save_named_policy(unbiased_agent, f"last_policy_unbiased{policy_suffix}.{_policy_extension(unbiased_agent)}")
            _save_named_policy(biased_agent, f"last_policy_biased{policy_suffix}.{_policy_extension(biased_agent)}")
            _write_log(f"Last learner snapshots saved after episode {episodes}. Algorithm output: unbiased policy. Best unbiased evaluation: episode {best_policy_episode}, mean task reward={best_mean_reward:.3f}\n", log_handle)
    finally:
        # Always close the log, including when training raises an exception.
        if log_handle:
            log_handle.close()

    # Return named histories to avoid ambiguous tuple positions.
    return _build_training_results( histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode, abstract_mdp.gamma, biased_agent.gamma, unbiased_agent.gamma, gamma_shaping, unbiased_reward_scale, zero_init_unbiased_output, )


# ==============================
# Experiment setup and outputs
# ==============================

def main(args):
    """Configure the experiment, run or load training, and generate diagnostic plots."""
    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be greater than zero")
    fine_tune_mode = args.fine_tune_from_biased is not None
    if fine_tune_mode and args.unbiased_learner != "ddqn":
        raise ValueError("--fine-tune-from-biased requires --unbiased-learner ddqn")
    if fine_tune_mode and args.zero_init_unbiased_output:
        raise ValueError("--zero-init-unbiased-output cannot be combined with --fine-tune-from-biased")
    if args.fine_tune_eps_min > args.fine_tune_eps_start:
        raise ValueError("--fine-tune-eps-min cannot exceed --fine-tune-eps-start")
    fine_tune_checkpoint = _resolve_policy_path(args.fine_tune_from_biased) if fine_tune_mode and not args.post_process else None
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
    regions = load_regions(config.get("regions"))
    gamma = float(config.get("gamma", 0.99))
    biased_gamma = gamma if args.biased_gamma is None else args.biased_gamma
    unbiased_gamma = gamma if args.unbiased_gamma is None else args.unbiased_gamma
    goal_reward = float(config.get("goal_reward", 10000))
    primary_level = abstraction_config.primary
    if fine_tune_mode and not args.post_process:
        _save_fine_tuning_config(experiment_dir, fine_tune_checkpoint, args, unbiased_gamma, goal_reward)

    # Build the DFA once for both training and post-processing.
    automaton = LTLfAutomaton(formula)
    validation_report = validate_automaton( automaton, regions, )
    level_summary = ", ".join( f"{index}:{level.name}={level.width}x{level.height}" for index, level in enumerate(abstraction_config.levels, start=1) )
    fine_tune_summary = f"Fine-tuning source: {fine_tune_checkpoint}\nFine-tuning transferred Q values: unchanged\nFine-tuning epsilon: start={args.fine_tune_eps_start}, min={args.fine_tune_eps_min}, decay={args.fine_tune_eps_decay}\nFine-tuning learning rate/batch/replay: {args.fine_tune_learning_rate}/{args.fine_tune_batch_size}/{args.fine_tune_replay_capacity}\n" if fine_tune_mode else "Fine-tuning: disabled\n"
    print( "=== LTLf TRAINING (dual ground learners) ===\n" f"Formula: {formula}\n" f"Regions: { {name: region.as_dict() for name, region in regions.items()} }\n" f"Abstractions: {level_summary}\n" "Ground DDQN return: one-step\n" f"Unbiased learner: {args.unbiased_learner}\n" f"Discount factors: abstract={gamma}, biased={biased_gamma}, " f"unbiased={unbiased_gamma}\n" f"Unbiased reward scale: {args.unbiased_reward_scale}\n" f"Zero-init unbiased output: {args.zero_init_unbiased_output}\n" f"{fine_tune_summary}" "Automaton coordinates and training potential: level1\n" f"DFA: states={automaton.states}, pre-trace={automaton.initial_state}, " f"accepting={sorted(automaton.accepting_states)}\n" "Gym reward is ignored by design.\n" f"{validation_report.format()}" )

    if not args.post_process:
        automaton.render_graph(directory=image_dir)

    # Heatmaps depend only on the saved task configuration, not on agent training.
    multilevel_mdp = MultiLevelWaypointMDP( regions=regions, ltlf_automaton=automaton, abstraction_config=abstraction_config, gamma=gamma, goal_reward=goal_reward, )
    multilevel_mdp.compute_value_functions()
    save_multilevel_heatmaps( multilevel_mdp, filename_prefix="single_epsilon_exp", output_root=os.path.join(image_dir, "heatmaps"), )
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
                biased_initialization_seed = run_seed
                unbiased_initialization_seed = run_seed
                biased_exploration_seed = run_seed + 2_000_003
                print( "Learner seeds: " f"biased_init={biased_initialization_seed}, " f"unbiased_init={unbiased_initialization_seed}, " f"biased_exploration={biased_exploration_seed}, " f"shared_minibatches={run_seed + 4_000_003}" )
                biased_agent = _create_seeded_learner( initialization_seed=biased_initialization_seed, random_seed=biased_exploration_seed, env=env, max_episodes=args.episodes, eps_decay=args.eps_decay, gamma=biased_gamma, extra_state_dims=len(automaton.states), use_polyak=args.polyak, tau=args.polyak_tau, target_update_freq=args.target_update_freq, network_type=args.network_type, policy_dir=policy_dir, )
                if args.unbiased_learner == "tabular":
                    unbiased_agent = TabularQLearner(env=env, num_phases=len(automaton.states), gamma=unbiased_gamma, alpha=args.tabular_alpha, policy_dir=policy_dir, random_seed=run_seed + 3_000_003)
                else:
                    unbiased_eps_decay = args.fine_tune_eps_decay if fine_tune_mode else args.eps_decay
                    unbiased_agent = _create_seeded_learner(initialization_seed=unbiased_initialization_seed, random_seed=run_seed + 3_000_003, env=env, max_episodes=args.episodes, eps_decay=unbiased_eps_decay, gamma=unbiased_gamma, extra_state_dims=len(automaton.states), use_polyak=args.polyak, tau=args.polyak_tau, target_update_freq=args.target_update_freq, network_type=args.network_type, policy_dir=policy_dir)
                if args.zero_init_unbiased_output and args.unbiased_learner == "tabular":
                    raise ValueError("--zero-init-unbiased-output is only valid with --unbiased-learner ddqn")
                if args.zero_init_unbiased_output:
                    unbiased_agent.zero_initialize_output_layer()
                if fine_tune_mode:
                    _load_neural_policy(biased_agent, fine_tune_checkpoint)
                    _load_neural_policy(unbiased_agent, fine_tune_checkpoint, learning_rate=args.fine_tune_learning_rate)
                    unbiased_agent.eps = args.fine_tune_eps_start
                    unbiased_agent.eps_min = args.fine_tune_eps_min
                    unbiased_agent.batch_size = args.fine_tune_batch_size
                policy_suffix = "" if args.num_seeds == 1 else f"_seed_{run_seed}"
                metrics = run_sequential_training(env=env, biased_agent=biased_agent, unbiased_agent=unbiased_agent, abstract_mdp=abstract_mdp, episodes=args.episodes, goal_reward=goal_reward, use_shaping=False if fine_tune_mode else not args.no_shaping, gamma_shaping=(biased_gamma if args.gamma_shaping is None else args.gamma_shaping), unbiased_reward_scale=args.unbiased_reward_scale, log_file=f"{log_dir}/dual_learner_training_seed_{run_seed}.log", log_interval=args.log_interval, eval_interval=args.eval_interval, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed, seed=run_seed, policy_suffix=policy_suffix, fine_tune_unbiased=fine_tune_mode, fine_tune_replay_capacity=args.fine_tune_replay_capacity)
                seed_metrics.append(metrics)
                save_training_data(f"{data_dir}/dual_learner_data_seed_{run_seed}.npz", **metrics)
            finally:
                env.close()
        save_training_data(f"{data_dir}/dual_learner_data.npz", **_aggregate_seed_metrics(seed_metrics, seeds))

    # Load saved metrics and generate the final diagnostic plots.
    data_path = (
        _resolve_metrics_path(experiment_dir, "dual_learner_data.npz")
        if args.post_process
        else Path(data_dir) / "dual_learner_data.npz"
    )
    print(f"Training data: {data_path}")
    data = np.load(data_path, allow_pickle=False)
    task_reward_runs = data["task_rewards_runs"] if "task_rewards_runs" in data else data["task_rewards"][np.newaxis, :]
    learning_reward_runs = data["learning_rewards_runs"] if "learning_rewards_runs" in data else data["learning_rewards"][np.newaxis, :]
    epsilon_runs = data["epsilon_history_runs"] if "epsilon_history_runs" in data else data["epsilon_history"][np.newaxis, ...]
    buffer_runs = data["buffer_histories_runs"] if "buffer_histories_runs" in data else data["buffer_histories"][np.newaxis, ...]
    seed_values = data["seeds"] if "seeds" in data else np.asarray([args.seed])
    for obsolete_name in ("buffer_fractions_dual_learner.png", "reward_breakdown_dual_learner.png"):
        (Path(plot_dir) / obsolete_name).unlink(missing_ok=True)
    for run_index, (run_seed, task_rewards, learning_rewards, epsilon_history) in enumerate(zip(seed_values, task_reward_runs, learning_reward_runs, epsilon_runs)):
        seed_plot_dir = os.path.join(plot_dir, f"seed_{int(run_seed)}")
        os.makedirs(seed_plot_dir, exist_ok=True)
        plot_shaping_reward_breakdown(task_rewards, learning_rewards, epsilon_history, window_size=args.plot_window, filename=f"{seed_plot_dir}/reward_breakdown_dual_learner.png", title=f"Biased vs Unbiased Reward — Seed {int(run_seed)}")
        if run_index < len(buffer_runs):
            plot_buffer_fractions(buffer_runs[run_index], filename=f"{seed_plot_dir}/buffer_fractions_dual_learner.png", window_size=args.plot_window, state_labels=data["automaton_states"], title=f"Shared Experience Composition — Seed {int(run_seed)}")
    plot_training_variance( learning_reward_runs, window_size=args.plot_window, filename=f"{plot_dir}/training_variance_dual_learner.png", epsilon_histories=epsilon_runs, )
    plot_buffer_variance(buffer_runs, window_size=args.plot_window, filename=f"{plot_dir}/buffer_variance_dual_learner.png", state_labels=data["automaton_states"])
    evaluation_steps = data["evaluation_steps_runs"] if "evaluation_steps_runs" in data else data["evaluation_steps"][np.newaxis, ...]
    biased_eval_success = data["biased_eval_success_rates_runs"] if "biased_eval_success_rates_runs" in data else data["biased_eval_success_rates"][np.newaxis, ...]
    unbiased_eval_success = data["unbiased_eval_success_rates_runs"] if "unbiased_eval_success_rates_runs" in data else data["unbiased_eval_success_rates"][np.newaxis, ...]
    plot_dual_learner_evaluation(evaluation_steps, biased_eval_success, unbiased_eval_success, filename=f"{plot_dir}/dual_learner_evaluation.png")
    tabular_table_runs = data["tabular_table_sizes_runs"] if "tabular_table_sizes_runs" in data else data["tabular_table_sizes"][np.newaxis, ...] if "tabular_table_sizes" in data else None
    if tabular_table_runs is not None and np.isfinite(tabular_table_runs).any():
        tabular_pair_runs = data["tabular_updated_state_actions_runs"] if "tabular_updated_state_actions_runs" in data else data["tabular_updated_state_actions"][np.newaxis, ...]
        tabular_coverage_runs = data["tabular_state_action_coverage_runs"] if "tabular_state_action_coverage_runs" in data else data["tabular_state_action_coverage"][np.newaxis, ...]
        known_state_runs = data["unbiased_eval_known_state_fractions_runs"] if "unbiased_eval_known_state_fractions_runs" in data else data["unbiased_eval_known_state_fractions"][np.newaxis, ...]
        plot_tabular_learning_diagnostics(tabular_table_runs, tabular_pair_runs, tabular_coverage_runs, evaluation_steps, known_state_runs, filename=f"{plot_dir}/tabular_learning_diagnostics.png")
    print("\nFinished.")


# ==============================
# Command-line entry point
# ==============================

if __name__ == "__main__":
    # Expose the main training and post-processing options.
    parser = argparse.ArgumentParser(description="LTLf DDQN training with biased behavior and unbiased output learners.")
    parser.add_argument("--experiment-name", type=_experiment_name, required=True, help="Output directory name under results/.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-seeds", type=_positive_int, default=1, help="Number of training runs with consecutive seeds.")
    parser.add_argument("--seed", type=int, default=42, help="First training seed.")
    parser.add_argument("--config", default="trajectory.json")
    parser.add_argument( "--abstraction-config", default="abstraction.json", help="Ordered grid hierarchy (level1 defines automaton coordinates).", )
    parser.add_argument("--eps-decay", type=float, default=0.9996)
    parser.add_argument( "--biased-gamma", type=_discount_factor, default=None, help="Discount factor for the biased DDQN (default: trajectory.json gamma).", )
    parser.add_argument( "--unbiased-gamma", type=_discount_factor, default=None, help="Discount factor for the unbiased DDQN (default: trajectory.json gamma).", )
    parser.add_argument( "--gamma-shaping", type=_discount_factor, default=None, help="Discount used in Phi shaping (default: biased gamma; use 1 for the cell-change-equivalent heuristic).", )
    parser.add_argument( "--zero-init-unbiased-output", action="store_true", help="Initialize only the unbiased policy/target output layers to zero (default: disabled).", )
    parser.add_argument( "--unbiased-reward-scale", type=_positive_float, default=1.0, help="Multiply only the unbiased learner reward by this factor (default: 1.0).", )
    parser.add_argument("--unbiased-learner", choices=["ddqn", "tabular"], default="ddqn", help="Output learner: neural DDQN or sparse tabular Q-learning.")
    parser.add_argument("--tabular-alpha", type=_learning_rate, default=0.1, help="Learning rate used by the tabular unbiased learner.")
    parser.add_argument("--fine-tune-from-biased", type=Path, default=None, help="Start an unbiased-only DDQN fine-tuning run from this biased policy checkpoint.")
    parser.add_argument("--fine-tune-eps-start", type=_probability, default=0.1, help="Initial epsilon used by the transferred policy during tuning.")
    parser.add_argument("--fine-tune-eps-min", type=_probability, default=0.01, help="Minimum epsilon used during tuning.")
    parser.add_argument("--fine-tune-eps-decay", type=_discount_factor, default=0.9996, help="Per-episode epsilon decay used during tuning.")
    parser.add_argument("--fine-tune-learning-rate", type=_positive_float, default=1e-3, help="Learning rate of the fresh tuning optimizer.")
    parser.add_argument("--fine-tune-batch-size", type=_positive_int, default=64, help="Replay minibatch size used during tuning.")
    parser.add_argument("--fine-tune-replay-capacity", type=_positive_int, default=300000, help="Capacity of the fresh replay buffer used during tuning.")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument( "--eval-interval", type=_positive_int, default=1000, help="Run autonomous greedy evaluation every N training episodes.", )
    parser.add_argument( "--eval-episodes", type=_positive_int, default=50, help="Number of fixed-seed episodes per learner and evaluation point.", )
    parser.add_argument( "--eval-seed", type=int, default=100000, help="First held-out seed reused at every evaluation point.", )
    parser.add_argument("--plot-window", type=int, default=500)
    parser.add_argument( "--polyak", action=argparse.BooleanOptionalAction, default=True, help="Use Polyak target updates (disable with --no-polyak).", )
    parser.add_argument("--polyak-tau", type=float, default=0.005)
    parser.add_argument( "--target-update-freq", type=int, default=1000, help="Hard target-network update interval used with --no-polyak.", )
    parser.add_argument( "--network-type", choices=["standard", "dueling"], default="standard", help="Q-network architecture: standard MLP or dueling value/advantage streams.", )
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--post-process", action="store_true")
    main(parser.parse_args())
