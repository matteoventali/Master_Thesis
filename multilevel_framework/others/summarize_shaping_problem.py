"""Build the summary table for the shaping temporal-alignment experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


EXPERIMENTS = {
    "0.99": "gamma_ablation_traj2_12x12_g099",
    "1.00": "gamma_ablation_traj2_12x12_g100",
}


def _as_runs(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    key = f"{name}_runs"
    if key not in data:
        raise KeyError(f"{key} is missing from the aggregate archive")
    values = np.asarray(data[key], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return values


def _best_success_rates(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    best_episodes = _as_runs(data, "best_policy_episode").reshape(-1)
    evaluation_steps = _as_runs(data, "evaluation_steps")
    success_rates = _as_runs(data, "eval_success_rates")

    selected = np.empty(best_episodes.size, dtype=np.float64)
    for seed_index, episode in enumerate(best_episodes):
        matches = np.flatnonzero(np.isclose(evaluation_steps[seed_index], episode))
        if matches.size != 1:
            raise ValueError(
                f"seed {seed_index}: best episode {episode:g} does not uniquely match "
                "the periodic-evaluation checkpoints"
            )
        selected[seed_index] = success_rates[seed_index, matches[0]]
    return selected, best_episodes


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, label: str) -> np.ndarray:
    denominator_sum = denominator.sum(axis=1)
    if np.any(denominator_sum == 0):
        raise ValueError(f"cannot compute {label}: a seed has a zero denominator")
    return numerator.sum(axis=1) / denominator_sum


def _summarize(archive: Path) -> dict[str, np.ndarray]:
    with np.load(archive, allow_pickle=False) as data:
        best_success, best_episode = _best_success_rates(data)
        same_steps = _as_runs(data, "same_abstract_state_steps")
        episode_lengths = _as_runs(data, "episode_lengths")
        same_rewards = _as_runs(data, "same_state_shaping_rewards")

    return {
        "best_success_rate": best_success,
        "best_policy_episode": best_episode,
        "same_state_fraction": _safe_ratio(
            same_steps, episode_lengths, "same-state transition fraction"
        ),
        "same_state_reward_per_transition": _safe_ratio(
            same_rewards, same_steps, "same-state reward per transition"
        ),
        "same_state_reward_per_episode": same_rewards.mean(axis=1),
    }


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _formatted(mean: float, std: float, *, percentage: bool = False) -> str:
    if percentage:
        return f"{100 * mean:.2f} ± {100 * std:.2f}%"
    return f"{mean:.3f} ± {std:.3f}"


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    default_root = repository / "experiments" / "final" / "rq1" / "shaping_problem"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.root / "comparison" / "shaping_problem_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for gamma, directory in EXPERIMENTS.items():
        archive = args.root / directory / "results" / "single_epsilon_data.npz"
        summary = _summarize(archive)
        statistics = {name: _mean_std(values) for name, values in summary.items()}
        row = {
            "gamma_shaping": gamma,
            "best_policy_success_rate": _formatted(
                *statistics["best_success_rate"], percentage=True
            ),
            "best_policy_episode": _formatted(*statistics["best_policy_episode"]),
            "same_state_transitions": _formatted(
                *statistics["same_state_fraction"], percentage=True
            ),
            "shaping_per_same_state_transition": _formatted(
                *statistics["same_state_reward_per_transition"]
            ),
            "same_state_shaping_per_episode": _formatted(
                *statistics["same_state_reward_per_episode"]
            ),
        }
        rows.append(row)

        print(f"gamma_shaping={gamma}")
        for name, values in summary.items():
            mean, std = statistics[name]
            percentage = name in {"best_success_rate", "same_state_fraction"}
            print(f"  {name}: {_formatted(mean, std, percentage=percentage)}")
            if name == "best_policy_episode":
                print(f"    per seed: {', '.join(f'{value:g}' for value in values)}")

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
