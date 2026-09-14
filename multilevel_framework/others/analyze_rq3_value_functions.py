#!/usr/bin/env python3
"""RQ3 block-2 analysis: extract and compare V-functions from learned Q-tables.

For every learned unbiased Q-table, this script derives both max_a Q(s,a) and
the exact (up to ``theta``) value of its deterministic greedy policy.  The two
estimates are compared with a value-iteration solution of the same abstract
MDP and aggregated over seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = FRAMEWORK_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from abstract_mdps import LTLfWaypointMDP, build_task_automaton
from spatial_regions import load_task_propositions


METRICS = (
    "mae",
    "normalized_mae",
    "rmse",
    "normalized_rmse",
    "pearson",
    "spearman",
    "policy_agreement",
    "optimality_residual_mean",
    "optimality_residual_max",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="360x360 VI value_function.npz")
    parser.add_argument("--candidates-glob", required=True, help="Glob selecting learned level1 value_function.npz files")
    parser.add_argument("--trajectory", type=Path, required=True, help="trajectory.json used by VI and Q-learning")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.001, help="Policy-evaluation convergence tolerance")
    args = parser.parse_args()
    if args.theta <= 0:
        parser.error("--theta must be greater than zero")
    return args


def scalar(archive, key):
    if key not in archive.files:
        raise ValueError(f"Missing NPZ field {key!r}")
    return np.asarray(archive[key]).item()


def load_reference(path):
    with np.load(path, allow_pickle=False) as data:
        key = "unbiased_values" if "unbiased_values" in data.files else "values"
        required = {key, "dfa_states", "width", "height", "gamma", "goal_reward"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"VI reference is missing fields: {missing}")
        result = {
            "values": np.asarray(data[key], dtype=np.float64).copy(),
            "dfa_states": np.asarray(data["dfa_states"], dtype=np.int64).copy(),
            "accepting_states": np.asarray(data.get("accepting_dfa_states", []), dtype=np.int64).copy(),
            "width": int(scalar(data, "width")),
            "height": int(scalar(data, "height")),
            "gamma": float(scalar(data, "gamma")),
            "goal_reward": float(scalar(data, "goal_reward")),
            "task_type": str(scalar(data, "task_type")) if "task_type" in data.files else None,
        }
    expected = (len(result["dfa_states"]), result["height"], result["width"])
    if result["values"].shape != expected or not np.isfinite(result["values"]).all():
        raise ValueError(f"Invalid VI array: got {result['values'].shape}, expected {expected}")
    return result


def seed_from_path(path, fallback):
    matches = re.findall(r"seed[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(matches[-1]) if matches else fallback


def load_candidate(path, reference):
    with np.load(path, allow_pickle=False) as data:
        if "q_function_unbiased" not in data.files:
            raise ValueError(f"Candidate has no q_function_unbiased: {path}")
        q_values = np.asarray(data["q_function_unbiased"], dtype=np.float64).copy()
        dfa_states = np.asarray(data["dfa_states"], dtype=np.int64)
        metadata = {
            "width": int(scalar(data, "width")),
            "height": int(scalar(data, "height")),
            "gamma": float(scalar(data, "gamma")),
            "goal_reward": float(scalar(data, "goal_reward")),
            "task_type": str(scalar(data, "task_type")) if "task_type" in data.files else None,
        }
    for key in ("width", "height"):
        if metadata[key] != reference[key]:
            raise ValueError(f"{path}: {key}={metadata[key]} does not match VI {reference[key]}")
    for key in ("gamma", "goal_reward"):
        if not math.isclose(metadata[key], reference[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{path}: {key}={metadata[key]} does not match VI {reference[key]}")
    if reference["task_type"] is not None and metadata["task_type"] is not None and metadata["task_type"] != reference["task_type"]:
        raise ValueError(f"{path}: task type does not match VI")
    candidate_index = {int(q): index for index, q in enumerate(dfa_states)}
    reference_states = [int(q) for q in reference["dfa_states"]]
    if set(candidate_index) != set(reference_states):
        raise ValueError(f"{path}: DFA states do not match VI")
    q_values = q_values[[candidate_index[q] for q in reference_states]]
    expected = reference["values"].shape + (q_values.shape[-1],)
    if q_values.shape != expected or not np.isfinite(q_values).all():
        raise ValueError(f"{path}: invalid Q array {q_values.shape}; expected {expected}")
    return q_values


def build_dense_model(mdp, dfa_states):
    """Build transition tensors indexed as [q,y,x,a] once for all seeds."""
    q_index = {int(q): i for i, q in enumerate(dfa_states)}
    shape = (len(dfa_states), mdp.height, mdp.width, len(mdp.actions))
    next_index = np.zeros(shape, dtype=np.int64)
    rewards = np.zeros(shape, dtype=np.float64)
    terminal = np.zeros(shape, dtype=bool)
    valid = np.zeros(shape, dtype=bool)
    action_index = {action: i for i, action in enumerate(mdp.actions)}
    for x, y, q in mdp.states:
        qi = q_index[int(q)]
        for action in mdp.get_available_actions((x, y, q)):
            ai = action_index[action]
            next_state, reward, done, _ = mdp.get_transition_outcome((x, y, q), action)
            nx, ny, nq = next_state
            next_index[qi, y, x, ai] = np.ravel_multi_index((q_index[int(nq)], ny, nx), mdp_shape(mdp, dfa_states))
            rewards[qi, y, x, ai] = reward
            terminal[qi, y, x, ai] = done
            valid[qi, y, x, ai] = True
    return next_index, rewards, terminal, valid


def mdp_shape(mdp, dfa_states):
    return len(dfa_states), mdp.height, mdp.width


def greedy_from_q(q_values, valid):
    masked = np.where(valid, q_values, -np.inf)
    return np.max(masked, axis=-1), np.argmax(masked, axis=-1)


def selected_transition(policy, next_index, rewards, terminal):
    selector = policy[..., None]
    return (
        np.take_along_axis(next_index, selector, axis=-1)[..., 0],
        np.take_along_axis(rewards, selector, axis=-1)[..., 0],
        np.take_along_axis(terminal, selector, axis=-1)[..., 0],
    )


def policy_evaluation(policy, next_index, rewards, terminal, gamma, theta):
    selected_next, selected_reward, selected_terminal = selected_transition(policy, next_index, rewards, terminal)
    values = np.zeros(policy.shape, dtype=np.float64)
    iterations = 0
    while True:
        iterations += 1
        updated = selected_reward + (~selected_terminal) * gamma * values.ravel()[selected_next]
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta < theta:
            return values, iterations


def optimal_targets(values, next_index, rewards, terminal, valid, gamma):
    targets = rewards + (~terminal) * gamma * values.ravel()[next_index]
    return np.where(valid, targets, -np.inf)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metric_row(seed, method, scope, reference_values, candidate_values, mask, goal_reward, policy_agreement, residual):
    ref = reference_values[mask]
    cand = candidate_values[mask]
    difference = cand - ref
    absolute = np.abs(difference)
    scale = abs(goal_reward) or 1.0
    return {
        "seed": seed,
        "method": method,
        "scope": scope,
        "cells": int(ref.size),
        "mae": float(np.mean(absolute)),
        "normalized_mae": float(np.mean(absolute) / scale),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "normalized_rmse": float(np.sqrt(np.mean(difference * difference)) / scale),
        "pearson": correlation(ref, cand),
        "spearman": correlation(rankdata(ref), rankdata(cand)),
        "policy_agreement": float(np.mean(policy_agreement[mask])),
        "optimality_residual_mean": float(np.mean(residual[mask])),
        "optimality_residual_max": float(np.max(residual[mask])),
    }


def analyze_seed(seed, q_values, reference, model, theta):
    next_index, rewards, terminal, valid = model
    maxq_values, learned_policy = greedy_from_q(q_values, valid)
    pe_values, pe_iterations = policy_evaluation(learned_policy, next_index, rewards, terminal, reference["gamma"], theta)
    vi_targets = optimal_targets(reference["values"], next_index, rewards, terminal, valid, reference["gamma"])
    vi_best = np.max(vi_targets, axis=-1)
    learned_is_vi_optimal = np.take_along_axis(vi_targets, learned_policy[..., None], axis=-1)[..., 0] >= vi_best - 1e-9
    masks = {}
    nonterminal = np.ones(reference["values"].shape, dtype=bool)
    q_lookup = {int(q): i for i, q in enumerate(reference["dfa_states"])}
    for q in reference["accepting_states"]:
        if int(q) in q_lookup:
            nonterminal[q_lookup[int(q)]] = False
    masks["all_nonterminal"] = nonterminal
    for q, qi in q_lookup.items():
        if q not in set(map(int, reference["accepting_states"])):
            mask = np.zeros_like(nonterminal)
            mask[qi] = True
            masks[f"q{q}"] = mask
    rows = []
    for method, values in (("Policy Evaluation", pe_values), ("MaxQ", maxq_values)):
        candidate_targets = optimal_targets(values, next_index, rewards, terminal, valid, reference["gamma"])
        residual = np.abs(values - np.max(candidate_targets, axis=-1))
        for scope, mask in masks.items():
            rows.append(metric_row(seed, method, scope, reference["values"], values, mask, reference["goal_reward"], learned_is_vi_optimal, residual))
    return rows, {"vi": reference["values"], "pe": pe_values, "maxq": maxq_values}, pe_iterations


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    aggregate = []
    keys = sorted({(row["method"], row["scope"]) for row in rows})
    for method, scope in keys:
        selected = [row for row in rows if row["method"] == method and row["scope"] == scope]
        entry = {"method": method, "scope": scope, "seeds": len(selected)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            entry[f"{metric}_mean"] = float(np.nanmean(values))
            entry[f"{metric}_std"] = float(np.nanstd(values, ddof=0))
        aggregate.append(entry)
    return aggregate


def direct_comparison_rows(seed, arrays, reference):
    """Quantify disagreement between the two extractions from the same Q."""
    accepting = set(map(int, reference["accepting_states"]))
    q_lookup = {int(q): i for i, q in enumerate(reference["dfa_states"])}
    masks = {}
    nonterminal = np.ones(arrays["pe"].shape, dtype=bool)
    for q in accepting:
        nonterminal[q_lookup[q]] = False
    masks["all_nonterminal"] = nonterminal
    for q, qi in q_lookup.items():
        if q not in accepting:
            mask = np.zeros_like(nonterminal)
            mask[qi] = True
            masks[f"q{q}"] = mask
    rows = []
    scale = abs(reference["goal_reward"]) or 1.0
    for scope, mask in masks.items():
        pe = arrays["pe"][mask]
        maxq = arrays["maxq"][mask]
        difference = maxq - pe
        rows.append({
            "seed": seed,
            "comparison": "MaxQ vs Policy Evaluation",
            "scope": scope,
            "cells": int(pe.size),
            "mae": float(np.mean(np.abs(difference))),
            "normalized_mae": float(np.mean(np.abs(difference)) / scale),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "normalized_rmse": float(np.sqrt(np.mean(difference * difference)) / scale),
            "pearson": correlation(pe, maxq),
            "spearman": correlation(rankdata(pe), rankdata(maxq)),
        })
    return rows


def aggregate_direct_rows(rows):
    metrics = ("mae", "normalized_mae", "rmse", "normalized_rmse", "pearson", "spearman")
    aggregate = []
    for scope in sorted({row["scope"] for row in rows}):
        selected = [row for row in rows if row["scope"] == scope]
        entry = {"comparison": "MaxQ vs Policy Evaluation", "scope": scope, "seeds": len(selected)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            entry[f"{metric}_mean"] = float(np.nanmean(values))
            entry[f"{metric}_std"] = float(np.nanstd(values, ddof=0))
        aggregate.append(entry)
    return aggregate


def save_extracted(path, arrays, reference, seed, pe_iterations):
    np.savez_compressed(
        path,
        vi_values=arrays["vi"],
        policy_evaluation_values=arrays["pe"],
        maxq_values=arrays["maxq"],
        dfa_states=reference["dfa_states"],
        accepting_dfa_states=reference["accepting_states"],
        gamma=np.float64(reference["gamma"]),
        goal_reward=np.float64(reference["goal_reward"]),
        seed=np.int64(seed),
        policy_evaluation_iterations=np.int64(pe_iterations),
    )


def representative_seed(rows):
    scores = {}
    for seed in sorted({row["seed"] for row in rows}):
        selected = [row["normalized_rmse"] for row in rows if row["seed"] == seed and row["scope"] == "all_nonterminal"]
        scores[seed] = float(np.mean(selected))
    median_score = float(np.median(list(scores.values())))
    return min(scores, key=lambda seed: (abs(scores[seed] - median_score), seed))


def save_heatmaps(output_dir, arrays, reference, seed):
    nonaccepting = [int(q) for q in reference["dfa_states"] if int(q) not in set(map(int, reference["accepting_states"]))]
    q_lookup = {int(q): i for i, q in enumerate(reference["dfa_states"])}
    value_panels = (("VI", "vi"), ("Policy Evaluation", "pe"), ("MaxQ", "maxq"))
    vmin = min(float(np.min(arrays[key][q_lookup[q]])) for q in nonaccepting for _, key in value_panels)
    vmax = max(float(np.max(arrays[key][q_lookup[q]])) for q in nonaccepting for _, key in value_panels)
    fig, axes = plt.subplots(len(nonaccepting), 3, figsize=(12.2, 3.7 * len(nonaccepting)), squeeze=False, constrained_layout=True)
    image = None
    for row, q in enumerate(nonaccepting):
        for column, (title, key) in enumerate(value_panels):
            image = axes[row, column].imshow(arrays[key][q_lookup[q]], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].set_ylabel(f"$q_{q}$ — Grid y")
            axes[row, column].set_xlabel("Grid x")
    fig.colorbar(image, ax=axes, label="$V(s)$", shrink=0.88)
    fig.suptitle("Value-function comparison", fontsize=13)
    fig.savefig(output_dir / "value_function_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    errors = (
        ("|PE − VI|", np.abs(arrays["pe"] - arrays["vi"])),
        ("|MaxQ − VI|", np.abs(arrays["maxq"] - arrays["vi"])),
        ("|MaxQ − PE|", np.abs(arrays["maxq"] - arrays["pe"])),
    )
    error_max = max(float(np.max(values[[q_lookup[q] for q in nonaccepting]])) for _, values in errors)
    fig, axes = plt.subplots(len(nonaccepting), 3, figsize=(12.2, 3.7 * len(nonaccepting)), squeeze=False, constrained_layout=True)
    image = None
    for row, q in enumerate(nonaccepting):
        for column, (title, values) in enumerate(errors):
            image = axes[row, column].imshow(values[q_lookup[q]], origin="lower", cmap="magma", vmin=0, vmax=max(error_max, 1e-12), interpolation="nearest")
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].set_ylabel(f"$q_{q}$ — Grid y")
            axes[row, column].set_xlabel("Grid x")
    fig.colorbar(image, ax=axes, label="Absolute value error", shrink=0.88)
    fig.suptitle("Value-function errors", fontsize=13)
    fig.savefig(output_dir / "value_function_errors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(output_dir, rows):
    selected = [row for row in rows if row["scope"] == "all_nonterminal"]
    methods = ("Policy Evaluation", "MaxQ")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    specs = (
        ("normalized_rmse", "Normalized RMSE"),
        ("spearman", "Spearman correlation"),
        ("policy_agreement", "Greedy-policy agreement"),
    )
    colors = ("#1f77b4", "#ff7f0e")
    for axis, (metric, label) in zip(axes, specs):
        data = [[row[metric] for row in selected if row["method"] == method] for method in methods]
        box = axis.boxplot(data, tick_labels=("PE", "MaxQ"), patch_artist=True, widths=0.55)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "value_function_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_latex_table(output_dir, aggregate):
    selected = {row["method"]: row for row in aggregate if row["scope"] == "all_nonterminal"}
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{lcccc}",
        r"        \hline",
        r"        \textbf{Extraction} & \textbf{nMAE} & \textbf{nRMSE} & \textbf{Spearman} & \textbf{Policy agreement} \\",
        r"        \hline",
    ]
    for method in ("Policy Evaluation", "MaxQ"):
        row = selected[method]
        label = "Policy Evaluation" if method == "Policy Evaluation" else r"$\max_a Q(s,a)$"
        lines.append(
            f"        {label} & "
            f"${100*row['normalized_mae_mean']:.2f}\\pm{100*row['normalized_mae_std']:.2f}$ & "
            f"${100*row['normalized_rmse_mean']:.2f}\\pm{100*row['normalized_rmse_std']:.2f}$ & "
            f"${row['spearman_mean']:.3f}\\pm{row['spearman_std']:.3f}$ & "
            f"${100*row['policy_agreement_mean']:.2f}\\pm{100*row['policy_agreement_std']:.2f}\\%$ \\\\"
        )
    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"    \caption{Comparison with the $360\times360$ value-iteration reference, reported as mean $\pm$ standard deviation over five seeds. Errors are normalized by the goal reward.}",
        r"    \label{tab:rq3-value-function-quality}",
        r"\end{table}",
    ])
    (output_dir / "value_function_metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    reference_path = args.reference.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    candidate_paths = sorted(Path().glob(args.candidates_glob) if not Path(args.candidates_glob).is_absolute() else Path("/").glob(args.candidates_glob.lstrip("/")))
    if not reference_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError("Reference or trajectory file not found")
    if not candidate_paths:
        raise FileNotFoundError(f"No candidates match: {args.candidates_glob}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    reference = load_reference(reference_path)
    with trajectory_path.open(encoding="utf-8") as handle:
        trajectory = json.load(handle)
    automaton = build_task_automaton(trajectory)
    _, _, propositions = load_task_propositions(trajectory.get("regions"), trajectory.get("predicates"))
    mdp = LTLfWaypointMDP(propositions, automaton, width=reference["width"], height=reference["height"], gamma=reference["gamma"], goal_reward=reference["goal_reward"], level_name="rq3-block2")
    if set(map(int, reference["dfa_states"])) != set(map(int, automaton.states)):
        raise ValueError("Reference DFA states do not match trajectory")
    print("Building the dense transition model (reused for every seed)...")
    model = build_dense_model(mdp, reference["dfa_states"])

    all_rows = []
    direct_rows = []
    arrays_by_seed = {}
    metadata = []
    used_seeds = set()
    for fallback, path in enumerate(candidate_paths):
        seed = seed_from_path(path, fallback)
        if seed in used_seeds:
            raise ValueError(f"Duplicate seed {seed}: {path}")
        used_seeds.add(seed)
        print(f"Analyzing seed {seed}: {path}")
        q_values = load_candidate(path, reference)
        rows, arrays, iterations = analyze_seed(seed, q_values, reference, model, args.theta)
        all_rows.extend(rows)
        direct_rows.extend(direct_comparison_rows(seed, arrays, reference))
        arrays_by_seed[seed] = arrays
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        save_extracted(seed_dir / "extracted_value_functions.npz", arrays, reference, seed, iterations)
        metadata.append({"seed": seed, "candidate": str(path.resolve()), "policy_evaluation_iterations": iterations})

    aggregate = aggregate_rows(all_rows)
    write_csv(output_dir / "per_seed_metrics.csv", all_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(output_dir / "maxq_vs_policy_evaluation_per_seed.csv", direct_rows)
    write_csv(output_dir / "maxq_vs_policy_evaluation_aggregate.csv", aggregate_direct_rows(direct_rows))
    representative = representative_seed(all_rows)
    save_heatmaps(output_dir / "figures", arrays_by_seed[representative], reference, representative)
    save_boxplot(output_dir / "figures", all_rows)
    save_latex_table(output_dir, aggregate)
    manifest = {
        "reference": str(reference_path),
        "trajectory": str(trajectory_path),
        "theta": args.theta,
        "representative_seed": representative,
        "seeds": metadata,
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Representative seed: {representative}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
