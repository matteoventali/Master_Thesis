#!/usr/bin/env python3
"""Compare learned abstract V-functions against a value-iteration reference."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FRAMEWORK_DIR = Path(__file__).resolve().parent
SRC_DIR = FRAMEWORK_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP
from spatial_regions import load_task_propositions


def parse_args():
    parser = argparse.ArgumentParser(description="Compare VI and learned abstract V-functions numerically and visually.")
    parser.add_argument("--reference", type=Path, help="VI value_function.npz used as the reference.")
    parser.add_argument("--candidate", type=Path, help="Learned value_function.npz containing unbiased and optionally biased values.")
    parser.add_argument("--trajectory", type=Path, help="Task trajectory.json; inferred from the reference experiment when omitted.")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults beside the candidate experiment results.")
    parser.add_argument("--gamma-shaping", type=float, default=1.0, help="Gamma used in F=gamma_shaping*Phi(next)-Phi(state).")
    parser.add_argument("--sign-tolerance", type=float, default=1e-9, help="Absolute tolerance used when classifying shaping-signal signs.")
    parser.add_argument("--scatter-samples", type=int, default=100000, help="Maximum shaping transitions drawn per DFA state and candidate variant.")
    args = parser.parse_args()
    if not 0.0 < args.gamma_shaping <= 1.0:
        parser.error("--gamma-shaping must be in (0, 1]")
    if args.sign_tolerance < 0.0:
        parser.error("--sign-tolerance must be non-negative")
    if args.scatter_samples <= 0:
        parser.error("--scatter-samples must be greater than zero")
    return args


def select_file(title, filetypes):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError("Tkinter is unavailable; pass the file paths with --reference and --candidate") from error
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not selected:
        raise RuntimeError(f"No file selected for: {title}")
    return Path(selected)


def resolve_inputs(args):
    reference_path = args.reference or select_file("Select the VI reference value_function.npz", [("NumPy archives", "*.npz"), ("All files", "*")])
    candidate_path = args.candidate or select_file("Select the learned candidate value_function.npz", [("NumPy archives", "*.npz"), ("All files", "*")])
    reference_path = reference_path.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    for label, path in (("reference", reference_path), ("candidate", candidate_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} NPZ not found: {path}")
    trajectory_path = args.trajectory.expanduser().resolve() if args.trajectory else select_file("Select the trajectory.json used by both experiments", [("JSON files", "*.json"), ("All files", "*")])
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory.json not found: {trajectory_path}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else infer_output_dir(candidate_path)
    return reference_path, candidate_path, trajectory_path.resolve(), output_dir


def infer_output_dir(candidate_path):
    for parent in candidate_path.parents:
        if parent.name == "results" and (parent.parent / "trajectory.json").is_file():
            return parent / "value_function_comparison"
    return candidate_path.parent / "value_function_comparison"


def scalar(data, key):
    if key not in data.files:
        raise ValueError(f"Missing metadata key {key!r}")
    return data[key].item()


def load_value_archive(path, role):
    data = np.load(path, allow_pickle=False)
    required = {"dfa_states", "width", "height", "gamma", "goal_reward"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{role} archive is missing keys: {missing}")
    value_key = "unbiased_values" if "unbiased_values" in data.files else "values"
    values = np.asarray(data[value_key], dtype=np.float64)
    dfa_states = np.asarray(data["dfa_states"], dtype=np.int64)
    expected_shape = (len(dfa_states), int(scalar(data, "height")), int(scalar(data, "width")))
    if values.shape != expected_shape:
        raise ValueError(f"{role} {value_key} has shape {values.shape}, expected {expected_shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{role} {value_key} contains NaN or infinite values")
    archive = {"path": path, "data": data, "values": values, "dfa_states": dfa_states, "width": expected_shape[2], "height": expected_shape[1], "gamma": float(scalar(data, "gamma")), "goal_reward": float(scalar(data, "goal_reward"))}
    if "biased_values" in data.files:
        biased_values = np.asarray(data["biased_values"], dtype=np.float64)
        if biased_values.shape != expected_shape or not np.isfinite(biased_values).all():
            raise ValueError(f"{role} biased_values must be finite and have shape {expected_shape}")
        archive["biased_values"] = biased_values
    return archive


def align_candidate(reference, candidate):
    for key in ("width", "height"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Grid mismatch: reference {key}={reference[key]}, candidate {key}={candidate[key]}")
    for key in ("gamma", "goal_reward"):
        if not math.isclose(reference[key], candidate[key], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Metadata mismatch: reference {key}={reference[key]}, candidate {key}={candidate[key]}")
    reference_states = [int(q) for q in reference["dfa_states"]]
    candidate_indices = {int(q): index for index, q in enumerate(candidate["dfa_states"])}
    if set(reference_states) != set(candidate_indices):
        raise ValueError(f"DFA-state mismatch: reference={reference_states}, candidate={sorted(candidate_indices)}")
    order = [candidate_indices[q] for q in reference_states]
    variants = {"unbiased": candidate["values"][order]}
    if "biased_values" in candidate:
        variants["biased"] = candidate["biased_values"][order]
    return variants


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def basic_metrics(reference_values, candidate_values, goal_reward):
    difference = candidate_values - reference_values
    absolute = np.abs(difference)
    scale = abs(goal_reward) if goal_reward != 0.0 else 1.0
    return {"cells": int(difference.size), "mae": float(np.mean(absolute)), "normalized_mae": float(np.mean(absolute) / scale), "rmse": float(np.sqrt(np.mean(difference * difference))), "normalized_rmse": float(np.sqrt(np.mean(difference * difference)) / scale), "max_error": float(np.max(absolute)), "mean_signed_error": float(np.mean(difference)), "pearson": pearson(reference_values, candidate_values), "within_1pct": float(np.mean(absolute <= 0.01 * scale)), "within_5pct": float(np.mean(absolute <= 0.05 * scale)), "within_10pct": float(np.mean(absolute <= 0.10 * scale))}


def new_signal_stats():
    return {"count": 0, "abs_sum": 0.0, "sq_sum": 0.0, "x_sum": 0.0, "y_sum": 0.0, "xx_sum": 0.0, "yy_sum": 0.0, "xy_sum": 0.0, "same_sign": 0, "vi_positive_learned_negative": 0, "vi_negative_learned_positive": 0, "learned_near_zero": 0, "scatter_x": [], "scatter_y": []}


def update_signal_stats(stats, reference_signal, candidate_signal, tolerance, keep_scatter):
    difference = candidate_signal - reference_signal
    stats["count"] += 1
    stats["abs_sum"] += abs(difference)
    stats["sq_sum"] += difference * difference
    stats["x_sum"] += reference_signal
    stats["y_sum"] += candidate_signal
    stats["xx_sum"] += reference_signal * reference_signal
    stats["yy_sum"] += candidate_signal * candidate_signal
    stats["xy_sum"] += reference_signal * candidate_signal
    reference_sign = 1 if reference_signal > tolerance else -1 if reference_signal < -tolerance else 0
    candidate_sign = 1 if candidate_signal > tolerance else -1 if candidate_signal < -tolerance else 0
    stats["same_sign"] += int(reference_sign == candidate_sign)
    stats["vi_positive_learned_negative"] += int(reference_sign > 0 and candidate_sign < 0)
    stats["vi_negative_learned_positive"] += int(reference_sign < 0 and candidate_sign > 0)
    stats["learned_near_zero"] += int(candidate_sign == 0)
    if keep_scatter:
        stats["scatter_x"].append(reference_signal)
        stats["scatter_y"].append(candidate_signal)


def finalize_signal_stats(stats):
    count = stats["count"]
    if count == 0:
        return {"shaping_transitions": 0, "shaping_mae": float("nan"), "shaping_rmse": float("nan"), "shaping_pearson": float("nan"), "shaping_sign_agreement": float("nan"), "vi_positive_learned_negative": float("nan"), "vi_negative_learned_positive": float("nan"), "learned_near_zero": float("nan")}
    numerator = count * stats["xy_sum"] - stats["x_sum"] * stats["y_sum"]
    denominator = math.sqrt(max(0.0, count * stats["xx_sum"] - stats["x_sum"] ** 2) * max(0.0, count * stats["yy_sum"] - stats["y_sum"] ** 2))
    correlation = numerator / denominator if denominator > 0.0 else float("nan")
    return {"shaping_transitions": count, "shaping_mae": stats["abs_sum"] / count, "shaping_rmse": math.sqrt(stats["sq_sum"] / count), "shaping_pearson": correlation, "shaping_sign_agreement": stats["same_sign"] / count, "vi_positive_learned_negative": stats["vi_positive_learned_negative"] / count, "vi_negative_learned_positive": stats["vi_negative_learned_positive"] / count, "learned_near_zero": stats["learned_near_zero"] / count}


def local_gradient_metrics(reference_values, candidate_values):
    reference_deltas = np.concatenate([(reference_values[:, 1:] - reference_values[:, :-1]).ravel(), (reference_values[1:, :] - reference_values[:-1, :]).ravel()])
    candidate_deltas = np.concatenate([(candidate_values[:, 1:] - candidate_values[:, :-1]).ravel(), (candidate_values[1:, :] - candidate_values[:-1, :]).ravel()])
    difference = candidate_deltas - reference_deltas
    return {"gradient_mae": float(np.mean(np.abs(difference))), "gradient_rmse": float(np.sqrt(np.mean(difference * difference))), "gradient_max_error": float(np.max(np.abs(difference))), "gradient_pearson": pearson(reference_deltas, candidate_deltas), "candidate_total_variation": float(np.mean(np.abs(candidate_deltas))), "reference_total_variation": float(np.mean(np.abs(reference_deltas)))}


def analyze_transitions(mdp, reference_values, variants, dfa_states, gamma_shaping, sign_tolerance, scatter_samples):
    q_indices = {int(q): index for index, q in enumerate(dfa_states)}
    residuals = {variant: defaultdict(list) for variant in variants}
    policy_agreements = {variant: defaultdict(lambda: [0, 0]) for variant in variants}
    shaping_stats = {variant: defaultdict(new_signal_stats) for variant in variants}
    accepting_entry_masks = np.zeros(reference_values.shape, dtype=bool)
    movement_state_count = sum(mdp.width * mdp.height for q in dfa_states if not mdp.automaton.is_goal_reached(int(q)))
    scatter_stride = max(1, movement_state_count * len(mdp.movement_actions) // scatter_samples)
    shaping_index = defaultdict(int)
    for x, y, q in mdp.states:
        q_index = q_indices[q]
        state = (x, y, q)
        actions = mdp.get_available_actions(state)
        reference_targets = []
        candidate_targets = {variant: [] for variant in variants}
        for action in actions:
            next_state, reward, terminal = mdp.get_transitions(state, action)
            next_x, next_y, next_q = next_state
            next_q_index = q_indices[next_q]
            reference_target = reward if terminal else reward + mdp.gamma * reference_values[next_q_index, next_y, next_x]
            reference_targets.append(reference_target)
            if not mdp.automaton.is_goal_reached(q) and mdp.automaton.is_goal_reached(next_q):
                accepting_entry_masks[next_q_index, next_y, next_x] = True
            for variant, candidate_values in variants.items():
                candidate_target = reward if terminal else reward + mdp.gamma * candidate_values[next_q_index, next_y, next_x]
                candidate_targets[variant].append(candidate_target)
                if action in mdp.movement_actions:
                    reference_signal = gamma_shaping * reference_values[next_q_index, next_y, next_x] - reference_values[q_index, y, x]
                    candidate_signal = gamma_shaping * candidate_values[next_q_index, next_y, next_x] - candidate_values[q_index, y, x]
                    keep_scatter = shaping_index[(variant, q)] % scatter_stride == 0 and len(shaping_stats[variant][q]["scatter_x"]) < scatter_samples
                    update_signal_stats(shaping_stats[variant][q], reference_signal, candidate_signal, sign_tolerance, keep_scatter)
                    shaping_index[(variant, q)] += 1
        reference_best = max(reference_targets)
        reference_best_actions = {index for index, target in enumerate(reference_targets) if abs(target - reference_best) <= 1e-9}
        for variant, candidate_values in variants.items():
            candidate_best = max(candidate_targets[variant])
            candidate_best_actions = {index for index, target in enumerate(candidate_targets[variant]) if abs(target - candidate_best) <= 1e-9}
            residuals[variant][q].append(abs(candidate_values[q_index, y, x] - candidate_best))
            policy_agreements[variant][q][0] += int(bool(reference_best_actions.intersection(candidate_best_actions)))
            policy_agreements[variant][q][1] += 1
    return residuals, policy_agreements, shaping_stats, accepting_entry_masks


def residual_metrics(values):
    values = np.asarray(values, dtype=np.float64)
    return {"bellman_mean": float(np.mean(values)), "bellman_rmse": float(np.sqrt(np.mean(values * values))), "bellman_max": float(np.max(values)), "bellman_p50": float(np.percentile(values, 50)), "bellman_p90": float(np.percentile(values, 90)), "bellman_p95": float(np.percentile(values, 95)), "bellman_p99": float(np.percentile(values, 99))}


def make_rows(reference_values, variants, dfa_states, goal_reward, residuals, policy_agreements, shaping_stats, accepting_entry_masks):
    rows = []
    for variant, candidate_values in variants.items():
        for q_index, q in enumerate(dfa_states):
            scopes = [("all", np.ones(reference_values[q_index].shape, dtype=bool))]
            if accepting_entry_masks[q_index].any():
                scopes.append(("accepting-entry-cells", accepting_entry_masks[q_index]))
            for scope_name, mask in scopes:
                row = {"variant": variant, "dfa_state": int(q), "scope": scope_name}
                row.update(basic_metrics(reference_values[q_index][mask], candidate_values[q_index][mask], goal_reward))
                row.update(local_gradient_metrics(reference_values[q_index], candidate_values[q_index]))
                row.update(residual_metrics(residuals[variant][int(q)]))
                agreement, total = policy_agreements[variant][int(q)]
                row["policy_agreement"] = agreement / total if total else float("nan")
                row.update(finalize_signal_stats(shaping_stats[variant][int(q)]))
                rows.append(row)
    return rows


def save_value_figures(output_dir, reference_values, variants, dfa_states):
    for q_index, q in enumerate(dfa_states):
        panels = [("VI reference", reference_values[q_index])] + [(f"{variant.capitalize()} learned", values[q_index]) for variant, values in variants.items()]
        vmin = min(float(np.min(values)) for _, values in panels)
        vmax = max(float(np.max(values)) for _, values in panels)
        figure, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.4), constrained_layout=True)
        axes = np.atleast_1d(axes)
        image = None
        for axis, (label, values) in zip(axes, panels):
            image = axis.imshow(values, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            axis.set_title(label)
            axis.set_xlabel("Grid x")
            axis.set_ylabel("Grid y")
        figure.colorbar(image, ax=list(axes), label="V(s)", shrink=0.88)
        figure.savefig(output_dir / f"values_q{int(q)}.png", dpi=220, bbox_inches="tight")
        plt.close(figure)

        differences = [(f"{variant.capitalize()} − VI", values[q_index] - reference_values[q_index]) for variant, values in variants.items()]
        error_limit = max(max(float(np.max(np.abs(values))) for _, values in differences), 1e-12)
        figure, axes = plt.subplots(1, len(differences), figsize=(5.0 * len(differences), 4.4), constrained_layout=True)
        axes = np.atleast_1d(axes)
        image = None
        for axis, (label, values) in zip(axes, differences):
            image = axis.imshow(values, origin="lower", cmap="coolwarm", vmin=-error_limit, vmax=error_limit, interpolation="nearest")
            axis.set_title(label)
            axis.set_xlabel("Grid x")
            axis.set_ylabel("Grid y")
        figure.colorbar(image, ax=list(axes), label="V error", shrink=0.88)
        figure.savefig(output_dir / f"value_errors_q{int(q)}.png", dpi=220, bbox_inches="tight")
        plt.close(figure)


def save_shaping_figures(output_dir, shaping_stats, dfa_states):
    for q in dfa_states:
        available = [(variant, stats[int(q)]) for variant, stats in shaping_stats.items() if stats[int(q)]["scatter_x"]]
        if not available:
            continue
        figure, axes = plt.subplots(1, len(available), figsize=(5.0 * len(available), 4.4), constrained_layout=True)
        axes = np.atleast_1d(axes)
        all_values = [value for _, stats in available for value in stats["scatter_x"] + stats["scatter_y"]]
        lower, upper = min(all_values), max(all_values)
        for axis, (variant, stats) in zip(axes, available):
            axis.scatter(stats["scatter_x"], stats["scatter_y"], s=3, alpha=0.16, rasterized=True)
            axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
            axis.axhline(0.0, color="gray", linewidth=0.7)
            axis.axvline(0.0, color="gray", linewidth=0.7)
            axis.set_title(f"{variant.capitalize()} shaping")
            axis.set_xlabel("VI shaping signal")
            axis.set_ylabel("Learned shaping signal")
        figure.savefig(output_dir / f"shaping_signal_q{int(q)}.png", dpi=220, bbox_inches="tight")
        plt.close(figure)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def save_metrics(output_dir, rows, metadata):
    fieldnames = list(rows[0])
    with (output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "comparison_metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(json_safe({"metadata": metadata, "metrics": rows}), json_file, indent=2)


def print_summary(rows):
    print("\n=== VALUE-FUNCTION COMPARISON ===")
    for row in rows:
        if row["scope"] != "all":
            continue
        print(f"\n{row['variant']} vs VI | q{row['dfa_state']}")
        print(f"  normalized MAE                 : {row['normalized_mae']:.2%}")
        print(f"  normalized RMSE                : {row['normalized_rmse']:.2%}")
        print(f"  value correlation              : {row['pearson']:.4f}")
        print(f"  policy agreement               : {row['policy_agreement']:.2%}")
        print(f"  Bellman residual mean / p99    : {row['bellman_mean']:.3f} / {row['bellman_p99']:.3f}")
        if math.isfinite(row["shaping_sign_agreement"]):
            print(f"  shaping-signal correlation     : {row['shaping_pearson']:.4f}")
            print(f"  shaping-sign agreement         : {row['shaping_sign_agreement']:.2%}")
            print(f"  VI positive, learned negative  : {row['vi_positive_learned_negative']:.2%}")


def main():
    args = parse_args()
    reference_path, candidate_path, trajectory_path, output_dir = resolve_inputs(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_value_archive(reference_path, "reference")
    candidate = load_value_archive(candidate_path, "candidate")
    variants = align_candidate(reference, candidate)
    with trajectory_path.open(encoding="utf-8") as trajectory_file:
        trajectory = json.load(trajectory_file)
    automaton = LTLfAutomaton(trajectory.get("formula", "F(goal)"))
    _, _, task_propositions = load_task_propositions(trajectory.get("regions"), trajectory.get("predicates"))
    mdp = LTLfWaypointMDP(regions=task_propositions, ltlf_automaton=automaton, width=reference["width"], height=reference["height"], gamma=reference["gamma"], goal_reward=reference["goal_reward"], level_name="comparison")
    reference_states = [int(q) for q in reference["dfa_states"]]
    if set(reference_states) != set(automaton.states):
        raise ValueError(f"trajectory DFA states {sorted(automaton.states)} do not match NPZ states {reference_states}")
    automaton_order = {int(q): index for index, q in enumerate(reference["dfa_states"])}
    reference_values = reference["values"][[automaton_order[q] for q in reference_states]]
    residuals, policy_agreements, shaping_stats, accepting_entry_masks = analyze_transitions(mdp, reference_values, variants, reference_states, args.gamma_shaping, args.sign_tolerance, args.scatter_samples)
    rows = make_rows(reference_values, variants, reference_states, reference["goal_reward"], residuals, policy_agreements, shaping_stats, accepting_entry_masks)
    metadata = {"reference": str(reference_path), "candidate": str(candidate_path), "trajectory": str(trajectory_path), "gamma_shaping": args.gamma_shaping, "grid": [reference["width"], reference["height"]], "dfa_states": reference_states}
    save_metrics(output_dir, rows, metadata)
    save_value_figures(output_dir, reference_values, variants, reference_states)
    save_shaping_figures(output_dir, shaping_stats, reference_states)
    print_summary(rows)
    print(f"\nComparison artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
