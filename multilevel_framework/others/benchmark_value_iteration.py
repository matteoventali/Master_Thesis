#!/usr/bin/env python3
"""Find the largest square grid solved by value iteration within a timeout.

The task is always loaded from the framework's continuous-region
``trajectory.json``.  Only the grid resolution changes.  Evaluations are
strictly sequential: an exponential search brackets the boundary, then a
binary search refines it.

The timeout applies only to value iteration; MDP construction is measured
separately.  This script targets Linux/macOS because it uses SIGALRM to
interrupt the existing implementation without modifying the framework.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_FRAMEWORK = Path(__file__).resolve().parent.parent


class EvaluationTimeout(BaseException):
    """Raised when one grid evaluation exceeds its wall-clock budget."""


@dataclass
class Evaluation:
    status: str
    elapsed_seconds: float
    build_seconds: float | None
    value_iteration_seconds: float | None
    iterations: int | None
    product_states: int | None
    dfa_states: int
    region_cells: dict[str, int]
    error: str = ""

    @property
    def completed(self) -> bool:
        return self.status == "completed"


CSV_FIELDS = [
    "timestamp_utc",
    "config_hash",
    "phase",
    "grid_w",
    "grid_h",
    "grid_cells",
    "dfa_states",
    "product_states",
    "region_cells",
    "status",
    "elapsed_seconds",
    "build_seconds",
    "value_iteration_seconds",
    "iterations",
    "gamma",
    "theta",
    "timeout_seconds",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-dir", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("value_iteration_traj2.csv"))
    parser.add_argument("--initial-size", type=int, default=3)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Value-iteration timeout per grid in seconds (default: 3600)",
    )
    parser.add_argument("--theta", type=float, default=0.001)
    parser.add_argument(
        "--gamma",
        type=float,
        help="Override trajectory.json gamma (default there: 0.99)",
    )
    parser.add_argument("--growth-factor", type=int, default=2)
    parser.add_argument(
        "--max-size",
        type=int,
        help="Optional safety ceiling for the exponential search",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching measurements already present in the CSV",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.theta <= 0:
        raise ValueError("--theta must be greater than zero")
    if args.growth_factor < 2:
        raise ValueError("--growth-factor must be at least 2")
    if args.initial_size is not None and args.initial_size <= 0:
        raise ValueError("--initial-size must be positive")
    if args.max_size is not None and args.max_size <= 0:
        raise ValueError("--max-size must be positive")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configuration_hash(trajectory: dict, gamma: float, theta: float, timeout: float) -> str:
    payload = {
        "trajectory": trajectory,
        "gamma": gamma,
        "theta": theta,
        "timeout_seconds": timeout,
        "benchmark_version": 2,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def read_cached_results(csv_path: Path, config_hash: str) -> dict[int, dict[str, str]]:
    cached: dict[int, dict[str, str]] = {}
    if not csv_path.exists():
        return cached
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("config_hash") != config_hash:
                continue
            if row.get("status") not in {"completed", "timeout"}:
                continue
            width = int(row["grid_w"])
            if width == int(row["grid_h"]):
                cached[width] = row
    return cached


def append_csv(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def iterations_from_traceback(exc: BaseException) -> int | None:
    """Recover the existing value_iteration local counter after SIGALRM."""
    tb = exc.__traceback__
    latest = None
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_name == "value_iteration":
            value = frame.f_locals.get("iterations")
            if isinstance(value, int):
                latest = value
        tb = tb.tb_next
    return latest


def alarm_handler(signum, frame):  # noqa: ARG001
    raise EvaluationTimeout("grid evaluation exceeded its timeout")


def evaluate_grid(
    size: int,
    timeout_seconds: float,
    theta: float,
    gamma: float,
    goal_reward: float,
    regions,
    automaton,
    abstraction_config_class,
    grid_level_class,
    multilevel_mdp_class,
) -> Evaluation:
    started = time.perf_counter()
    build_seconds = None
    vi_started = None
    mdp = None

    old_handler = None
    try:
        config = abstraction_config_class(
            (grid_level_class(width=size, height=size, name="level1", algorithm="value_iteration"),)
        )
        mdp = multilevel_mdp_class(
            regions=regions,
            ltlf_automaton=automaton,
            abstraction_config=config,
            gamma=gamma,
            goal_reward=goal_reward,
        )
        vi_started = time.perf_counter()
        build_seconds = vi_started - started
        old_handler = signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        mdp.compute_value_functions(theta=theta, print_policies=False)
        finished = time.perf_counter()
        level = mdp.primary_mdp
        return Evaluation(
            status="completed",
            elapsed_seconds=finished - started,
            build_seconds=build_seconds,
            value_iteration_seconds=finished - vi_started,
            iterations=level.value_iteration_iterations,
            product_states=len(level.states),
            dfa_states=len(automaton.states),
            region_cells={name: len(cells) for name, cells in level.region_cells.items()},
        )
    except EvaluationTimeout as exc:
        finished = time.perf_counter()
        level = mdp.primary_mdp if mdp is not None else None
        return Evaluation(
            status="timeout",
            elapsed_seconds=finished - started,
            build_seconds=build_seconds,
            value_iteration_seconds=(finished - vi_started) if vi_started is not None else None,
            iterations=iterations_from_traceback(exc),
            product_states=len(level.states) if level is not None else None,
            dfa_states=len(automaton.states),
            region_cells=(
                {name: len(cells) for name, cells in level.region_cells.items()}
                if level is not None
                else {}
            ),
        )
    except Exception:
        finished = time.perf_counter()
        level = mdp.primary_mdp if mdp is not None else None
        return Evaluation(
            status="error",
            elapsed_seconds=finished - started,
            build_seconds=build_seconds,
            value_iteration_seconds=(finished - vi_started) if vi_started is not None else None,
            iterations=None,
            product_states=len(level.states) if level is not None else None,
            dfa_states=len(automaton.states),
            region_cells={},
            error=traceback.format_exc(),
        )
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        del mdp
        gc.collect()


def format_optional(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


def main() -> int:
    args = parse_args()
    validate_args(args)

    framework_dir = args.framework_dir.resolve()
    src_dir = framework_dir / "src"
    trajectory_path = (args.trajectory or framework_dir / "config" / "trajectory.json").resolve()
    csv_path = args.csv.resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Framework src directory not found: {src_dir}")
    sys.path.insert(0, str(src_dir))

    from abstraction import AbstractionConfig, GridLevel
    from abstract_mdps import MultiLevelWaypointMDP, build_task_automaton
    from spatial_regions import load_task_propositions

    trajectory = load_json(trajectory_path)
    initial_size = args.initial_size

    regions, predicates, task_propositions = load_task_propositions(trajectory.get("regions"), trajectory.get("predicates"))
    gamma = args.gamma if args.gamma is not None else float(trajectory.get("gamma", 0.99))
    goal_reward = float(trajectory.get("goal_reward", 10000))
    automaton = build_task_automaton(trajectory)
    config_hash = configuration_hash(trajectory, gamma, args.theta, args.timeout)
    cached = {} if args.no_resume else read_cached_results(csv_path, config_hash)

    print(f"Task: {trajectory_path}")
    print(f"Continuous regions: {', '.join(regions)}")
    print(f"Derived predicates: {', '.join(predicates) if predicates else 'none'}")
    print(f"CSV: {csv_path}")
    print(f"Config hash: {config_hash}")
    print(f"Value-iteration timeout per grid: {args.timeout:g} s")

    def test(size: int, phase: str) -> bool:
        if size in cached:
            status = cached[size]["status"]
            print(f"[{phase}] {size}x{size}: cached {status}")
            return status == "completed"

        print(f"[{phase}] testing {size}x{size}...", flush=True)
        result = evaluate_grid(
            size=size,
            timeout_seconds=args.timeout,
            theta=args.theta,
            gamma=gamma,
            goal_reward=goal_reward,
            regions=task_propositions,
            automaton=automaton,
            abstraction_config_class=AbstractionConfig,
            grid_level_class=GridLevel,
            multilevel_mdp_class=MultiLevelWaypointMDP,
        )
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "phase": phase,
            "grid_w": size,
            "grid_h": size,
            "grid_cells": size * size,
            "dfa_states": result.dfa_states,
            "product_states": format_optional(result.product_states),
            "region_cells": json.dumps(result.region_cells, sort_keys=True),
            "status": result.status,
            "elapsed_seconds": format_optional(result.elapsed_seconds),
            "build_seconds": format_optional(result.build_seconds),
            "value_iteration_seconds": format_optional(result.value_iteration_seconds),
            "iterations": format_optional(result.iterations),
            "gamma": gamma,
            "theta": args.theta,
            "timeout_seconds": args.timeout,
            "error": result.error.replace("\n", "\\n"),
        }
        append_csv(csv_path, row)
        cached[size] = {key: str(value) for key, value in row.items()}
        print(
            f"[{phase}] {size}x{size}: {result.status}, "
            f"elapsed={result.elapsed_seconds:.3f}s, iterations={result.iterations}"
        )
        if result.status == "error":
            raise RuntimeError(f"Evaluation failed for {size}x{size}:\n{result.error}")
        return result.completed

    low = None
    candidate = initial_size
    while True:
        if args.max_size is not None and candidate > args.max_size:
            print(f"Reached --max-size={args.max_size} before finding a timeout.")
            return 2
        if test(candidate, "exponential"):
            low = candidate
            candidate *= args.growth_factor
        else:
            high = candidate
            break

    if low is None:
        print(f"No feasible grid found: initial {initial_size}x{initial_size} timed out.")
        return 3

    while high - low > 1:
        middle = (low + high) // 2
        if test(middle, "binary"):
            low = middle
        else:
            high = middle

    print("\nSearch complete")
    print(f"Largest completed grid: {low}x{low}")
    print(f"Smallest timed-out grid: {high}x{high}")
    print(f"Measurements: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
