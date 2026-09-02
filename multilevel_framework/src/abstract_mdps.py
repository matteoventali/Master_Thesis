import math
import os
import random
import re
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from abstraction import map_state
from spatial_regions import (
    CircularRegion,
    rasterize_regions,
    truth_assignment_from_cell,
    truth_assignment_from_observation,
)


@dataclass(frozen=True)
class AutomatonStep:
    """Outcome of consuming one spatial valuation."""

    next_state: int
    accepted: bool = False
    failed: bool = False
    completed_cycle: bool = False
    reached_proposition: str | None = None

    @property
    def succeeded(self):
        return self.accepted or self.completed_cycle

    @property
    def terminal(self):
        return self.accepted or self.failed


def build_task_automaton(config):
    """Build an LTLf or continuing cyclic automaton from a task config."""
    task_type = config.get("task_type")
    if task_type is None:
        task_type = "cyclic_waypoints" if "waypoint_cycle" in config else "ltlf"
    if task_type == "ltlf":
        if "waypoint_cycle" in config:
            raise ValueError("An LTLf task must not define waypoint_cycle")
        return LTLfAutomaton(config.get("formula", "F(goal)"))
    if task_type == "cyclic_waypoints":
        if "formula" in config:
            raise ValueError("A cyclic_waypoints task must not define formula")
        return CyclicWaypointsAutomaton(config.get("waypoint_cycle"))
    raise ValueError("task_type must be either 'ltlf' or 'cyclic_waypoints'")


class LTLfAutomaton:
    """
    Wrap ltlf2dfa and expose its DFA as a graph that can be traversed by the MDP.
    """
    def __init__(self, formula_str):
        # Keep MDP/mapping utilities importable without the optional DFA parser.
        from ltlf2dfa.parser.ltlf import LTLfParser

        self.formula_str = formula_str
        self.task_type = "ltlf"
        self.is_continuing = False
        
        # Parse the formula and generate its DFA in DOT format.
        parser = LTLfParser()
        parsed_formula = parser(formula_str)
        dot_string = parsed_formula.to_dfa()
        self.dot_string = parsed_formula.to_dfa()
        
        # Initialize the automaton data structures.
        self.states = set()
        self.accepting_states = set()
        self.transitions = {}  # {source_state: [(Boolean_guard, destination_state), ...]}
        self.initial_state = None
        
        # Extract states and transitions from the DOT representation.
        self._parse_dot(dot_string)
        
        # Keep a stable state order for the MDP and one-hot encodings.
        self.states = sorted(list(self.states))
        self.num_phases = len(self.states)
        self.failure_states = self._compute_failure_states()

    def _parse_dot(self, dot_string):
        """
        Parse the DOT output and extract states, accepting states, the initial
        state, and guarded transitions.
        """
        # Extract accepting states, e.g. node [shape = doublecircle]; 2 3;.
        match_acc = re.search(r'node\s*\[shape\s*=\s*doublecircle\]\s*;\s*(.*?);', dot_string)
        if match_acc:
            acc_str = match_acc.group(1).replace(',', ' ')
            self.accepting_states = set(int(s) for s in acc_str.split() if s.strip().isdigit())
            
        # Extract guarded transitions, e.g. 1 -> 2 [label="wp1 & ~wp2"].
        trans_matches = re.findall(r'(\d+)\s*->\s*(\d+)\s*\[label\s*=\s*"(.*?)"\]', dot_string)
        for src_str, dst_str, guard in trans_matches:
            src = int(src_str)
            dst = int(dst_str)
            self.states.add(src)
            self.states.add(dst)
            
            if src not in self.transitions:
                self.transitions[src] = []
            self.transitions[src].append((guard, dst))
            
        # Extract the initial state from the unlabeled edge leaving the invisible node.
        # Example: 0 [style=invis]; 0 -> 1;.
        init_match = re.search(r'(\d+)\s*->\s*(\d+)\s*;', dot_string)
        if init_match:
            self.initial_state = int(init_match.group(2))
        else:
            self.initial_state = min(self.states) if self.states else 0

    def get_initial_q(self):
        """Return the identifier of the DFA pre-trace state."""
        return self.initial_state

    def is_goal_reached(self, current_q):
        """Return whether the current DFA state is accepting."""
        return current_q in self.accepting_states

    def _compute_failure_states(self):
        """Return states from which no accepting state is graph-reachable."""
        reverse_edges = defaultdict(set)
        for source, guarded_destinations in self.transitions.items():
            for _, destination in guarded_destinations:
                reverse_edges[destination].add(source)
        acceptance_reachable = set(self.accepting_states)
        frontier = list(self.accepting_states)
        while frontier:
            destination = frontier.pop()
            for source in reverse_edges[destination]:
                if source not in acceptance_reachable:
                    acceptance_reachable.add(source)
                    frontier.append(source)
        return set(self.states).difference(acceptance_reachable)

    def is_failure(self, current_q):
        """Return whether acceptance is irreversibly unreachable from a state."""
        return current_q in self.failure_states

    def is_terminal(self, current_q):
        """Return whether a state ends the task with either success or failure."""
        return self.is_goal_reached(current_q) or self.is_failure(current_q)

    def get_next_q(self, current_q, truth_assignment):
        """
        Evaluate outgoing transition guards and return the next DFA state.
        """
        if current_q not in self.transitions:
            return current_q
            
        for guard, next_q in self.transitions[current_q]:
            if self._eval_guard(guard, truth_assignment):
                return next_q
                
        return current_q

    def advance(self, current_q, truth_assignment):
        """Consume one valuation and expose success/failure as explicit events."""
        next_q = self.get_next_q(current_q, truth_assignment)
        return AutomatonStep(
            next_state=next_q,
            accepted=self.is_goal_reached(next_q),
            failed=self.is_failure(next_q),
        )

    def _eval_guard(self, guard, truth_assignment):
        """
        Convert a DOT guard such as "wp1 & ~wp2" to Python syntax and evaluate
        it against the current truth assignment.
        """
        guard = guard.strip()
        
        # Handle numeric and textual Boolean constants.
        if guard.lower() in ["1", "true"]: return True
        if guard.lower() in ["0", "false"]: return False
        
        # Convert the standard Boolean operators to Python syntax.
        expr = guard.replace('&', ' and ').replace('|', ' or ').replace('~', ' not ').replace('!', ' not ')
        
        try:
            # Disable built-ins while evaluating the Boolean expression.
            return eval(expr, {"__builtins__": {}}, truth_assignment)
        except Exception as e:
            print(f"[LTLfAutomaton error] Could not evaluate transition guard '{guard}': {e}")
            return False

    def render_graph(self, filename="ltlf_automaton", directory="img"):
        """Render the DFA and save it as a PNG image."""
        try:
            from graphviz import Source

            # ltlf2dfa emits a left-to-right graph.  With complex formulae the
            # transition guards become wide, leaving the resulting PNG only a
            # few pixels high.  A top-to-bottom layout gives labels enough room
            # and keeps the automaton readable independently of formula length.
            render_dot = re.sub(r"rankdir\s*=\s*LR\s*;", "rankdir = TB;", self.dot_string, count=1)
            render_dot = re.sub(r"(digraph[^{]*\{)", r"\1\n" r'graph [pad="0.35", nodesep="0.55", ranksep="0.75"];' "\n" r'node [width="0.55", height="0.55"];' "\n" r'edge [fontsize="10"];', render_dot, count=1)
            src = Source(render_dot)
            src.render(filename=filename, directory=directory, format='png', cleanup=True)
            print(f"Automaton graph saved to: {directory}/{filename}.png")
        except Exception as e:
            print(f"[Graphviz error] Could not render the automaton graph: {e}")


class CyclicWaypointsAutomaton:
    """Continuing automaton that repeatedly visits an ordered waypoint cycle."""

    def __init__(self, waypoint_cycle: Sequence[str] | None):
        if waypoint_cycle is None:
            raise ValueError("A cyclic_waypoints task must define waypoint_cycle")
        if isinstance(waypoint_cycle, (str, bytes)) or not isinstance(waypoint_cycle, Sequence):
            raise TypeError("waypoint_cycle must be a sequence of proposition names")
        cycle = tuple(waypoint_cycle)
        if not cycle:
            raise ValueError("waypoint_cycle must contain at least one waypoint")
        if any(not isinstance(name, str) or not name for name in cycle):
            raise ValueError("Waypoint proposition names must be non-empty strings")
        if len(set(cycle)) != len(cycle):
            raise ValueError("waypoint_cycle cannot contain duplicate propositions")

        self.task_type = "cyclic_waypoints"
        self.is_continuing = True
        self.waypoint_cycle = cycle
        self.states = list(range(len(cycle)))
        self.initial_state = self.states[0]
        self.accepting_states = set()
        self.failure_states = set()
        self.num_phases = len(self.states)
        self.formula_str = "cycle(" + ", ".join(cycle) + ")"
        self.transitions = {}

    @property
    def required_propositions(self):
        return frozenset(self.waypoint_cycle)

    def get_initial_q(self):
        return self.initial_state

    def is_goal_reached(self, current_q):
        self._validate_state(current_q)
        return False

    def is_failure(self, current_q):
        self._validate_state(current_q)
        return False

    def is_terminal(self, current_q):
        self._validate_state(current_q)
        return False

    def advance(self, current_q, truth_assignment: Mapping[str, bool]):
        self._validate_state(current_q)
        expected = self.waypoint_cycle[current_q]
        if not bool(truth_assignment.get(expected, False)):
            return AutomatonStep(next_state=current_q)
        completed_cycle = current_q == self.states[-1]
        next_q = self.initial_state if completed_cycle else current_q + 1
        return AutomatonStep(
            next_state=next_q,
            completed_cycle=completed_cycle,
            reached_proposition=expected,
        )

    def get_next_q(self, current_q, truth_assignment):
        return self.advance(current_q, truth_assignment).next_state

    def validate_propositions(self, propositions):
        missing = sorted(self.required_propositions - set(propositions))
        if missing:
            raise ValueError(f"Missing required cycle propositions: {missing}")

    def render_graph(self, filename="cyclic_waypoints_automaton", directory="img"):
        try:
            from graphviz import Source

            lines = ["digraph cyclic_waypoints {", "    rankdir=LR;", "    node [shape=circle];", "    start [shape=point];", f"    start -> {self.initial_state};"]
            for state, waypoint in enumerate(self.waypoint_cycle):
                destination = self.initial_state if state == self.states[-1] else state + 1
                escaped = waypoint.replace("\\", "\\\\").replace('"', '\\"')
                suffix = " / reward" if state == self.states[-1] else ""
                lines.append(f'    {state} -> {destination} [label="{escaped}{suffix}"];')
                lines.append(f'    {state} -> {state} [label="not {escaped}"];')
            lines.append("}")
            Source("\n".join(lines)).render(filename=filename, directory=str(Path(directory)), format="png", cleanup=True)
            print(f"Automaton graph saved to: {directory}/{filename}.png")
        except Exception as error:
            print(f"[Graphviz error] Could not render the automaton graph: {error}")

    def _validate_state(self, state):
        if state not in self.states:
            raise ValueError(f"Unknown automaton state {state!r}")


class LTLfWaypointMDP:
    """
    Abstract product MDP guided by an LTLf or cyclic task automaton.
    Each abstract state is (x, y, q), where q is the automaton state.
    """
    DONE_ACTION = 8

    def __init__(self, regions, ltlf_automaton, width=12, height=12, gamma=0.99, goal_reward=10000, level_name="level1"):
        if width <= 0 or height <= 0:
            raise ValueError("Abstract grid dimensions must be positive")
        self.width = width
        self.height = height
        self.level_name = level_name
        self.gamma = gamma
        self.movement_actions = [0, 1, 2, 3, 4, 5, 6, 7] # Include diagonal movements.
        self.actions = self.movement_actions if ltlf_automaton.is_continuing else self.movement_actions + [self.DONE_ACTION]
        
        self.regions = regions
        self.region_cells = rasterize_regions(regions, width, height)
        self.automaton = ltlf_automaton
        self.num_phases = self.automaton.num_phases
        
        # Generate every combination of grid position and DFA state.
        self.states = [(x, y, q) for x in range(width) for y in range(height) for q in self.automaton.states]
        
        self.goal_reward = goal_reward
        self.v_star = defaultdict(float)
        self.unbiased_v_star = self.v_star
        self.unbiased_q = None
        self.biased_v_star = None
        self.upper_level_mdp = None
        self.inter_level_gamma_shaping = gamma
        self.value_iteration_iterations = 0
        self.solution_algorithm = None
        self.learning_episodes = 0
        self.learning_updates = 0
        self.learning_history = None
        self.value_function_method = None
        self.policy_evaluation_iterations = 0
        self.checkpoint_path = None
        
    def _get_truth_assignment(self, x, y):
        """
        Map the current grid coordinates to a Boolean proposition assignment.
        """
        return truth_assignment_from_cell(self.region_cells, x, y)

    def get_environment_truth_assignment(self, observation):
        """Evaluate propositions exactly on a continuous environment state."""
        return truth_assignment_from_observation(self.regions, observation)

    def get_available_actions(self, state):
        """Return only done at success/failure terminals, movements elsewhere."""
        return [self.DONE_ACTION] if self.automaton.is_terminal(state[2]) else self.movement_actions

    def get_transition_outcome(self, state, action):
        """Apply an abstract action and retain the automaton event metadata."""
        x, y, q = state
        available_actions = self.get_available_actions(state)
        if action not in available_actions:
            raise ValueError(f"Action {action} is not available in state {state}; expected one of {available_actions}")
        if action == self.DONE_ACTION:
            reward = self.goal_reward if self.automaton.is_goal_reached(q) else 0.0
            step = AutomatonStep(
                next_state=q,
                accepted=self.automaton.is_goal_reached(q),
                failed=self.automaton.is_failure(q),
            )
            return state, reward, True, step
        
        # Apply the abstract physical movement.
        next_y = y
        if action in [0, 4, 5]:    next_y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  next_y = max(y - 1, 0)
            
        next_x = x
        if action in [2, 4, 6]:    next_x = max(x - 1, 0)
        elif action in [3, 5, 7]:  next_x = min(x + 1, self.width - 1)
        
        # Evaluate propositions at the arrival coordinates.
        truth_assignment = self._get_truth_assignment(next_x, next_y)
        
        # Advance the automaton using the arrival-state valuation.
        step = self.automaton.advance(q, truth_assignment)
        reward = self.goal_reward if step.completed_cycle else 0.0
        return (next_x, next_y, step.next_state), reward, False, step

    def map_state_to_upper_level(self, state):
        """Map a state spatially while preserving its real-trace DFA state.

        Task progress belongs to the real continuous trace. Mapping a product
        state between spatial abstractions must therefore preserve ``q``.
        """
        if self.upper_level_mdp is None:
            raise ValueError(f"{self.level_name} has no upper abstraction level")
        upper_state = map_state(state, source_width=self.width, source_height=self.height, target_width=self.upper_level_mdp.width, target_height=self.upper_level_mdp.height)
        return upper_state

    def get_upper_level_potential(self, state):
        """Map ``state`` canonically and read the upper-level V*."""
        if self.upper_level_mdp is None:
            return 0.0
        upper_state = self.map_state_to_upper_level(state)
        return self.upper_level_mdp.v_star.get(upper_state, 0.0)

    def get_inter_level_shaping_reward(self, state, next_state):
        """Return the gamma-discounted inter-level potential difference."""
        if self.upper_level_mdp is None:
            return 0.0
        state_potential = self.get_upper_level_potential(state)
        next_state_potential = self.get_upper_level_potential(next_state)
        return self.inter_level_gamma_shaping * next_state_potential - state_potential

    def print_policy(self):
        arrows = {
            0: "↑",
            1: "↓",
            2: "←",
            3: "→",
            4: "↖",
            5: "↗",
            6: "↙",
            7: "↘",
            self.DONE_ACTION: "✓",
        }

        for q in self.automaton.states:
            print(f"\n===== POLICY - DFA STATE q={q} =====")

            for y in reversed(range(self.height)):
                row = []

                for x in range(self.width):
                    state = (x, y, q)

                    if self.automaton.is_goal_reached(q):
                        row.append(" G ")
                        continue
                    if self.automaton.is_failure(q):
                        row.append(" E ")
                        continue

                    best_action = None
                    best_value = -float("inf")

                    for a in self.get_available_actions(state):
                        next_state, reward, terminal, _ = self.get_transition_outcome(state, a)
                        shaping_reward = 0.0 if terminal else self.get_inter_level_shaping_reward(state, next_state)
                        value = reward if terminal else reward + shaping_reward + self.gamma * self.v_star[next_state]

                        if value > best_value:
                            best_value = value
                            best_action = a

                    row.append(f" {arrows[best_action]} ")

                print("".join(row))
    
    def value_iteration(self, theta=0.001, print_policy=True):
        """Compute the unbiased V* of a top abstraction selected for VI."""
        if theta <= 0:
            raise ValueError("theta must be greater than zero")

        self.upper_level_mdp = None
        self.v_star = defaultdict(float)
        self.unbiased_v_star = self.v_star
        self.unbiased_q = None
        self.biased_v_star = None
        self.learning_history = None
        print(f"Value Iteration [{self.level_name}: {self.width}x{self.height}, top-level unbiased solution]...")

        iterations = 0
        while True:
            iterations += 1
            delta = 0
            new_v = self.v_star.copy()
            for s in self.states:
                v_actions = []
                for action in self.get_available_actions(s):
                    next_state, reward, terminal, _ = self.get_transition_outcome(s, action)
                    v_actions.append(reward if terminal else reward + self.gamma * self.v_star[next_state])
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - self.v_star[s]))
                new_v[s] = best_v
            self.v_star = new_v
            if delta < theta:
                break

        self.unbiased_v_star = self.v_star
        self.value_iteration_iterations = iterations
        self.solution_algorithm = "value_iteration"
        self.value_function_method = "value_iteration"
        self.policy_evaluation_iterations = 0
        self.checkpoint_path = None
        self.learning_episodes = 0
        self.learning_updates = 0
        if print_policy:
            self.print_policy()
        return self.v_star

    def _max_q_value_function(self, q_table):
        """Return ``max_a Q(s, a)`` over the actions available in each state."""
        state_index = {state: index for index, state in enumerate(self.states)}
        action_index = {action: index for index, action in enumerate(self.actions)}
        return defaultdict(float, {
            state: max(q_table[state_index[state]][action_index[action]] for action in self.get_available_actions(state))
            for state in self.states
        })

    def policy_evaluation(self, q_unbiased, theta=0.001):
        """Evaluate the deterministic greedy policy induced by ``q_unbiased``.

        Ties use the same deterministic rule as greedy evaluation: the action
        with the smallest numeric identifier wins.  The Bellman expectation
        equation is solved for the infinite-horizon discounted abstract MDP.
        """
        if theta <= 0:
            raise ValueError("theta must be greater than zero")
        state_index = {state: index for index, state in enumerate(self.states)}
        action_index = {action: index for index, action in enumerate(self.actions)}
        policy = {
            state: max(
                self.get_available_actions(state),
                key=lambda action: (q_unbiased[state_index[state]][action_index[action]], -action),
            )
            for state in self.states
        }
        values = defaultdict(float)
        iterations = 0
        while True:
            iterations += 1
            delta = 0.0
            new_values = values.copy()
            for state in self.states:
                next_state, reward, terminal, _ = self.get_transition_outcome(state, policy[state])
                value = reward if terminal else reward + self.gamma * values[next_state]
                delta = max(delta, abs(value - values[state]))
                new_values[state] = value
            values = new_values
            if delta < theta:
                break
        self.policy_evaluation_iterations = iterations
        return values

    def load_checkpoint(self, checkpoint_path, upper_level_mdp=None, print_policy=True):
        """Load an unbiased abstract Q/V checkpoint and skip level training."""
        checkpoint_path = os.fspath(checkpoint_path)
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            if "q_function_unbiased" not in checkpoint.files:
                raise ValueError(f"Checkpoint has no q_function_unbiased: {checkpoint_path}")
            value_key = "v_function_unbiased" if "v_function_unbiased" in checkpoint.files else "unbiased_values"
            if value_key not in checkpoint.files or "dfa_states" not in checkpoint.files:
                raise ValueError(f"Checkpoint has no unbiased V-function or dfa_states: {checkpoint_path}")
            q_values = np.asarray(checkpoint["q_function_unbiased"], dtype=np.float64)
            v_values = np.asarray(checkpoint[value_key], dtype=np.float64)
            saved_dfa_states = [int(q) for q in np.asarray(checkpoint["dfa_states"]).tolist()]
            saved_value_function_method = str(checkpoint["value_function_method"].item()) if "value_function_method" in checkpoint.files else "checkpoint"
            saved_task_type = str(checkpoint["task_type"].item()) if "task_type" in checkpoint.files else None

        if saved_task_type is not None and saved_task_type != self.automaton.task_type:
            raise ValueError(
                f"Checkpoint task_type {saved_task_type!r} does not match "
                f"the current task_type {self.automaton.task_type!r}"
            )

        expected_q_shape = (len(saved_dfa_states), self.height, self.width, len(self.actions))
        expected_v_shape = (len(saved_dfa_states), self.height, self.width)
        if q_values.shape != expected_q_shape or v_values.shape != expected_v_shape:
            raise ValueError(
                f"Checkpoint arrays cannot be indexed on {self.width}x{self.height}: "
                f"Q shape={q_values.shape}, V shape={v_values.shape}"
            )
        saved_q_index = {q: index for index, q in enumerate(saved_dfa_states)}
        missing_q = sorted(set(self.automaton.states).difference(saved_q_index))
        if missing_q:
            raise ValueError(f"Checkpoint has no rows for DFA states: {missing_q}")

        self.upper_level_mdp = upper_level_mdp
        self.unbiased_q = [
            q_values[saved_q_index[state[2]], state[1], state[0], :].tolist()
            for state in self.states
        ]
        self.unbiased_v_star = defaultdict(float, {
            state: float(v_values[saved_q_index[state[2]], state[1], state[0]])
            for state in self.states
        })
        self.v_star = self.unbiased_v_star
        self.biased_v_star = None
        self.solution_algorithm = "checkpoint"
        self.value_function_method = saved_value_function_method
        self.value_iteration_iterations = 0
        self.policy_evaluation_iterations = 0
        self.learning_episodes = 0
        self.learning_updates = 0
        self.learning_history = None
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"Checkpoint loaded [{self.level_name}, V={self.value_function_method}] <- {self.checkpoint_path}")
        if print_policy:
            self.print_policy()
        return self.v_star

    def q_learning(self, config, upper_level_mdp=None, print_policy=True, log_file=None, value_function_method="max", policy_evaluation_theta=0.001):
        """Learn an unbiased value estimate.

        A top level uses classic single-table Q-learning. A lower level uses a
        biased table for PBRS-guided behaviour and an unbiased table for the
        value function exported to the next lower level.
        """
        self.upper_level_mdp = upper_level_mdp
        self.value_iteration_iterations = 0
        self.policy_evaluation_iterations = 0
        self.checkpoint_path = None
        if value_function_method not in ("max", "policy_evaluation"):
            raise ValueError("value_function_method must be either 'max' or 'policy_evaluation'")
        uses_shaping = upper_level_mdp is not None
        self.inter_level_gamma_shaping = self.gamma if config.gamma_shaping is None else config.gamma_shaping
        rng = random.Random(config.seed)
        state_index = {state: index for index, state in enumerate(self.states)}
        action_index = {action: index for index, action in enumerate(self.actions)}
        q_unbiased = [[0.0 for _ in self.actions] for _ in self.states]
        q_biased = [[0.0 for _ in self.actions] for _ in self.states] if uses_shaping else q_unbiased
        restart_states = self.states
        if self.automaton.is_continuing:
            recoverable_non_accepting_q = set(self.automaton.states)
        else:
            reverse_dfa_edges = defaultdict(set)
            for source_q, guarded_destinations in self.automaton.transitions.items():
                for _, destination_q in guarded_destinations:
                    reverse_dfa_edges[destination_q].add(source_q)
            acceptance_reachable_q = set(self.automaton.accepting_states)
            reachability_frontier = list(self.automaton.accepting_states)
            while reachability_frontier:
                destination_q = reachability_frontier.pop()
                for source_q in reverse_dfa_edges[destination_q]:
                    if source_q not in acceptance_reachable_q:
                        acceptance_reachable_q.add(source_q)
                        reachability_frontier.append(source_q)
            recoverable_non_accepting_q = acceptance_reachable_q.difference(self.automaton.accepting_states)
        evaluation_restart_states = [state for state in self.states if state[2] in recoverable_non_accepting_q]
        full_formula_restart_states = [(x, y, self.automaton.get_initial_q()) for x in range(self.width) for y in range(self.height)]
        evaluation_rng = random.Random(config.eval_seed)
        evaluation_starts = [evaluation_rng.choice(evaluation_restart_states) for _ in range(config.eval_episodes)] if evaluation_restart_states else []
        full_formula_rng = random.Random(config.eval_seed + 1)
        full_formula_evaluation_starts = [full_formula_rng.choice(full_formula_restart_states) for _ in range(config.eval_episodes)] if full_formula_restart_states else []

        epsilon = config.epsilon_start
        updates = 0
        total_valid_pairs = sum(len(self.get_available_actions(state)) for state in self.states)
        unbiased_positive_mask = bytearray(len(self.states) * len(self.actions))
        unbiased_positive_pairs = 0
        unbiased_td_sum = 0.0
        unbiased_td_count = 0
        unbiased_td_max = 0.0
        biased_td_sum = 0.0
        biased_td_count = 0
        biased_td_max = 0.0
        learning_history = {"episodes": [], "epsilon": [], "unbiased_episode_reward": [], "successes": [], "episode_lengths": [], "dfa_transitions": [], "initial_acceptances": [], "evaluation_steps": [], "unbiased_eval_success_rates": [], "unbiased_eval_episode_lengths": [], "unbiased_full_eval_success_rates": [], "unbiased_full_eval_episode_lengths": []}
        if uses_shaping:
            learning_history["biased_episode_reward"] = []
            learning_history["biased_eval_success_rates"] = []
            learning_history["biased_eval_episode_lengths"] = []
            learning_history["biased_full_eval_success_rates"] = []
            learning_history["biased_full_eval_episode_lengths"] = []

        learning_description = "dual-table inter-level PBRS" if uses_shaping else "classic single-table unbiased"
        log_handle = None
        if log_file is not None:
            log_directory = os.path.dirname(os.fspath(log_file))
            if log_directory:
                os.makedirs(log_directory, exist_ok=True)
            log_handle = open(log_file, "w", encoding="utf-8")

        def log(message):
            print(message)
            if log_handle is not None:
                log_handle.write(message + "\n")
                log_handle.flush()

        def finite_mean(values):
            finite_values = [value for value in values if math.isfinite(value)]
            return sum(finite_values) / len(finite_values) if finite_values else float("nan")

        def format_percentage(value):
            return "n/a" if not math.isfinite(value) else f"{value:.1%}"

        def evaluate_greedy(q_table, starts):
            if not starts:
                return float("nan"), float("nan"), Counter(), {}
            successes = 0
            total_steps = 0
            transition_counts = Counter()
            results_by_initial_q = defaultdict(lambda: {"episodes": 0, "successes": 0, "steps": 0})
            for evaluation_start in starts:
                evaluation_state = evaluation_start
                evaluation_steps = 0
                initial_q = evaluation_start[2]
                evaluation_succeeded = False
                for evaluation_step in range(config.max_steps + 1):
                    if evaluation_step == config.max_steps and not self.automaton.is_goal_reached(evaluation_state[2]):
                        break
                    evaluation_state_i = state_index[evaluation_state]
                    evaluation_actions = self.get_available_actions(evaluation_state)
                    evaluation_action = max(evaluation_actions, key=lambda candidate: (q_table[evaluation_state_i][action_index[candidate]], -candidate))
                    previous_q = evaluation_state[2]
                    evaluation_state, _, evaluation_terminal, automaton_step = self.get_transition_outcome(evaluation_state, evaluation_action)
                    if evaluation_state[2] != previous_q:
                        transition_counts[(previous_q, evaluation_state[2])] += 1
                    evaluation_steps += 1
                    if automaton_step.succeeded:
                        evaluation_succeeded = True
                    # Finite LTLf tasks retain the abstract DONE transition,
                    # which is where their reward and terminal flag live.
                    # Continuing tasks have no DONE action, so their greedy
                    # diagnostic stops after observing the first full cycle.
                    cycle_completed = self.automaton.is_continuing and automaton_step.completed_cycle
                    if evaluation_terminal or cycle_completed:
                        successes += int(evaluation_succeeded)
                        break
                total_steps += evaluation_steps
                results_by_initial_q[initial_q]["episodes"] += 1
                results_by_initial_q[initial_q]["successes"] += int(evaluation_succeeded)
                results_by_initial_q[initial_q]["steps"] += evaluation_steps
            return successes / len(starts), total_steps / len(starts), transition_counts, dict(results_by_initial_q)

        def format_evaluation_transitions(label, transition_counts):
            transition_lines = "\n".join(f"  {source} -> {target} : {count}" for (source, target), count in sorted(transition_counts.items()))
            return f"DFA transitions {label}       :\n{transition_lines if transition_lines else '  none'}"

        def format_evaluation_by_initial_q(label, results_by_initial_q):
            result_lines = "\n".join(f"  start q{initial_q}: {result['successes']}/{result['episodes']} = {format_percentage(result['successes'] / result['episodes'])}, length={result['steps'] / result['episodes']:.1f}" for initial_q, result in sorted(results_by_initial_q.items()))
            return f"success by initial q ({label}) :\n{result_lines if result_lines else '  none'}"

        start_time = time.monotonic()
        log(f"Q-learning [{self.level_name}: {self.width}x{self.height}, {learning_description}, episodes={config.episodes}]...")
        log(f"Configuration: max_steps={config.max_steps}, alpha={config.alpha}, epsilon_start={config.epsilon_start}, epsilon_min={config.epsilon_min}, epsilon_decay={config.epsilon_decay}, gamma_shaping={self.inter_level_gamma_shaping if uses_shaping else 'not applicable'}, seed={config.seed}, states={len(self.states)}, eval_interval={config.eval_interval}, eval_episodes={config.eval_episodes}, eval_seed={config.eval_seed}")

        for episode in range(1, config.episodes + 1):
            # Random restarts cover the full product space. Accepting states
            # and failure states must also be sampled so done is learned.
            state = rng.choice(restart_states)
            started_accepting = self.automaton.is_goal_reached(state[2])
            started_terminal = self.automaton.is_terminal(state[2])
            biased_episode_reward = 0.0
            unbiased_episode_reward = 0.0
            episode_steps = 0
            episode_dfa_transitions = 0
            episode_succeeded = False
            for step in range(config.max_steps + 1):
                if step == config.max_steps and not self.automaton.is_goal_reached(state[2]):
                    break
                state_row = q_biased[state_index[state]]
                available_actions = self.get_available_actions(state)
                if rng.random() < epsilon:
                    action = rng.choice(available_actions)
                else:
                    best = max(state_row[action_index[candidate]] for candidate in available_actions)
                    candidates = [candidate for candidate in available_actions if abs(state_row[action_index[candidate]] - best) <= 1e-12]
                    action = rng.choice(candidates)

                next_state, reward, terminal, automaton_step = self.get_transition_outcome(state, action)
                episode_steps += 1
                if next_state[2] != state[2]:
                    episode_dfa_transitions += 1
                shaping_reward = 0.0 if terminal else self.get_inter_level_shaping_reward(state, next_state)
                biased_episode_reward += reward + shaping_reward
                unbiased_episode_reward += reward
                state_i = state_index[state]
                next_i = state_index[next_state]
                action_i = action_index[action]

                next_actions = self.get_available_actions(next_state)
                next_unbiased_value = max(q_unbiased[next_i][action_index[candidate]] for candidate in next_actions)
                unbiased_target = reward if terminal else reward + self.gamma * next_unbiased_value
                unbiased_td_error = unbiased_target - q_unbiased[state_i][action_i]
                q_unbiased[state_i][action_i] += config.alpha * unbiased_td_error
                unbiased_td_sum += abs(unbiased_td_error)
                unbiased_td_count += 1
                unbiased_td_max = max(unbiased_td_max, abs(unbiased_td_error))
                positive_index = state_i * len(self.actions) + action_i
                if not unbiased_positive_mask[positive_index] and q_unbiased[state_i][action_i] > 0.0:
                    unbiased_positive_mask[positive_index] = 1
                    unbiased_positive_pairs += 1
                if uses_shaping:
                    next_biased_value = max(q_biased[next_i][action_index[candidate]] for candidate in next_actions)
                    biased_target = reward if terminal else reward + shaping_reward + self.gamma * next_biased_value
                    biased_td_error = biased_target - q_biased[state_i][action_i]
                    q_biased[state_i][action_i] += config.alpha * biased_td_error
                    biased_td_sum += abs(biased_td_error)
                    biased_td_count += 1
                    biased_td_max = max(biased_td_max, abs(biased_td_error))
                updates += 1
                state = next_state
                episode_succeeded = episode_succeeded or automaton_step.succeeded
                if terminal:
                    break

            learning_history["episodes"].append(episode)
            metric_unbiased_reward = float("nan") if started_terminal else unbiased_episode_reward
            learning_history["unbiased_episode_reward"].append(metric_unbiased_reward)
            learning_history["successes"].append(float("nan") if started_terminal else float(episode_succeeded))
            learning_history["episode_lengths"].append(float("nan") if started_terminal else episode_steps)
            learning_history["dfa_transitions"].append(float("nan") if started_terminal else episode_dfa_transitions)
            learning_history["initial_acceptances"].append(int(started_accepting))
            if uses_shaping:
                learning_history["biased_episode_reward"].append(float("nan") if started_terminal else biased_episode_reward)
            epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
            learning_history["epsilon"].append(epsilon)
            if episode == 1 or episode % config.log_interval == 0 or episode == config.episodes:
                window = min(config.log_interval, episode)
                recent_slice = slice(-window, None)
                recent_success_rate = finite_mean(learning_history["successes"][recent_slice])
                cumulative_success_rate = finite_mean(learning_history["successes"])
                recent_task_reward = finite_mean(learning_history["unbiased_episode_reward"][recent_slice])
                recent_learning_reward = finite_mean(learning_history["biased_episode_reward"][recent_slice]) if uses_shaping else recent_task_reward
                recent_shaping_reward = recent_learning_reward - recent_task_reward
                recent_episode_length = finite_mean(learning_history["episode_lengths"][recent_slice])
                recent_dfa_transitions = finite_mean(learning_history["dfa_transitions"][recent_slice])
                cumulative_initial_acceptances = sum(learning_history["initial_acceptances"])
                elapsed_seconds = time.monotonic() - start_time
                mean_unbiased_td = unbiased_td_sum / unbiased_td_count if unbiased_td_count else float("nan")
                biased_td_line = f"biased |TD| mean/max       : {biased_td_sum / biased_td_count:.4g}/{biased_td_max:.4g}\n" if uses_shaping and biased_td_count else ""
                behavior_label = "biased" if uses_shaping else "unbiased"
                log("\n" f"[Abstract {self.level_name} | Episode {episode}/{config.episodes} | last {window}]\n" f"success rate ({behavior_label})       : {format_percentage(recent_success_rate)} (cumulative {format_percentage(cumulative_success_rate)})\n" f"synthetic task reward       : {recent_task_reward:.3f}\n" f"shaping reward              : {recent_shaping_reward:.3f}\n" f"learning reward             : {recent_learning_reward:.3f}\n" f"episode length (non-goal)   : {recent_episode_length:.1f}\n" f"DFA transitions / episode   : {recent_dfa_transitions:.2f}\n" f"epsilon (next episode)      : {epsilon:.5f}\n" f"Q updates cumulative        : {updates}\n" f"unbiased |TD| mean/max     : {mean_unbiased_td:.4g}/{unbiased_td_max:.4g}\n" f"{biased_td_line}" f"unbiased positive Q pairs   : {unbiased_positive_pairs}/{total_valid_pairs} ({unbiased_positive_pairs / total_valid_pairs:.2%})\n" f"accepting-state restarts    : {cumulative_initial_acceptances}\n" f"elapsed                     : {elapsed_seconds:.1f}s")
                if episode % config.log_interval == 0 or episode == config.episodes:
                    unbiased_td_sum = 0.0
                    unbiased_td_count = 0
                    unbiased_td_max = 0.0
                    biased_td_sum = 0.0
                    biased_td_count = 0
                    biased_td_max = 0.0
            if episode == 1 or episode % config.eval_interval == 0 or episode == config.episodes:
                unbiased_eval_success, unbiased_eval_length, unbiased_eval_transitions, unbiased_eval_by_initial_q = evaluate_greedy(q_unbiased, evaluation_starts)
                unbiased_full_eval_success, unbiased_full_eval_length, unbiased_full_eval_transitions, _ = evaluate_greedy(q_unbiased, full_formula_evaluation_starts)
                learning_history["evaluation_steps"].append(episode)
                learning_history["unbiased_eval_success_rates"].append(unbiased_eval_success)
                learning_history["unbiased_eval_episode_lengths"].append(unbiased_eval_length)
                learning_history["unbiased_full_eval_success_rates"].append(unbiased_full_eval_success)
                learning_history["unbiased_full_eval_episode_lengths"].append(unbiased_full_eval_length)
                if uses_shaping:
                    biased_eval_success, biased_eval_length, biased_eval_transitions, biased_eval_by_initial_q = evaluate_greedy(q_biased, evaluation_starts)
                    biased_full_eval_success, biased_full_eval_length, biased_full_eval_transitions, _ = evaluate_greedy(q_biased, full_formula_evaluation_starts)
                    learning_history["biased_eval_success_rates"].append(biased_eval_success)
                    learning_history["biased_eval_episode_lengths"].append(biased_eval_length)
                    learning_history["biased_full_eval_success_rates"].append(biased_full_eval_success)
                    learning_history["biased_full_eval_episode_lengths"].append(biased_full_eval_length)
                    log("\n" f"[Abstract greedy evaluation at episode {episode} | {config.eval_episodes} fixed starts: random position, random recoverable non-accepting DFA state]\n" f"shaping-guided biased Q     : success={format_percentage(biased_eval_success)}, length={biased_eval_length:.1f}\n" f"{format_evaluation_by_initial_q('shaping-guided biased Q', biased_eval_by_initial_q)}\n" f"{format_evaluation_transitions('shaping-guided biased Q', biased_eval_transitions)}\n" f"original-reward unbiased Q  : success={format_percentage(unbiased_eval_success)}, length={unbiased_eval_length:.1f}\n" f"{format_evaluation_by_initial_q('original-reward unbiased Q', unbiased_eval_by_initial_q)}\n" f"{format_evaluation_transitions('original-reward unbiased Q', unbiased_eval_transitions)}\n\n" f"[Abstract greedy evaluation at episode {episode} | {config.eval_episodes} fixed starts: random position, starting from q{self.automaton.get_initial_q()}]\n" f"shaping-guided biased Q     : success={format_percentage(biased_full_eval_success)}, length={biased_full_eval_length:.1f}\n" f"{format_evaluation_transitions('shaping-guided biased Q', biased_full_eval_transitions)}\n" f"original-reward unbiased Q  : success={format_percentage(unbiased_full_eval_success)}, length={unbiased_full_eval_length:.1f}\n" f"{format_evaluation_transitions('original-reward unbiased Q', unbiased_full_eval_transitions)}")
                else:
                    log("\n" f"[Abstract greedy evaluation at episode {episode} | {config.eval_episodes} fixed starts: random position, random recoverable non-accepting DFA state]\n" f"original-reward unbiased Q  : success={format_percentage(unbiased_eval_success)}, length={unbiased_eval_length:.1f}\n" f"{format_evaluation_by_initial_q('original-reward unbiased Q', unbiased_eval_by_initial_q)}\n" f"{format_evaluation_transitions('original-reward unbiased Q', unbiased_eval_transitions)}\n\n" f"[Abstract greedy evaluation at episode {episode} | {config.eval_episodes} fixed starts: random position, starting from q{self.automaton.get_initial_q()}]\n" f"original-reward unbiased Q  : success={format_percentage(unbiased_full_eval_success)}, length={unbiased_full_eval_length:.1f}\n" f"{format_evaluation_transitions('original-reward unbiased Q', unbiased_full_eval_transitions)}")

        self.unbiased_q = q_unbiased
        if value_function_method == "policy_evaluation":
            self.unbiased_v_star = self.policy_evaluation(q_unbiased, theta=policy_evaluation_theta)
        else:
            self.unbiased_v_star = self._max_q_value_function(q_unbiased)
        self.biased_v_star = defaultdict(float, {state: max(q_biased[state_index[state]][action_index[action]] for action in self.get_available_actions(state)) for state in self.states}) if uses_shaping else None
        self.v_star = self.unbiased_v_star
        self.solution_algorithm = "learning"
        self.value_function_method = value_function_method
        self.learning_episodes = config.episodes
        self.learning_updates = updates
        self.learning_history = learning_history
        policy_evaluation_summary = f" | policy-evaluation iterations={self.policy_evaluation_iterations}" if value_function_method == "policy_evaluation" else ""
        log(f"Completed [{self.level_name}] | episodes={config.episodes} | updates={updates} | V={value_function_method}{policy_evaluation_summary} | elapsed={time.monotonic() - start_time:.1f}s")
        if log_handle is not None:
            log_handle.close()
        if print_policy:
            self.print_policy()
        return self.v_star


class MultiLevelWaypointMDP:
    """Ordered hierarchy of grid MDPs sharing one temporal-task automaton.

    The coarsest level can use VI or classic Q-learning. Every lower
    abstraction is learned with biased exploration and exports its unbiased
    value estimate.
    """
    def __init__(self, regions, ltlf_automaton, abstraction_config, gamma=0.99, goal_reward=10000):
        self.automaton = ltlf_automaton
        self.abstraction_config = abstraction_config
        self.gamma = gamma
        self.goal_reward = goal_reward
        self.levels = []

        for level in abstraction_config.levels:
            level_mdp = LTLfWaypointMDP(regions=regions, ltlf_automaton=ltlf_automaton, width=level.width, height=level.height, gamma=gamma, goal_reward=goal_reward, level_name=level.name)
            self._warn_on_region_collisions(level_mdp)
            self.levels.append(level_mdp)

    @staticmethod
    def _warn_on_region_collisions(level):
        cells_to_names = defaultdict(list)
        for name, cells in level.region_cells.items():
            if not isinstance(level.regions[name], CircularRegion):
                continue
            for cell in cells:
                cells_to_names[cell].append(name)
        collisions = {
            cell: names
            for cell, names in cells_to_names.items()
            if len(names) > 1
        }
        if collisions:
            warnings.warn(f"{level.level_name} maps multiple propositions to the same cells: {collisions}. They will be true simultaneously on that level.", UserWarning, stacklevel=3)

    @property
    def primary_mdp(self):
        """Return level 1, used unchanged by automaton handling and training."""
        return self.levels[0]

    def compute_value_functions(self, theta=0.001, print_policies=False, learning_log_dir=None):
        """Solve the configurable top, then learn every lower abstraction."""
        following_mdp = None
        for index in reversed(range(len(self.levels))):
            level_config = self.abstraction_config.levels[index]
            current_mdp = self.levels[index]
            if level_config.checkpoint is not None:
                current_mdp.load_checkpoint(level_config.checkpoint, upper_level_mdp=following_mdp, print_policy=print_policies)
            elif self.abstraction_config.algorithm_for_index(index) == "value_iteration":
                current_mdp.value_iteration(theta=theta, print_policy=print_policies)
            else:
                log_file = os.path.join(learning_log_dir, f"level{index + 1}.log") if learning_log_dir is not None else None
                current_mdp.q_learning(config=level_config.learning, upper_level_mdp=following_mdp, print_policy=print_policies, log_file=log_file, value_function_method=level_config.value_function_method, policy_evaluation_theta=theta)
            following_mdp = current_mdp
        return [level.v_star for level in self.levels]
