import random
import re
import warnings
from collections import defaultdict

from abstraction import map_state
from spatial_regions import (
    rasterize_regions,
    truth_assignment_from_cell,
    truth_assignment_from_observation,
)

class LTLfAutomaton:
    """
    Wrap ltlf2dfa and expose its DFA as a graph that can be traversed by the MDP.
    """
    def __init__(self, formula_str):
        # Keep MDP/mapping utilities importable without the optional DFA parser.
        from ltlf2dfa.parser.ltlf import LTLfParser

        self.formula_str = formula_str
        
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


class LTLfWaypointMDP:
    """
    Abstract MDP guided by an LTLf automaton.
    Each abstract state is (x, y, q), where q is the DFA state identifier.
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
        self.actions = self.movement_actions + [self.DONE_ACTION]
        
        self.regions = regions
        self.region_cells = rasterize_regions(regions, width, height)
        self.automaton = ltlf_automaton
        self.num_phases = self.automaton.num_phases
        
        # Generate every combination of grid position and DFA state.
        self.states = [(x, y, q) for x in range(width) for y in range(height) for q in self.automaton.states]
        
        self.goal_reward = goal_reward
        self.v_star = defaultdict(float)
        self.upper_level_mdp = None
        self.value_iteration_iterations = 0
        self.solution_algorithm = None
        self.learning_episodes = 0
        self.learning_updates = 0
        self.learning_history = None
        
    def _get_truth_assignment(self, x, y):
        """
        Map the current grid coordinates to a Boolean proposition assignment.
        """
        return truth_assignment_from_cell(self.region_cells, x, y)

    def get_environment_truth_assignment(self, observation):
        """Evaluate propositions exactly on a continuous environment state."""
        return truth_assignment_from_observation(self.regions, observation)

    def get_available_actions(self, state):
        """Return only done at the goal, and only movements elsewhere."""
        return [self.DONE_ACTION] if self.automaton.is_goal_reached(state[2]) else self.movement_actions

    def get_transitions(self, state, action):
        x, y, q = state
        available_actions = self.get_available_actions(state)
        if action not in available_actions:
            raise ValueError(f"Action {action} is not available in state {state}; expected one of {available_actions}")
        if action == self.DONE_ACTION:
            return state, self.goal_reward, True
        
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
        next_q = self.automaton.get_next_q(q, truth_assignment)

        return (next_x, next_y, next_q), 0.0, False

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
        return self.gamma * next_state_potential - state_potential

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

                    best_action = None
                    best_value = -float("inf")

                    for a in self.get_available_actions(state):
                        next_state, reward, terminal = self.get_transitions(state, a)
                        shaping_reward = 0.0 if terminal else self.get_inter_level_shaping_reward(state, next_state)
                        value = reward if terminal else reward + shaping_reward + self.gamma * self.v_star[next_state]

                        if value > best_value:
                            best_value = value
                            best_action = a

                    row.append(f" {arrows[best_action]} ")

                print("".join(row))
    
    def value_iteration(self, theta=0.001, print_policy=True):
        """Compute the unbiased V* of the unique top abstraction."""
        if theta <= 0:
            raise ValueError("theta must be greater than zero")

        self.upper_level_mdp = None
        self.v_star = defaultdict(float)
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
                    next_state, reward, terminal = self.get_transitions(s, action)
                    v_actions.append(reward if terminal else reward + self.gamma * self.v_star[next_state])
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - self.v_star[s]))
                new_v[s] = best_v
            self.v_star = new_v
            if delta < theta:
                break

        self.value_iteration_iterations = iterations
        self.solution_algorithm = "vi"
        self.learning_episodes = 0
        self.learning_updates = 0
        if print_policy:
            self.print_policy()
        return self.v_star

    def q_learning(self, config, upper_level_mdp=None, print_policy=True):
        """Learn an unbiased value estimate.

        Two tabular learners consume every sampled transition.  The biased
        learner receives inter-level PBRS and supplies the epsilon-greedy
        behaviour policy; the unbiased learner receives the original reward
        and supplies the value function exported to the next lower level.
        """
        if upper_level_mdp is None:
            raise ValueError("Abstract Q-learning requires an already solved upper level")
        self.upper_level_mdp = upper_level_mdp
        self.value_iteration_iterations = 0
        rng = random.Random(config.seed)
        state_index = {state: index for index, state in enumerate(self.states)}
        action_index = {action: index for index, action in enumerate(self.actions)}
        q_biased = [[0.0 for _ in self.actions] for _ in self.states]
        q_unbiased = [[0.0 for _ in self.actions] for _ in self.states]
        restart_states = self.states

        epsilon = config.epsilon_start
        updates = 0
        learning_history = {"episodes": [], "epsilon": [], "biased_episode_reward": [], "unbiased_episode_reward": []}

        print(f"Q-learning [{self.level_name}: {self.width}x{self.height}, dual-table inter-level PBRS, episodes={config.episodes}]...")

        for episode in range(1, config.episodes + 1):
            # Random restarts cover the full product space. Accepting states
            # must also be sampled so their only action, done, is learned.
            state = rng.choice(restart_states)
            biased_episode_reward = 0.0
            unbiased_episode_reward = 0.0
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

                next_state, reward, terminal = self.get_transitions(state, action)
                shaping_reward = 0.0 if terminal else self.get_inter_level_shaping_reward(state, next_state)
                biased_episode_reward += reward + shaping_reward
                unbiased_episode_reward += reward
                state_i = state_index[state]
                next_i = state_index[next_state]
                action_i = action_index[action]

                next_actions = self.get_available_actions(next_state)
                next_biased_value = max(q_biased[next_i][action_index[candidate]] for candidate in next_actions)
                next_unbiased_value = max(q_unbiased[next_i][action_index[candidate]] for candidate in next_actions)
                biased_target = reward if terminal else reward + shaping_reward + self.gamma * next_biased_value
                unbiased_target = reward if terminal else reward + self.gamma * next_unbiased_value
                q_biased[state_i][action_i] += config.alpha * (biased_target - q_biased[state_i][action_i])
                q_unbiased[state_i][action_i] += config.alpha * (unbiased_target - q_unbiased[state_i][action_i])
                updates += 1
                state = next_state
                if terminal:
                    break

            learning_history["episodes"].append(episode)
            learning_history["epsilon"].append(epsilon)
            learning_history["biased_episode_reward"].append(biased_episode_reward)
            learning_history["unbiased_episode_reward"].append(unbiased_episode_reward)
            epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        self.v_star = defaultdict(float, {state: max(q_unbiased[state_index[state]][action_index[action]] for action in self.get_available_actions(state)) for state in self.states})
        self.solution_algorithm = "learning"
        self.learning_episodes = config.episodes
        self.learning_updates = updates
        self.learning_history = learning_history
        if print_policy:
            self.print_policy()
        return self.v_star


class MultiLevelWaypointMDP:
    """Ordered hierarchy of grid MDPs sharing one LTLf automaton.

    Waypoints and automaton interaction are defined on level 1.  Waypoint
    coordinates for every other grid are projections of those coordinates.
    The coarsest level is solved by value iteration. Every lower abstraction
    is learned with biased exploration and exports its unbiased value estimate.
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

    def compute_value_functions(self, theta=0.001, print_policies=False):
        """Solve the top with VI, then learn every lower abstraction."""
        following_mdp = None
        for index in reversed(range(len(self.levels))):
            level_config = self.abstraction_config.levels[index]
            current_mdp = self.levels[index]
            if self.abstraction_config.algorithm_for_index(index) == "vi":
                current_mdp.value_iteration(theta=theta, print_policy=print_policies)
            else:
                current_mdp.q_learning(config=level_config.learning, upper_level_mdp=following_mdp, print_policy=print_policies)
            following_mdp = current_mdp
        return [level.v_star for level in self.levels]
