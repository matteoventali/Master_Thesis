import numpy as np
import random as ran
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        self.feature = nn.Sequential( nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_dim)

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + advantages - advantages.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity, num_phases):
        if num_phases < 0:
            raise ValueError("num_phases cannot be negative")
        self.capacity = capacity
        self.num_phases = num_phases
        self.buffer = []
        self.phase_indices = []
        self.phase_counts = np.zeros(num_phases, dtype=np.int64)
        self.position = 0

    def push(self, state, action, reward, next_state, done, task_reward=None):
        """Insert a transition and update DFA-state counts in constant time."""
        task_reward = reward if task_reward is None else task_reward
        transition = (state, action, reward, task_reward, next_state, done)
        phase_index = (
            int(np.argmax(state[-self.num_phases:]))
            if self.num_phases > 0
            else None
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.phase_indices.append(phase_index)
        else:
            replaced_phase_index = self.phase_indices[self.position]
            if replaced_phase_index is not None:
                self.phase_counts[replaced_phase_index] -= 1
            self.buffer[self.position] = transition
            self.phase_indices[self.position] = phase_index

        if phase_index is not None:
            self.phase_counts[phase_index] += 1
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample transitions efficiently from the indexable ring buffer."""
        batch = ran.sample(self.buffer, batch_size)
        state, action, reward, _task_reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def sample_dual(self, batch_size):
        """Sample one batch containing shaped and original task rewards."""
        batch = ran.sample(self.buffer, batch_size)
        return tuple(map(np.array, zip(*batch)))

    def __len__(self):
        return len(self.buffer)

    def q_fraction_onehot(self, q_index, num_phases):
        """Return a DFA-state fraction using incrementally maintained counts."""
        if num_phases != self.num_phases:
            raise ValueError(f"Expected {self.num_phases} DFA states, received {num_phases}")
        if not 0 <= q_index < self.num_phases:
            raise IndexError(f"DFA state index {q_index} is out of range")
        if len(self.buffer) == 0:
            return 0.0
        return float(self.phase_counts[q_index] / len(self.buffer))

class HierarchicalDQNLearner:
    def __init__( self, env, abstract_mdp=None, max_episodes=1000, eps_decay=0.995, gamma=0.99, policy_name="policy", extra_state_dims=0, use_polyak=True, tau=0.005, target_update_freq=1000, network_type="standard", policy_dir="policy", stochastic_bellman_update=False, bellman_alpha=0.1, ):
        if extra_state_dims < 0:
            raise ValueError("extra_state_dims cannot be negative")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in the interval (0, 1]")
        if target_update_freq <= 0:
            raise ValueError("target_update_freq must be greater than zero")
        if network_type not in {"standard", "dueling"}:
            raise ValueError("network_type must be one of: standard, dueling")
        if not 0.0 < bellman_alpha <= 1.0:
            raise ValueError("bellman_alpha must be in the interval (0, 1]")

        self.env = env
        self.abstract_mdp = abstract_mdp
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.policy_name = policy_name
        self.policy_dir = os.fspath(policy_dir)
        self.network_type = network_type
        self.algo_name = "Dueling DDQN" if network_type == "dueling" else "DDQN"
        self.stochastic_bellman_update = stochastic_bellman_update
        self.bellman_alpha = bellman_alpha
        
        self.batch_size = 64
        self.lr = 1e-3
        self.use_polyak = use_polyak
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.optimization_steps = 0
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = eps_decay
        
        # Account for dynamic one-hot phases appended to state
        state_dim = self.env.observation_space.shape[0] + extra_state_dims
        action_dim = self.env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network_cls = DuelingQNetwork if network_type == "dueling" else QNetwork
        self.policy_net = network_cls(state_dim, action_dim).to(self.device)
        print(f"Using device:{self.device}")
        
        self.target_net = network_cls(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=300000, num_phases=extra_state_dims)

    def select_action(self, state):
        if ran.random() < self.eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def optimize_model(self, batch=None):
        if batch is None:
            if len(self.memory) < self.batch_size:
                return
            batch = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, best_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            if self.stochastic_bellman_update:
                # Stochastic-approximation form of the Bellman update:
                # Q <- Q + alpha * (target_DDQN - Q).  q_values is detached
                # because the right-hand side is the regression target.
                target_q_values = q_values.detach() + self.bellman_alpha * (
                    target_q_values - q_values.detach()
                )
            
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.optimization_steps += 1
        if self.use_polyak:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        elif self.optimization_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _save_policy(self):
        os.makedirs(self.policy_dir, exist_ok=True)
        torch.save( self.policy_net.state_dict(), os.path.join(self.policy_dir, self.policy_name), )

class TabularQLearner:
    """Sparse online Q-learning over discretized LunarLander and DFA states."""

    PHYSICAL_BINS = ((-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8), (0.2, 0.5, 0.8, 1.0, 1.2, 1.5), (-0.3, -0.1, 0.1, 0.3), (-0.3, -0.1, 0.1, 0.3), (-0.2, 0.0, 0.2), (-0.2, 0.0, 0.2))

    def __init__(self, env, num_phases, eps_decay=0.995, gamma=0.99, alpha=0.1, policy_name="policy", policy_dir="policy", random_seed=None):
        if num_phases <= 0:
            raise ValueError("num_phases must be greater than zero")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")
        self.env = env
        self.num_phases = num_phases
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = eps_decay
        self.policy_name = policy_name
        self.policy_dir = os.fspath(policy_dir)
        self.action_dim = env.action_space.n
        self.algo_name = "Tabular Q-learning"
        self.random_rng = ran.Random(random_seed)
        self.q_table = {}
        self.visited_states = set()
        self.updated_state_actions = set()
        self.total_updates = 0
        self.positive_updates = 0
        self.memory = ReplayBuffer(capacity=300000, num_phases=num_phases)
        self.reset_diagnostics()

    def state_key(self, augmented_state):
        state = np.asarray(augmented_state, dtype=np.float64)
        expected_size = 8 + self.num_phases
        if state.size != expected_size:
            raise ValueError(f"Expected augmented state size {expected_size}, received {state.size}")
        physical = tuple(int(np.digitize(state[index], bins)) for index, bins in enumerate(self.PHYSICAL_BINS))
        contacts = (int(state[6] >= 0.5), int(state[7] >= 0.5))
        phase_index = int(np.argmax(state[-self.num_phases:]))
        return physical + contacts + (phase_index,)

    def _values(self, key, create=True):
        values = self.q_table.get(key)
        if values is None and create:
            values = np.zeros(self.action_dim, dtype=np.float64)
            self.q_table[key] = values
        return values

    def greedy_action(self, state, return_known=False):
        key = self.state_key(state)
        values = self._values(key, create=False)
        known = values is not None
        candidates = list(range(self.action_dim)) if values is None else np.flatnonzero(np.isclose(values, np.max(values))).tolist()
        action = self.random_rng.choice(candidates)
        return (action, known) if return_known else action

    def select_action(self, state):
        return self.random_rng.randrange(self.action_dim) if self.random_rng.random() < self.eps else self.greedy_action(state)

    def update(self, state, action, reward, next_state, terminal):
        state_key = self.state_key(state)
        next_key = self.state_key(next_state)
        self.visited_states.update((state_key, next_key))
        self.updated_state_actions.add((state_key, int(action)))
        values = self._values(state_key)
        next_values = self._values(next_key)
        target = float(reward) if terminal else float(reward) + self.gamma * float(np.max(next_values))
        td_error = target - values[action]
        values[action] += self.alpha * td_error
        self.total_updates += 1
        self.positive_updates += int(reward > 0)
        self._diagnostics["updates"] += 1
        self._diagnostics["positive_updates"] += int(reward > 0)
        self._diagnostics["abs_td_error_sum"] += abs(td_error)
        self._diagnostics["max_abs_td_error"] = max(self._diagnostics["max_abs_td_error"], abs(td_error))

    def metrics_snapshot(self):
        possible_pairs = len(self.visited_states) * self.action_dim
        return {"table_size": len(self.q_table), "visited_states": len(self.visited_states), "updated_state_actions": len(self.updated_state_actions), "state_action_coverage": len(self.updated_state_actions) / possible_pairs if possible_pairs else 0.0, "positive_updates": self.positive_updates}

    def reset_diagnostics(self):
        self._diagnostics = {"updates": 0, "positive_updates": 0, "abs_td_error_sum": 0.0, "max_abs_td_error": 0.0}

    def consume_diagnostics(self):
        diagnostics = self._diagnostics
        updates = diagnostics["updates"]
        result = {"updates": updates, "positive_update_fraction": diagnostics["positive_updates"] / updates if updates else 0.0, "mean_abs_td_error": diagnostics["abs_td_error_sum"] / updates if updates else 0.0, "max_abs_td_error": diagnostics["max_abs_td_error"]}
        self.reset_diagnostics()
        return result

    def _save_policy(self):
        destination = os.path.join(self.policy_dir, self.policy_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        serializable_table = {key: values.tolist() for key, values in self.q_table.items()}
        with open(destination, "wb") as policy_file:
            pickle.dump({"q_table": serializable_table, "num_phases": self.num_phases, "gamma": self.gamma, "alpha": self.alpha, "physical_bins": self.PHYSICAL_BINS}, policy_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_policy(self, policy_path=None):
        source = os.fspath(policy_path) if policy_path is not None else os.path.join(self.policy_dir, self.policy_name)
        with open(source, "rb") as policy_file:
            checkpoint = pickle.load(policy_file)
        if checkpoint.get("num_phases") != self.num_phases:
            raise ValueError("Tabular checkpoint DFA-state count does not match the learner")
        saved_bins = tuple(tuple(boundaries) for boundaries in checkpoint.get("physical_bins", ()))
        if saved_bins != self.PHYSICAL_BINS:
            raise ValueError("Tabular checkpoint discretization does not match the learner")
        self.q_table = {tuple(key): np.asarray(values, dtype=np.float64) for key, values in checkpoint["q_table"].items()}
        self.gamma = float(checkpoint["gamma"])
        self.alpha = float(checkpoint["alpha"])
