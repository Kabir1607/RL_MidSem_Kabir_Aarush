"""Uniform experience replay, with explicit copies at the environment boundary."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    """Arrays sampled together, so every row describes the same transition."""

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_masks: np.ndarray
    terminated: np.ndarray


class ReplayBuffer:
    """Fixed-capacity ring buffer; old transitions are overwritten when full.

    We keep the next observation explicitly. Inferring it from the following row
    would be wrong at episode boundaries and when a time limit resets the env.
    Only actual termination prevents bootstrapping; truncation is not stored as done.
    """

    def __init__(self, capacity, obs_size, action_size, rng):
        if min(capacity, obs_size, action_size) <= 0:
            raise ValueError("Replay dimensions and capacity must be positive")
        self.capacity = capacity
        self.obs_size = obs_size
        self.action_size = action_size
        self.rng = rng
        self.position = 0
        self.size = 0
        self.obs = np.empty((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_masks = np.empty((capacity, action_size), dtype=bool)
        self.terminated = np.empty(capacity, dtype=bool)

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, next_mask, terminated):
        """Copy one transition before the caller resets or reuses its arrays."""
        if np.shape(obs) != (self.obs_size,) or np.shape(next_obs) != (self.obs_size,):
            raise ValueError("Replay observations have the wrong shape")
        if np.shape(next_mask) != (self.action_size,):
            raise ValueError("Replay next-action mask has the wrong shape")
        if not 0 <= action < self.action_size:
            raise ValueError("Replay action is outside the action space")
        if not terminated and not np.any(next_mask):
            raise ValueError("A nonterminal transition needs a legal next action")
        if not np.isfinite(reward):
            raise ValueError("Replay reward must be finite")

        index = self.position
        self.obs[index] = obs
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_obs[index] = next_obs
        self.next_masks[index] = next_mask
        self.terminated[index] = terminated
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size) -> ReplayBatch:
        """Sample distinct populated rows using this buffer's own seeded RNG."""
        if not 0 < batch_size <= self.size:
            raise ValueError("Not enough replay transitions for this batch size")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return ReplayBatch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            next_masks=self.next_masks[indices],
            terminated=self.terminated[indices],
        )
