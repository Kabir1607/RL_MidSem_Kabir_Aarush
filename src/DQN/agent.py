"""Readable one-step DQN: legal exploration, legal targets, and a target network."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from torch.nn import functional as F

import config
from DQN.model import QNetwork
from DQN.replay import ReplayBatch


@dataclass(frozen=True)
class DQNSettings:
    """The agreed defaults; the smoke profile changes only its documented settings."""

    profile: str = "main"
    total_steps: int = config.TOTAL_STEPS
    opponent: str = "heuristic"
    suffix: str = ""
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    batch_size: int = 32
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    exploration_steps: int = 20_000
    gamma: float = config.GAMMA
    max_grad_norm: float = 10.0
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    huber_delta: float = 1.0

    def hyperparameters(self):
        """Run-wide settings only: team and seed must not differ in this dictionary."""
        return {**asdict(self), "net_arch": list(config.NET_ARCH), "optimizer": "Adam",
                "loss": "Huber", "algorithm": "vanilla_dqn"}

    @classmethod
    def for_profile(cls, profile):
        if profile == "main":
            return cls()
        if profile == "smoke":
            return replace(cls(), profile="smoke", total_steps=20_000, opponent="maxbp",
                           suffix="smoke", learning_starts=1_000, exploration_steps=10_000)
        raise ValueError(f"Unknown DQN profile: {profile}")

    def epsilon(self, decisions_collected):
        """The first action uses epsilon(0); decay is in decisions, not episodes."""
        if decisions_collected < 0:
            raise ValueError("Decision count cannot be negative")
        fraction = min(decisions_collected / self.exploration_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def should_update(self, step, replay_size):
        return (step > self.learning_starts and step % self.train_freq == 0
                and replay_size >= self.batch_size)


def greedy_action(q_values, mask):
    """Return the best legal index; NumPy argmax breaks ties by the first index."""
    q_values = np.asarray(q_values)
    mask = np.asarray(mask, dtype=bool)
    if q_values.ndim != 1 or mask.shape != q_values.shape:
        raise ValueError("Q-values and action mask must be matching 1-D arrays")
    if not mask.any():
        raise ValueError("Cannot act with an empty legal-action mask")
    if not np.isfinite(q_values).all():
        raise FloatingPointError("Non-finite Q-values")
    return int(np.where(mask, q_values, -np.inf).argmax())


@torch.no_grad()
def masked_td_targets(rewards, next_q, next_masks, terminated, gamma):
    """Reward plus legal next-state value, with zero bootstrap on true termination.

    Terminal masks may be all false. Compute maxima only on nonterminal rows,
    avoiding the undefined expression -inf * 0 in mixed minibatches.
    Truncated transitions arrive with terminated=False and still bootstrap.
    """
    if (next_q.ndim != 2 or next_masks.shape != next_q.shape
            or rewards.shape != (len(next_q),) or terminated.shape != rewards.shape):
        raise ValueError("TD target tensors have incompatible shapes")
    continuing = ~terminated
    bootstrap = torch.zeros_like(rewards)
    if continuing.any():
        masks = next_masks[continuing]
        if not masks.any(dim=1).all():
            raise ValueError("A nonterminal TD target needs a legal next action")
        legal_q = next_q[continuing].masked_fill(~masks, -torch.inf)
        bootstrap[continuing] = legal_q.max(dim=1).values
    targets = rewards + gamma * bootstrap
    if not torch.isfinite(targets).all():
        raise FloatingPointError("Non-finite DQN targets")
    return targets


class DQNAgent:
    """Owns the online network, frozen target, optimizer, and exploration RNG."""

    def __init__(self, obs_size, action_size, settings, exploration_rng):
        self.settings = settings
        self.exploration_rng = exploration_rng
        self.device = torch.device(config.DEVICE)
        self.online = QNetwork(obs_size, action_size, config.NET_ARCH).to(self.device)
        self.target = deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=settings.learning_rate, betas=settings.adam_betas,
            eps=settings.adam_eps, weight_decay=settings.weight_decay,
        )
        self.update_count = 0

    @torch.inference_mode()
    def q_values(self, obs):
        if np.shape(obs) != (self.online.obs_size,):
            raise ValueError("Observation has the wrong shape")
        values = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(values)[0].cpu().numpy()

    def greedy(self, obs, mask):
        """Deterministic policy used by the shared evaluation recorder."""
        return greedy_action(self.q_values(obs), mask)

    def act(self, obs, mask, epsilon):
        q_values = self.q_values(obs)
        best = greedy_action(q_values, mask)  # also validates the mask before exploring
        if self.exploration_rng.random() < epsilon:
            return int(self.exploration_rng.choice(np.flatnonzero(mask)))
        return best

    def update(self, batch: ReplayBatch):
        """One gradient update. Replay data and target predictions stay detached."""
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch.next_masks, dtype=torch.bool, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            targets = masked_td_targets(rewards, self.target(next_obs), next_masks,
                                        terminated, self.settings.gamma)
        chosen_q = self.online(obs).gather(1, actions[:, None]).squeeze(1)
        loss = F.huber_loss(chosen_q, targets, delta=self.settings.huber_delta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss")
        self.optimizer.zero_grad()
        loss.backward()
        # Fail on divergence rather than quietly writing a broken checkpoint.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.settings.max_grad_norm, error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.update_count += 1
        return {"train/loss": loss.item(), "train/q_mean": chosen_q.detach().mean().item(),
                "train/target_mean": targets.mean().item(), "train/grad_norm": grad_norm.item()}

    def sync_target(self):
        """Hard copy; called on a schedule measured in environment decisions."""
        self.target.load_state_dict(self.online.state_dict())
