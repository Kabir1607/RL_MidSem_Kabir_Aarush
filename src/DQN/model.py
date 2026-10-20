"""The Q-network: one expected-return estimate for every action index."""

import torch
from torch import nn


class QNetwork(nn.Module):
    """A small MLP with raw Q-values, rather than action probabilities, as outputs.

    Legality is handled outside the network because it changes at every decision.
    For the current experiment the dimensions are 763 -> 256 -> 256 -> 26.
    """

    def __init__(self, obs_size, action_size, hidden_sizes):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_sizes = tuple(hidden_sizes)
        layers = []
        previous_size = obs_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(previous_size, hidden_size), nn.ReLU()])
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, action_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map a batch shaped (batch_size, obs_size) to (batch_size, action_size)."""
        return self.layers(observations)
