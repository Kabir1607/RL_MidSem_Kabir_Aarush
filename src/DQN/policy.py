"""Inference checkpoints shared by the trainer and future move-usage evaluation.

    from DQN.policy import load_policy
    policy = load_policy(Path('runs/dqn_lowvar_s0/ckpt/step_200000.pt'))
    action = policy(observation, legal_action_mask)

Loading needs the project and its dependencies, but no server or training objects.
Checkpoints intentionally do not promise resumable training: replay is not saved.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch

import config
from DQN.agent import greedy_action
from DQN.model import QNetwork
from encoder import OBS_SIZE

FORMAT_VERSION = 1


def save_checkpoint(path, network, step, team, seed, settings):
    """Atomically save the exact online weights used for the current evaluation."""
    path = Path(path).with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Gymnasium's Discrete.n is a NumPy integer. Restricted torch loading needs
    # primitive Python metadata, even though NumPy integers work in nn.Linear.
    checkpoint = dict(
        format_version=FORMAT_VERSION, algo="dqn", step=int(step), team=team, seed=int(seed),
        profile=settings.profile, obs_size=int(network.obs_size), action_size=int(network.action_size),
        net_arch=[int(size) for size in network.hidden_sizes], gamma=float(settings.gamma),
        hyperparameters=settings.hyperparameters(),
        q_net={name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()},
    )
    temporary = path.with_suffix(".pt.tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_policy(checkpoint_path: Path) -> Callable[[np.ndarray, np.ndarray], int]:
    """Load a CPU network and return deterministic, legal greedy inference.

    Metadata is checked against the current encoder/config to catch incompatible
    models before playing a battle. torch's restricted weights loader accepts our
    primitive metadata and tensor state dictionary without arbitrary Python objects.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = dict(format_version=FORMAT_VERSION, algo="dqn", obs_size=OBS_SIZE,
                    action_size=config.ACTION_SPACE_SIZE, net_arch=list(config.NET_ARCH),
                    gamma=config.GAMMA)
    if not isinstance(checkpoint, dict):
        raise ValueError("DQN checkpoint must be a dictionary")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible DQN checkpoint: {key} must be {value!r}")
    if not isinstance(checkpoint.get("q_net"), dict):
        raise ValueError("DQN checkpoint is missing its q_net state dictionary")
    model = QNetwork(OBS_SIZE, config.ACTION_SPACE_SIZE, config.NET_ARCH)
    model.load_state_dict(checkpoint["q_net"], strict=True)
    model.eval()

    @torch.inference_mode()
    def policy(obs, mask):
        if np.shape(obs) != (OBS_SIZE,):
            raise ValueError(f"Expected observation shape {(OBS_SIZE,)}, got {np.shape(obs)}")
        tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return greedy_action(model(tensor)[0].numpy(), mask)

    return policy
