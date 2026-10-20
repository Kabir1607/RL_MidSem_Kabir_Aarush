"""SB3 callbacks (spec Sections 6.1-6.2): thin wrappers that feed a RunRecorder."""

from stable_baselines3.common.callbacks import BaseCallback

import config


def masked_greedy(model):
    """Deterministic policy(obs, mask) -> action for a MaskablePPO model."""
    return lambda obs, mask: int(model.predict(obs, action_masks=mask, deterministic=True)[0])


class TrainEpisodeLogger(BaseCallback):
    """Appends one row to train_episodes.csv at the end of every training battle."""

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def _on_step(self):
        for info in self.locals["infos"]:
            if "battle_end" in info:
                self.recorder.episode_end(self.num_timesteps, info["battle_end"])
        return True


class EvalCurveCallback(BaseCallback):
    """Evaluates at step 0 and every eval_freq steps, and stops training at exactly total_steps."""

    def __init__(self, recorder, total_steps, eval_freq=config.EVAL_FREQ, verbose=0):
        super().__init__(verbose)
        if total_steps % eval_freq:
            raise ValueError(f"total_steps {total_steps} must be a multiple of eval_freq {eval_freq}")
        self.recorder = recorder
        self.total_steps = total_steps
        self.eval_freq = eval_freq

    def _on_training_start(self):
        self._evaluate()

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return self.num_timesteps < self.total_steps

    def _evaluate(self):
        res = self.recorder.evaluate(self.num_timesteps, masked_greedy(self.model), self.model.save)
        for key in ("win_rate", "wilson_low", "wilson_high"):
            self.logger.record(f"eval/{key}", res[key])
        self.logger.record("eval/auc_so_far", res["auc"])
        if self.verbose:
            print(f"[eval] step {self.num_timesteps}: win rate {res['win_rate']:.3f} "
                  f"[{res['wilson_low']:.3f}, {res['wilson_high']:.3f}], AUC {res['auc']:.3f}", flush=True)
