"""Assign query-aligned shaped rewards before a Dobot rollout enters replay."""

from pathlib import Path

import numpy as np


class DobotShapedReward:
    def __init__(
        self,
        *,
        reward_type,
        checkpoint_path,
        gaussian_model_h5_path,
        calibration_h5_path,
        context_stride,
        query_freq,
        discount,
        device="cpu",
        ood_p_value_threshold=0.05,
        posterior_temperature=1.0e4,
        shaping_scale=1.0,
    ):
        if reward_type not in ("dense", "pbrs"):
            raise ValueError("Shaped reward type must be 'dense' or 'pbrs'")
        if context_stride is None or context_stride < 1 or query_freq < 1:
            raise ValueError("context_stride and query_freq must be positive integers")
        if context_stride % query_freq:
            raise ValueError(
                "Progress context_stride must be an integer multiple of query_freq "
                "so model context uses only pi0 query frames"
            )
        if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")
        if not np.isfinite(shaping_scale):
            raise ValueError("shaping_scale must be finite")

        paths = {}
        for name, value in (
            ("checkpoint_path", checkpoint_path),
            ("gaussian_model_h5_path", gaussian_model_h5_path),
            ("calibration_h5_path", calibration_h5_path),
        ):
            if not value:
                raise ValueError(f"{name} is required for shaped reward")
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{name} not found: {path}")
            paths[name] = str(path)

        self.reward_type = reward_type
        self.query_freq = int(query_freq)
        self.discount = float(discount)
        self.shaping_scale = float(shaping_scale)
        from .gaussian_progress import BatchedGaussianProgressGatedProvider

        self.provider = BatchedGaussianProgressGatedProvider(
            1,
            **paths,
            device=device,
            ood_p_value_threshold=ood_p_value_threshold,
            posterior_temperature=posterior_temperature,
            frame_history_stride=context_stride // query_freq,
        )

    def apply(self, traj, preprocess_frame):
        """Replace rollout rewards; no progress inference uses the final observation."""
        count = len(traj["actions"])
        frames = traj["reward_frames"]
        lengths = np.asarray(traj["query_step_lengths"])
        masks = np.asarray(traj["masks"], dtype=np.float64)
        if count < 1 or len(frames) != count or len(traj["observations"]) != count + 1:
            raise ValueError("Rollout needs one reward frame per SAC action and one final observation")
        if lengths.shape != (count,) or not np.issubdtype(lengths.dtype, np.integer):
            raise ValueError("query_step_lengths must contain one integer per SAC action")
        if np.any(lengths < 1) or np.any(lengths > self.query_freq) or lengths.sum() != traj["env_steps"]:
            raise ValueError("query_step_lengths do not match the executed robot steps")
        if masks.shape != (count,) or not np.isin(masks, (0.0, 1.0)).all():
            raise ValueError("masks must contain one binary value per SAC action")
        if not callable(preprocess_frame):
            raise TypeError("Dobot wrapper must provide prepare_progress_frame()")

        progress = np.empty(count, dtype=np.float64)
        is_ood = np.empty(count, dtype=bool)
        conformal_p_values = []
        for index, frame in enumerate(frames):
            prepared = preprocess_frame(frame)
            if index == 0:
                inferred, diagnostics = self.provider.reset_all(
                    [prepared], return_diagnostics=True)
            else:
                inferred, diagnostics = self.provider.advance_all(
                    [prepared], return_diagnostics=True)
            progress[index] = float(inferred[0])
            is_ood[index] = bool(diagnostics["is_ood"][0])
            conformal_p_values.append(diagnostics["conformal_p_value"][0])
        if not np.isfinite(progress).all():
            raise ValueError("Inferred progress contains NaN or Inf")

        sparse = np.zeros(count, dtype=np.float64)
        if traj["is_success"]:
            sparse[-1] = 1.0
        if self.reward_type == "dense":
            shaping = self.shaping_scale * progress
        else:
            # The final observation can be off the model's temporal grid. Reuse
            # the last query progress while retaining the replay discount/mask.
            progress_next = np.concatenate((progress[1:], progress[-1:]))
            discounts = self.discount ** lengths.astype(np.float64)
            shaping = self.shaping_scale * (discounts * masks * progress_next - progress)
        rewards = (sparse + shaping).astype(np.float32)
        if not np.isfinite(rewards).all():
            raise ValueError("Shaped rewards contain NaN or Inf")

        traj["rewards"] = rewards
        traj["progress_video_data"] = {
            "progress_gated": progress.astype(np.float32),
            "is_ood": is_ood,
            "conformal_p_value": np.asarray(conformal_p_values, dtype=np.float32),
            "bin_progress_values": np.asarray(
                self.provider.gaussian_model["bin_progress_values"], dtype=np.float32),
        }
        del traj["reward_frames"]
        return {
            "shaped_reward/episode_return": float(rewards.sum()),
            "shaped_reward/sparse_return": float(sparse.sum()),
            "shaped_reward/shaping_return": float(shaping.sum()),
        }
