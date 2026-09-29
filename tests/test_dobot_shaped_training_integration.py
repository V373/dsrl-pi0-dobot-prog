"""Exercise the Dobot entry point through shaped reward, replay and update.

Robot I/O, the remote policy, Gaussian inference and the learner are test doubles.
The production entry point, rollout collector, crop, reward calculation, replay
insertion, replay sampling and JAX batch sharding all run unchanged. Real asset
and GPU operator results are summarized in
shaped_reward/assets/pick_mango/GPU_ENV_VALIDATION_STATUS.md.
"""

from contextlib import nullcontext
import sys
from types import ModuleType

import jax
import numpy as np
import pytest

from examples import train_real_dobot, train_utils_real
from examples.dobot_data_wrapper import DobotDataWrapper
from examples.launch_train_real_dobot import parse_args
from jaxrl2.data import ReplayBuffer


class _FirstUpdateReached(Exception):
    """Stop the production loop before its fixed 5,000-update warmup."""


@pytest.mark.parametrize("context_stride", [10, 20])
@pytest.mark.parametrize("reward_type", ["dense", "pbrs"])
@pytest.mark.parametrize("success", [False, True])
def test_dobot_main_shapes_rewards_before_first_sac_update(
    monkeypatch, tmp_path, context_stride, reward_type, success
):
    captured = {"prepared_frames": [], "logs": [], "video": []}
    asset_paths = []
    for name in ("encoder.pt", "gaussian.h5", "calibration.h5"):
        path = tmp_path / name
        path.touch()
        asset_paths.append(str(path))
    variant = parse_args([
        "--instruction", "pick mango", "--reward_type", reward_type,
        "--reward_checkpoint", asset_paths[0],
        "--reward_gaussian_h5", asset_paths[1],
        "--reward_calibration_h5", asset_paths[2],
        "--reward_context_stride", str(context_stride),
        "--reward_device", "cuda", "--reward_shaping_scale", "2",
        "--max_timesteps", "23", "--query_freq", "10",
        "--resize_image", "8", "--batch_size", "32",
        "--max_steps", "30", "--multi_grad_step", "1",
    ])

    class Provider:
        def __init__(self, count, **kwargs):
            assert count == 1
            captured["provider_kwargs"] = kwargs
            self.gaussian_model = {"bin_progress_values": np.array([0., 0.5, 1.])}
            self.index = 0

        def _infer(self, frames, return_diagnostics):
            assert return_diagnostics
            assert len(frames) == 1
            frame = frames[0]
            assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8
            captured["prepared_frames"].append(frame.copy())
            progress = [0.2, 0.2, 0.6][self.index]
            diagnostics = {
                "is_ood": np.array([self.index == 1]),
                "conformal_p_value": np.array([[0.7, 0.8, 0.9]], np.float32),
            }
            self.index += 1
            return np.array([progress]), diagnostics

        def reset_all(self, frames, return_diagnostics):
            captured["provider_resets"] = captured.get("provider_resets", 0) + 1
            self.index = 0
            return self._infer(frames, return_diagnostics)

        def advance_all(self, frames, return_diagnostics):
            return self._infer(frames, return_diagnostics)

    provider_module = ModuleType("shaped_reward.gaussian_progress")
    provider_module.BatchedGaussianProgressGatedProvider = Provider
    monkeypatch.setitem(sys.modules, "shaped_reward.gaussian_progress", provider_module)

    class Robot(DobotDataWrapper):
        def __init__(self):
            self.steps = 0
            self.reset_count = 0
            self.halt_count = 0
            self.closed = False

        def reset(self):
            self.reset_count += 1

        def get_observation(self):
            # Only the crop has this marker, so the test also checks the actual
            # 480x640 topFullImg -> 224x224 FineProg crop boundary.
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[168:392, 256:480] = self.steps + 1
            return {
                "external_rgb": frame,
                "wrist_rgb": np.zeros((12, 16, 3), np.uint8),
                "joint_position": np.arange(6, dtype=np.float32),
                "gripper_position": np.array([0.5], np.float32),
            }

        def step(self, action):
            assert action.shape == (7,)
            assert np.isfinite(action).all()
            self.steps += 1

        def halt(self):
            self.halt_count += 1

        def close(self):
            self.closed = True

    class Policy:
        def __init__(self, **kwargs):
            self.requests = 0

        def get_server_metadata(self):
            return {"model_type": "pi0_flow", "action_horizon": 10,
                    "noise_dim": 5, "feature_dim": 4, "robot_action_dim": 7}

        def get_prefix_rep(self, request):
            assert request["prompt"] == "pick mango"
            return np.full((1, 4), 0.25, np.float32)

        def infer(self, request, noise):
            assert noise.shape == (1, 10, 5)
            assert np.array_equal(noise[:, :1].repeat(10, axis=1), noise)
            self.requests += 1
            return {"actions": np.zeros((10, 7), np.float32)}

    class Learner:
        def __init__(self, seed, observation, actions, **kwargs):
            assert observation["pixels"].shape == (1, 8, 8, 6, 1)
            assert observation["state"].shape == (1, 11, 1)
            assert actions.shape == (1, 1, 5)
            self._rng = jax.random.PRNGKey(seed)
            self.action_chunk_shape = (1, 5)

        def update(self, batch):
            captured["batch"] = jax.device_get(batch)
            raise _FirstUpdateReached

    class Logger:
        def __init__(self, *args, **kwargs):
            pass

        def log(self, values, step):
            captured["logs"].append((values, step))

    class Clip:
        def __init__(self, frames, fps):
            assert len(frames) == 24

        def write_videofile(self, path, codec):
            pass

    def make_buffer(*args, **kwargs):
        buffer = ReplayBuffer(*args, **kwargs)
        captured["buffer"] = buffer
        return buffer

    robot = Robot()
    monkeypatch.setenv("EXP", str(tmp_path))
    monkeypatch.setattr(train_real_dobot.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(train_real_dobot, "_robot_factory", lambda _: robot)
    monkeypatch.setattr(train_real_dobot, "WebsocketClientPolicy", Policy)
    monkeypatch.setattr(train_real_dobot, "PixelSACLearner", Learner)
    monkeypatch.setattr(train_real_dobot, "ReplayBuffer", make_buffer)
    monkeypatch.setattr(train_real_dobot, "WandBLogger", Logger)
    monkeypatch.setattr(train_real_dobot, "ImageSequenceClip", Clip)
    monkeypatch.setattr(train_real_dobot, "_rollout_keyboard", nullcontext)
    monkeypatch.setattr(train_real_dobot, "_quit_pressed", lambda: False)
    monkeypatch.setattr(train_real_dobot.time, "sleep", lambda _: None)
    monkeypatch.setattr(train_real_dobot.pdb, "set_trace", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _: "1" if success else "0")
    monkeypatch.setattr(train_utils_real, "save_dobot_progress_video",
                        lambda **kwargs: captured["video"].append(kwargs))
    monkeypatch.setattr(train_real_dobot.wandb, "Video", lambda *args, **kwargs: "video")

    with pytest.raises(_FirstUpdateReached):
        train_real_dobot.main(variant)

    assert robot.steps == 23 and robot.reset_count == 2
    assert robot.halt_count == 1 and robot.closed
    assert captured["provider_resets"] == 1
    assert captured["provider_kwargs"]["device"] == "cuda"
    assert captured["provider_kwargs"]["ood_p_value_threshold"] == 0.5
    assert captured["provider_kwargs"]["frame_history_stride"] == context_stride // 10
    for frame, marker in zip(captured["prepared_frames"], [1, 11, 21], strict=True):
        assert np.all(frame == marker)

    discounts = np.array([0.99**10, 0.99**10, 0.99**3])
    masks = np.array([1., 1., 0. if success else 1.])
    progress = np.array([0.2, 0.2, 0.6])
    expected = 2 * progress if reward_type == "dense" else (
        2 * (discounts * masks * np.array([0.2, 0.6, 0.6]) - progress)
    )
    expected[-1] += float(success)
    buffer = captured["buffer"]
    assert len(buffer) == 3 and buffer._traj_counter == 1
    np.testing.assert_allclose(buffer.data["rewards"][:3], expected, atol=1e-7)
    np.testing.assert_array_equal(buffer.data["masks"][:3], masks)
    np.testing.assert_allclose(buffer.data["discount"][:3], discounts)
    assert buffer.data["rewards"].dtype == np.float32

    batch = captured["batch"]
    assert batch["actions"].shape == (32, 1, 5)
    assert batch["observations"]["pixels"].shape == (32, 8, 8, 6, 1)
    assert batch["observations"]["state"].shape == (32, 11, 1)
    # Every sampled transition must preserve the same reward, mask and discount
    # tuple as replay; this catches accidental sparse reward insertion.
    valid = np.stack((expected, masks, discounts), axis=1)
    sampled = np.stack((batch["rewards"], batch["masks"], batch["discount"]), axis=1)
    assert np.all(np.any(np.isclose(sampled[:, None], valid[None], atol=1e-7).all(axis=-1), axis=1))
    assert len(captured["video"]) == 1
    assert captured["video"][0]["query_step_lengths"] == [10, 10, 3]
    np.testing.assert_allclose(captured["video"][0]["rewards"], expected, atol=1e-7)
    shaped_logs = [values for values, _ in captured["logs"]
                   if "shaped_reward/episode_return" in values]
    assert len(shaped_logs) == 1
    assert shaped_logs[0]["shaped_reward/episode_return"] == pytest.approx(expected.sum())
