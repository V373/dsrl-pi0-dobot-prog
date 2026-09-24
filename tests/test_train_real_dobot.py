from contextlib import nullcontext
import os
import pty
import select
from types import SimpleNamespace

import gym
import jax
import numpy as np
import pytest
from matplotlib.axes import Axes

from examples import train_real_dobot
from examples.launch_train_real_dobot import parse_args
from examples.serve_lerobot_pi0_dobot import TorchPi0Service
from examples.train_utils_real import add_online_data_to_buffer
from jaxrl2.agents.pixel_sac.pixel_sac_learner import make_visual
from jaxrl2.data import ReplayBuffer


class FakeRobot:
    def __init__(self):
        self.actions = []
        self.reset_count = 0
        self.halt_count = 0
        self.events = []

    def reset(self):
        self.reset_count += 1
        self.events.append("reset")

    def halt(self):
        self.halt_count += 1
        self.events.append("halt")

    def get_observation(self):
        return {
            "external_rgb": np.full((8, 8, 3), 11, dtype=np.uint8),
            "wrist_rgb": np.full((8, 8, 3), 22, dtype=np.uint8),
            "joint_position": np.arange(6, dtype=np.float32),
            "gripper_position": np.array([0.5], dtype=np.float32),
        }

    def step(self, action):
        self.actions.append(np.asarray(action).copy())


class FakePolicy:
    def __init__(self):
        self.noises = []
        self.requests = []

    def get_prefix_rep(self, request):
        self.requests.append(request)
        return np.full((1, 4), 3, dtype=np.float32)

    def infer(self, request, noise):
        self.noises.append(np.asarray(noise).copy())
        actions = np.zeros((3, 7), dtype=np.float32)
        actions[:, 0] = 10 * len(self.noises) + np.arange(3)
        return {"actions": actions}


class FakeAgent:
    def __init__(self):
        self._rng = jax.random.PRNGKey(0)
        self.action_chunk_shape = (1, 5)
        self.sample_count = 0

    def sample_actions(self, observation):
        self.sample_count += 1
        assert observation["pixels"].shape == (1, 8, 8, 6, 1)
        return np.full((1, 5), 0.25, dtype=np.float32)


@pytest.mark.parametrize("training_step, keys, expected_steps, expected_query_lengths", [
    (0, [False, False, False, False], 4, [2, 2]),
    (1, [False, False, False, False], 4, [2, 2]),
    (1, [False, False, False, True], 3, [2, 1]),
    (1, [True, False, False, True], 3, [2, 1]),
])
def test_dobot_trajectory_and_replay(monkeypatch, tmp_path, training_step, keys, expected_steps, expected_query_lengths):
    labels = iter(["invalid", "1"])
    monkeypatch.setattr(train_real_dobot.time, "sleep", lambda _: None)
    monkeypatch.setattr(train_real_dobot, "_rollout_keyboard", nullcontext)
    keys_iter = iter(keys)
    monkeypatch.setattr(train_real_dobot, "_quit_pressed", lambda: next(keys_iter))
    variant = SimpleNamespace(
        query_freq=2, max_timesteps=4, control_hz=1000,
        resize_image=8, instruction="pick up the object", add_states=True, discount=0.99,
        outputdir=str(tmp_path),
    )
    robot, policy, agent = FakeRobot(), FakePolicy(), FakeAgent()
    monkeypatch.setattr("builtins.input", lambda _: (robot.events.append("label"), next(labels))[1])
    monkeypatch.setattr(train_real_dobot.pdb, "set_trace", lambda: robot.events.append("pdb"))

    class FakeClip:
        def __init__(self, frames, fps):
            assert fps == variant.control_hz
            assert len(frames) == expected_steps + 1
            assert all(frame.shape == (8, 8, 3) for frame in frames)
            assert all(np.all(frame == 11) for frame in frames)
            robot.events.append("video")

        def write_videofile(self, path, codec):
            assert path == str(tmp_path / "video_agent_rollout0.mp4")
            assert codec == "libx264"

    monkeypatch.setattr(train_real_dobot, "ImageSequenceClip", FakeClip)

    traj = train_real_dobot.collect_traj_dobot(
        variant, agent, robot, training_step, policy, None, 0,
        {"action_horizon": 3, "noise_dim": 5, "feature_dim": 4},
    )

    assert robot.reset_count == 1
    assert robot.halt_count == 1
    assert robot.events == ["halt", "label", "label", "video", "reset", "pdb"]
    assert len(robot.actions) == expected_steps
    assert [action[0] for action in robot.actions] == [10, 11, 20, 21][:expected_steps]
    assert len(policy.noises) == 2
    assert all(noise.shape == (1, 3, 5) for noise in policy.noises)
    assert all(np.array_equal(noise[:, 0], noise[:, 1]) for noise in policy.noises)
    assert agent.sample_count == (2 if training_step else 0)
    assert all(request["prompt"] == variant.instruction for request in policy.requests)
    assert len(traj["observations"]) == 3

    pixels = traj["observations"][0]["pixels"]
    assert pixels.shape == (1, 8, 8, 6, 1)
    assert np.all(pixels[0, ..., :3, 0] == 11)
    assert np.all(pixels[0, ..., 3:6, 0] == 22)
    state = traj["observations"][0]["state"]
    np.testing.assert_array_equal(state[0, :, 0], [0, 1, 2, 3, 4, 5, 0.5, 3, 3, 3, 3])
    assert traj["env_steps"] == expected_steps
    assert traj["query_step_lengths"] == expected_query_lengths

    observation_space = gym.spaces.Dict({
        "pixels": gym.spaces.Box(0, 255, shape=(8, 8, 6, 1), dtype=np.uint8),
        "state": gym.spaces.Box(-np.inf, np.inf, shape=(11, 1), dtype=np.float32),
    })
    action_space = gym.spaces.Box(-1, 1, shape=(1, 5), dtype=np.float32)
    buffer = ReplayBuffer(observation_space, action_space, capacity=2)
    add_online_data_to_buffer(variant, traj, buffer)
    assert len(buffer) == 2
    assert buffer._traj_counter == 1
    np.testing.assert_array_equal(buffer.data["masks"][:2], [1, 0])
    np.testing.assert_array_equal(buffer.data["rewards"][:2], [-1, 0])
    np.testing.assert_allclose(buffer.data["discount"][:2], [0.99**steps for steps in expected_query_lengths])
    np.testing.assert_array_equal(buffer.data["observations"]["pixels"][0], pixels[0])


@pytest.mark.parametrize("channels", [3, 6, 9])
@pytest.mark.parametrize("steps", [1, 4])
def test_q_visualization_uses_agent_view(monkeypatch, channels, steps):
    captured = []
    original_imshow = Axes.imshow

    def capture_imshow(axis, image, *args, **kwargs):
        captured.append(np.asarray(image).copy())
        return original_imshow(axis, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture_imshow)
    images = np.full((steps, 8, 8, channels, 1), 22, dtype=np.uint8)
    images[..., :3, :] = 11
    result = make_visual(
        [np.array([1.0, 2.0])] * steps,
        np.array([-1] * (steps - 1) + [0]),
        np.array([1] * (steps - 1) + [0]),
        images,
    )
    assert result.ndim == 3 and result.shape[-1] == 3
    assert captured[0].shape[-1] == 3
    assert np.all(captured[0] == 11)


def test_server_rejects_non_uint8_image():
    with pytest.raises(ValueError, match="uint8 RGB"):
        TorchPi0Service._image(np.zeros((8, 8, 3), dtype=np.float32), "external_rgb")


def test_halt_runs_if_robot_step_fails(monkeypatch):
    robot, policy, agent = FakeRobot(), FakePolicy(), FakeAgent()
    monkeypatch.setattr(train_real_dobot, "_rollout_keyboard", nullcontext)
    monkeypatch.setattr(train_real_dobot, "_quit_pressed", lambda: False)

    def failed_step(_):
        raise RuntimeError("robot step failed")

    robot.step = failed_step
    variant = SimpleNamespace(query_freq=1, max_timesteps=1, control_hz=15,
                              resize_image=8, instruction="pick up the object")
    with pytest.raises(RuntimeError, match="robot step failed"):
        train_real_dobot.collect_traj_dobot(
            variant, agent, robot, 0, policy, None, 0,
            {"action_horizon": 3, "noise_dim": 5, "feature_dim": 4},
        )
    assert robot.halt_count == 1


def test_q_key_is_read_from_tty_and_terminal_is_restored(monkeypatch):
    master, slave = pty.openpty()
    reader = os.fdopen(slave, "r")
    old_settings = train_real_dobot.termios.tcgetattr(slave)
    try:
        monkeypatch.setattr(train_real_dobot.sys, "stdin", reader)
        with train_real_dobot._rollout_keyboard():
            assert not train_real_dobot._quit_pressed()
            os.write(master, b"q")
            assert select.select([reader], [], [], 1)[0]
            assert train_real_dobot._quit_pressed()
        assert train_real_dobot.termios.tcgetattr(slave) == old_settings
    finally:
        reader.close()
        os.close(master)


def test_dobot_launcher_exposes_run_settings():
    variant = parse_args([
        "--instruction", "pick up the object", "--robot_factory", "examples.dobot_data_wrapper:create_robot",
        "--control_hz", "20", "--max_timesteps", "100", "--query_freq", "5",
        "--hidden_dims", "512", "--num_qs", "3",
    ])
    assert (variant.control_hz, variant.max_timesteps, variant.query_freq) == (20, 100, 5)
    assert variant.hidden_dims == [512]
    assert variant.num_qs == 3
