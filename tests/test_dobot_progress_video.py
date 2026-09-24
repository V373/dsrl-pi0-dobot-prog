from types import SimpleNamespace

import cv2
import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from examples.dobot_progress_video import _expand_query_values, save_dobot_progress_video
from examples import train_utils_real
from shaped_reward.dobot import DobotShapedReward


def test_query_values_hold_for_executed_steps_and_terminal_frame():
    values = np.array([[0.1, 0.9], [0.4, 0.6]])
    np.testing.assert_array_equal(
        _expand_query_values(values, np.array([2, 1])),
        [[0.1, 0.9], [0.1, 0.9], [0.4, 0.6], [0.4, 0.6]],
    )


def test_provider_diagnostics_are_optional_and_use_current_inference():
    pytest.importorskip("torchvision")
    from shaped_reward.gaussian_progress import BatchedGaussianProgressGatedProvider

    provider = object.__new__(BatchedGaussianProgressGatedProvider)
    provider.n_envs = 1
    provider.device = torch.device("cpu")
    provider.history_len = 2
    provider.enable_ood_filter = False
    provider._frame_ring = torch.zeros((1, 2, 3, 2, 2), dtype=torch.uint8)
    provider._env_indices = torch.arange(1)
    provider._head = torch.zeros(1, dtype=torch.long)
    provider._since_reset = torch.zeros(1, dtype=torch.long)
    provider._last_in_distribution = torch.zeros(1, dtype=torch.float64)
    provider.progress_current = None
    provider._write_frames = lambda frames: torch.zeros((1, 3, 2, 2), dtype=torch.uint8)
    provider._infer_batch = lambda: (
        torch.tensor([0.4], dtype=torch.float64),
        torch.tensor([0.8], dtype=torch.float64),
        torch.tensor([True]),
        torch.tensor([[0.1, 0.3]], dtype=torch.float64),
    )

    progress, diagnostics = provider.reset_all([None], return_diagnostics=True)
    np.testing.assert_allclose(progress, [0.4])
    np.testing.assert_array_equal(diagnostics["is_ood"], [True])
    np.testing.assert_allclose(diagnostics["conformal_p_value"], [[0.1, 0.3]])
    assert isinstance(provider.advance_all([None]), np.ndarray)


@pytest.mark.parametrize("reward_type", ["dense", "pbrs"])
def test_reward_trace_uses_same_query_inferences_as_replay(reward_type):
    class FakeProvider:
        gaussian_model = {"bin_progress_values": np.array([0.25, 0.75])}

        def __init__(self):
            self.index = -1

        def reset_all(self, frames, *, return_diagnostics=False):
            assert return_diagnostics and len(frames) == 1
            self.index = -1
            return self.advance_all(frames, return_diagnostics=True)

        def advance_all(self, frames, *, return_diagnostics=False):
            assert return_diagnostics and len(frames) == 1
            self.index += 1
            progress = [0.2, 0.45, 0.45][self.index]
            return np.array([progress]), {
                "is_ood": np.array([self.index == 2]),
                "conformal_p_value": np.array([[0.8 - 0.2 * self.index, 0.4]]),
            }

    plugin = object.__new__(DobotShapedReward)
    plugin.provider = FakeProvider()
    plugin.reward_type = reward_type
    plugin.query_freq = 2
    plugin.discount = 0.99
    plugin.shaping_scale = 1.0
    traj = {
        "actions": [None] * 3,
        "observations": [None] * 4,
        "reward_frames": [np.zeros((224, 224, 3), dtype=np.uint8)] * 3,
        "query_step_lengths": [2, 2, 1],
        "masks": np.array([1, 1, 0]),
        "env_steps": 5,
        "is_success": True,
    }
    plugin.apply(traj, lambda frame: frame)

    progress = np.array([0.2, 0.45, 0.45])
    if reward_type == "dense":
        expected = np.array([0.2, 0.45, 1.45])
    else:
        expected = np.array([
            0.99 ** 2 * progress[1] - progress[0],
            0.99 ** 2 * progress[2] - progress[1],
            1.0 - progress[2],
        ])
    np.testing.assert_allclose(traj["rewards"], expected, rtol=1e-6)
    trace = traj["progress_video_data"]
    np.testing.assert_allclose(trace["progress_gated"], progress)
    np.testing.assert_array_equal(trace["is_ood"], [False, False, True])
    assert trace["conformal_p_value"].shape == (3, 2)
    assert "reward_frames" not in traj
    assert plugin.provider.index == 2


def test_composite_mp4_has_one_frame_per_control_observation(tmp_path):
    raw_path = tmp_path / "raw.mp4"
    with imageio.get_writer(raw_path, fps=10, codec="libx264", macro_block_size=1) as writer:
        for index in range(6):
            writer.append_data(np.full((64, 64, 3), index * 30, dtype=np.uint8))

    output_path = tmp_path / "progress.mp4"
    result = save_dobot_progress_video(
        raw_video_path=raw_path,
        output_path=output_path,
        query_step_lengths=[2, 3],
        progress_gated=[0.2, 0.6],
        is_ood=[False, True],
        conformal_p_value=[[0.8, 0.4, 0.1], [0.1, 0.3, 0.9]],
        bin_progress_values=[0.0, 0.5, 1.0],
        rewards=[-0.2, 0.8],
        is_success=True,
        reward_type="pbrs",
        fps=10,
    )
    assert result == output_path
    assert output_path.is_file()
    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10)
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 224
        assert sum(capture.read()[0] for _ in range(7)) == 6
    finally:
        capture.release()


def test_progress_video_is_saved_and_logged_before_replay_insert(monkeypatch, tmp_path):
    events = []

    class StopAtBuffer(Exception):
        pass

    def collect(*args):
        return {
            "rollout_video_path": "raw.mp4",
            "query_step_lengths": [2],
            "rewards": np.array([0.0]),
            "is_success": False,
            "env_steps": 2,
        }

    class RewardPlugin:
        reward_type = "pbrs"

        def apply(self, traj, preprocess_frame):
            events.append("infer")
            traj["progress_video_data"] = {
                "progress_gated": np.array([0.2]),
                "is_ood": np.array([False]),
                "conformal_p_value": np.array([[0.5, 0.6]]),
                "bin_progress_values": np.array([0.0, 1.0]),
            }
            return {"shaped_reward/episode_return": 0.0}

    def save_video(**kwargs):
        events.append("save")
        assert kwargs["query_step_lengths"] == [2]

    class FakeLogger:
        def log(self, payload, *, step):
            if "rollout/progress_video" in payload:
                events.append("wandb")
                assert step == 0

    def insert(*args):
        events.append("insert")
        raise StopAtBuffer

    monkeypatch.setattr(train_utils_real, "save_dobot_progress_video", save_video)
    monkeypatch.setattr(train_utils_real.wandb, "Video", lambda *args, **kwargs: "video")
    monkeypatch.setattr(train_utils_real, "add_online_data_to_buffer", insert)
    buffer = SimpleNamespace(get_iterator=lambda batch_size: iter(()))
    variant = SimpleNamespace(max_steps=0, batch_size=1, outputdir=str(tmp_path), control_hz=25)
    env = SimpleNamespace(prepare_progress_frame=lambda frame: frame)
    with pytest.raises(StopAtBuffer):
        train_utils_real.trajwise_alternating_training_loop(
            variant, None, env, env, buffer, buffer, FakeLogger(),
            collect_fn=collect, reward_plugin=RewardPlugin(), dobot_logging=True,
        )
    assert events == ["infer", "save", "wandb", "insert"]
