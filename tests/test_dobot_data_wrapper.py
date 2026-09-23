import numpy as np
import pytest

from examples.dobot_data_wrapper import DobotDataWrapper
from examples.train_real_dobot import _robot_factory


def observation():
    return {
        "external_rgb": np.full((4, 6, 3), 10, dtype=np.uint8),
        "wrist_rgb": np.full((4, 6, 3), 20, dtype=np.uint8),
        "joint_position": np.arange(6, dtype=np.float64),
        "gripper_position": [0.5],
    }


def test_factory_and_data_transfer(monkeypatch):
    robot = _robot_factory("examples.dobot_data_wrapper:create_robot")
    sent = []
    monkeypatch.setattr(robot, "_receive_observation", observation)
    monkeypatch.setattr(robot, "_send_action", lambda action: sent.append(action.copy()))

    received = robot.get_observation()
    assert received["external_rgb"].shape == (4, 6, 3)
    assert received["external_rgb"].dtype == np.uint8
    assert received["wrist_rgb"].dtype == np.uint8
    assert received["joint_position"].shape == (6,)
    assert received["joint_position"].dtype == np.float32
    assert received["gripper_position"].shape == (1,)
    action = np.arange(7, dtype=np.float32)
    robot.step(action)
    np.testing.assert_array_equal(sent, [action])
    robot.close()


def test_unimplemented_transport_fails_before_hardware_io():
    robot = DobotDataWrapper()
    with pytest.raises(NotImplementedError, match="start pose"):
        robot.reset()
    with pytest.raises(NotImplementedError, match="acquisition"):
        robot.get_observation()
    with pytest.raises(NotImplementedError, match="transmission"):
        robot.step(np.zeros(7, dtype=np.float32))
    with pytest.raises(NotImplementedError, match="halt"):
        robot.halt()


def test_invalid_observation_and_action_are_rejected(monkeypatch):
    robot = DobotDataWrapper()
    bad = observation()
    bad["external_rgb"] = bad["external_rgb"].astype(np.float32)
    monkeypatch.setattr(robot, "_receive_observation", lambda: bad)
    with pytest.raises(ValueError, match="external_rgb"):
        robot.get_observation()

    bad = observation()
    bad["joint_position"][0] = np.nan
    monkeypatch.setattr(robot, "_receive_observation", lambda: bad)
    with pytest.raises(ValueError, match="joint_position"):
        robot.get_observation()
    with pytest.raises(ValueError, match="seven finite"):
        robot.step(np.zeros(6))
    with pytest.raises(ValueError, match="seven finite"):
        robot.step([0, 0, 0, 0, 0, 0, np.nan])
