"""Minimal data boundary between Dobot hardware and train_real_dobot.py.

Use ``--robot_factory examples.dobot_data_wrapper:create_robot``. Fill in
reset, halt, and the transport methods once the robot SDK, cameras, and SFT
action format are known. Until then, hardware operations raise
NotImplementedError.

Observation returned by get_observation() (one robot control step, no batch axis):
    external_rgb:     (H_ext, W_ext, 3), uint8 RGB, values 0..255
    wrist_rgb:        (H_wrist, W_wrist, 3), uint8 RGB, values 0..255
    joint_position:   (6,), float32, in the SFT checkpoint's joint order/units
    gripper_position: (1,), float32, in the SFT checkpoint's gripper units
The two raw camera resolutions may differ. Downstream, train_real_dobot.py
resizes both to S x S and forms SAC pixels (1, S, S, 6, 1). It concatenates
the seven robot-state values with a pi0 prefix feature of checkpoint-dependent
size F to form SAC state (1, 7 + F, 1).

Action passed to step(): (7,), finite float32, one pi0 robot action per control
step. The pi0 server returns an action chunk (horizon, 7); SAC's separate
noise action has shape (1, noise_dim) and is never passed to this wrapper.
The meaning, order, and units of the seven robot-action values must be checked
against the SFT checkpoint before mapping them to SDK commands.
"""

import numpy as np


class DobotDataWrapper:
    def __init__(self):
        # TODO: Open and retain the Dobot SDK connection and both camera handles.
        pass

    def reset(self):
        # TODO: Move Dobot to the specified start pose and confirm completion.
        raise NotImplementedError("Implement Dobot reset to the start pose")

    def get_observation(self):
        """Return the unbatched image and (6,) + (1,) state fields above."""
        raw = self._receive_observation()
        observation = {}
        for name in ("external_rgb", "wrist_rgb"):
            image = np.asarray(raw[name])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
                raise ValueError(f"{name} must be a nonempty HWC uint8 RGB image, got {image.shape} {image.dtype}")
            observation[name] = np.ascontiguousarray(image)

        for name, size in (("joint_position", 6), ("gripper_position", 1)):
            value = np.asarray(raw[name], dtype=np.float32).reshape(-1)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain {size} finite values, got {value.shape}")
            observation[name] = value
        return observation

    def step(self, action):
        """Forward one unbatched pi0 action, shape (7,), unchanged."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"action must contain seven finite values, got {action.shape}")
        self._send_action(action)

    def halt(self):
        # TODO: Stop active motion before success/failure labeling or on error.
        raise NotImplementedError("Implement Dobot motion halt")

    def close(self):
        # TODO: Close the Dobot SDK connection and both camera handles here.
        pass

    def _receive_observation(self):
        # TODO: Read a time-aligned snapshot from the stored handles with these keys:
        #   external_rgb: (H_ext, W_ext, 3), uint8 RGB (convert BGR if needed);
        #   wrist_rgb: (H_wrist, W_wrist, 3), uint8 RGB;
        #   joint_position: (6,), six joints in the SFT dataset order and units;
        #   gripper_position: (1,), in the SFT dataset's units.
        #   TODO: 上面图像的obs keys，和pi0 SFT时候的输入时的crop、resize等预处理操作保持一致即可，receive预处理后、送入pi0推理的图（need confirm）；
        #   或者是receive原图，然后在这个类中进行crop、resize等预处理操作，再送入pi0推理、sac预处理+推理、progress推理等。
        raise NotImplementedError("Implement Dobot state and camera acquisition")

    def _send_action(self, action):
        # action: (7,) float32 from one row of the pi0 action chunk.
        # TODO: Confirm what the SFT checkpoint's seven action values mean.
        # TODO: Map them to the SDK's arm/gripper commands, enforce hardware
        # limits, and check the command acknowledgement before returning.
        raise NotImplementedError("Implement Dobot action transmission")


def create_robot():
    """No-argument factory used by train_real_dobot.py."""
    return DobotDataWrapper()
