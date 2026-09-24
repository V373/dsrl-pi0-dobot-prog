#!/usr/bin/env python3
"""Run DSRL on a Dobot wrapper with a remote, flow-based pi0 policy.

The wrapper factory passed through launch_train_real_dobot.py returns an
object with reset(), get_observation(), step(action), and halt() methods. Its observation
contains RGB uint8 external_rgb/wrist_rgb images, six joint positions, and one
gripper position. State and action units must match the pi0 SFT dataset.
"""

import importlib
import os
from pathlib import Path
import pdb
import select
import sys
import tempfile
import termios
import time
import tty
from contextlib import contextmanager
from functools import partial

import gym
import jax
import numpy as np
from gym.spaces import Box, Dict
from moviepy.editor import ImageSequenceClip
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.train_utils_real import trajwise_alternating_training_loop
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name


ROBOT_STATE_DIM = 7
CAMERA_COUNT = 2  # agent view and right wrist


def _validated_observation(raw):
    obs = {}
    for key in ("external_rgb", "wrist_rgb"):
        image = np.asarray(raw[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be an HWC uint8 RGB image, got {image.shape} {image.dtype}")
        obs[key] = np.ascontiguousarray(image)
    for key, size in (("joint_position", 6), ("gripper_position", 1)):
        value = np.asarray(raw[key], dtype=np.float32).reshape(-1)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain {size} finite values, got {value.shape}")
        obs[key] = value
    return obs


def _policy_request(obs, instruction):
    return {**obs, "prompt": instruction}


def _sac_observation(obs, prefix, image_size, feature_dim):
    feature = np.asarray(prefix, dtype=np.float32)
    if feature.shape != (1, feature_dim) or not np.isfinite(feature).all():
        raise ValueError(f"Expected pi0 prefix feature (1, {feature_dim}), got {feature.shape}")
    agent_view = image_tools.resize_with_pad(obs["external_rgb"], image_size, image_size)
    wrist_view = image_tools.resize_with_pad(obs["wrist_rgb"], image_size, image_size)
    pixels = np.concatenate((agent_view, wrist_view), axis=-1)
    state = np.concatenate((obs["joint_position"], obs["gripper_position"], feature[0]))
    return {
        "pixels": pixels[None, ..., None],
        "state": state[None, ..., None],
    }


@contextmanager
def _rollout_keyboard():
    fd = sys.stdin.fileno()
    settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)


def _quit_pressed():
    return bool(select.select([sys.stdin], [], [], 0)[0]) and sys.stdin.read(1).lower() == "q"


def collect_traj_dobot(variant, agent, env, i, agent_dp, wandb_logger, traj_id, robot_config):
    horizon = robot_config["action_horizon"]
    noise_dim = robot_config["noise_dim"]
    feature_dim = robot_config["feature_dim"]
    query_freq = variant.query_freq
    agent._rng, rng = jax.random.split(agent._rng)

    actions_noise = []
    observations = []
    query_step_lengths = []
    video_frames = []
    reward_frames = [] if getattr(variant, "reward_type", "sparse") != "sparse" else None
    action_chunk = None
    period = 1.0 / variant.control_hz
    env_steps = 0

    try:
        with _rollout_keyboard():
            for t in range(variant.max_timesteps):
                start = time.monotonic()
                if _quit_pressed():
                    if t > 0:
                        print("'q' pressed, ending this rollout.")
                        break
                    print("No action has run yet; ignoring 'q' at step 0.")

                obs = _validated_observation(env.get_observation())
                video_frames.append(obs["external_rgb"].copy())
                if t % query_freq == 0:
                    request = _policy_request(obs, variant.instruction)
                    feature = agent_dp.get_prefix_rep(request)
                    sac_obs = _sac_observation(obs, feature, variant.resize_image, feature_dim)

                    rng, key = jax.random.split(rng)
                    if i == 0:
                        steering = np.asarray(jax.random.normal(key, agent.action_chunk_shape), dtype=np.float32)
                    else:
                        steering = np.asarray(agent.sample_actions(sac_obs), dtype=np.float32).reshape(
                            agent.action_chunk_shape
                        )
                    if steering.shape != (1, noise_dim) or not np.isfinite(steering).all():
                        raise ValueError(f"Expected SAC noise action (1, {noise_dim}), got {steering.shape}")
                    noise = np.repeat(steering[None, ...], horizon, axis=1)
                    action_chunk = np.asarray(agent_dp.infer(request, noise=noise)["actions"], dtype=np.float32)
                    if action_chunk.shape != (horizon, ROBOT_STATE_DIM) or not np.isfinite(action_chunk).all():
                        raise ValueError(f"Expected pi0 actions ({horizon}, {ROBOT_STATE_DIM}), got {action_chunk.shape}")
                    observations.append(sac_obs)
                    actions_noise.append(steering)
                    query_step_lengths.append(0)
                    if reward_frames is not None:
                        reward_frames.append(video_frames[-1])

                env.step(action_chunk[t % query_freq])
                env_steps += 1
                query_step_lengths[-1] += 1
                remaining = period - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        env.halt()

    while True:
        label = input("Trial finished. Success (1) or failure (0)? ").strip()
        if label in ("0", "1"):
            is_success = label == "1"
            break

    final_obs = _validated_observation(env.get_observation())
    video_frames.append(final_obs["external_rgb"].copy())
    final_request = _policy_request(final_obs, variant.instruction)
    final_feature = agent_dp.get_prefix_rep(final_request)
    observations.append(_sac_observation(final_obs, final_feature, variant.resize_image, feature_dim))

    query_steps = len(actions_noise)
    rewards = -np.ones(query_steps, dtype=np.float32)
    masks = np.ones(query_steps, dtype=np.float32)
    if is_success:
        rewards[-1] = 0
        masks[-1] = 0
    if wandb_logger is not None:
        wandb_logger.log({"is_success": int(is_success), "total_num_traj": traj_id}, step=i)
    video_path = os.path.join(variant.outputdir, f"video_high_{traj_id}.mp4")
    ImageSequenceClip(video_frames, fps=variant.control_hz).write_videofile(video_path, codec="libx264")
    env.reset()
    print("Episode done. Enter 'c' in pdb to continue training.")
    pdb.set_trace()
    traj = {
        "observations": observations,
        "actions": actions_noise,
        "rewards": rewards,
        "masks": masks,
        "is_success": is_success,
        "env_steps": env_steps,
        "query_step_lengths": query_step_lengths,
    }
    if reward_frames is not None:
        traj["reward_frames"] = reward_frames
    return traj


def _robot_factory(path):
    module_name, separator, function_name = path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--robot-factory must be module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    robot = factory()
    for method in ("reset", "get_observation", "step", "halt"):
        if not callable(getattr(robot, method, None)):
            raise TypeError(f"Robot wrapper must implement {method}()")
    return robot


def _shard_batch(batch, sharding):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))),
        batch,
    )


def main(variant):
    if variant.query_freq < 1 or variant.max_timesteps < 1 or variant.control_hz <= 0:
        raise ValueError("query_freq, max_timesteps, and control_hz must be positive")
    if variant.multi_grad_step < 1 or variant.resize_image < 1:
        raise ValueError("multi_grad_step and resize_image must be positive")
    if not sys.stdin.isatty():
        raise RuntimeError("An interactive terminal is required for q, labeling, and pdb")
    variant.add_states = True  # Required by the shared replay-buffer insertion helper.
    variant.num_initial_traj_collect = 1
    variant.checkpoint_interval = -1
    variant.restore_path = ""
    variant.suffix = ""
    variant.launch_group_id = ""

    reward_type = getattr(variant, "reward_type", "sparse")
    if reward_type not in ("sparse", "dense", "pbrs"):
        raise ValueError("reward_type must be sparse, dense, or pbrs")
    reward_plugin = None
    if reward_type != "sparse":
        from shaped_reward.dobot import DobotShapedReward

        reward_plugin = DobotShapedReward(
            reward_type=reward_type,
            checkpoint_path=variant.reward_checkpoint,
            gaussian_model_h5_path=variant.reward_gaussian_h5,
            calibration_h5_path=variant.reward_calibration_h5,
            context_stride=variant.reward_context_stride,
            query_freq=variant.query_freq,
            discount=variant.discount,
            device=variant.reward_device,
            ood_p_value_threshold=variant.reward_ood_threshold,
            posterior_temperature=variant.reward_posterior_temperature,
            shaping_scale=variant.reward_shaping_scale,
        )

    kwargs = dict(
        actor_lr=1e-4, critic_lr=3e-4, temp_lr=3e-4,
        hidden_dims=tuple(variant.hidden_dims), cnn_features=(32, 32, 32, 32),
        cnn_strides=(3, 2, 2, 2), cnn_padding="VALID", latent_dim=50,
        discount=variant.discount, tau=0.005, critic_reduction="min", dropout_rate=0.0,
        aug_next=1, use_bottleneck=True, encoder_type="small", encoder_norm="group",
        use_spatial_softmax=True, softmax_temperature=-1, target_entropy=0.0,
        num_qs=variant.num_qs, action_magnitude=variant.action_magnitude,
        num_cameras=CAMERA_COUNT,
    )
    variant.train_kwargs = kwargs

    # Match the existing real-robot path: keep TensorFlow off the training GPU.
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    client = WebsocketClientPolicy(host=variant.remote_host, port=variant.remote_port)
    metadata = client.get_server_metadata()
    if metadata.get("model_type") != "pi0_flow":
        raise ValueError(f"A flow-based pi0 server is required, got {metadata!r}")
    horizon = int(metadata["action_horizon"])
    noise_dim = int(metadata["noise_dim"])
    feature_dim = int(metadata["feature_dim"])
    if min(horizon, noise_dim, feature_dim) < 1 or variant.query_freq > horizon:
        raise ValueError(f"Invalid policy dimensions or query_freq: {metadata!r}")
    if int(metadata["robot_action_dim"]) != ROBOT_STATE_DIM:
        raise ValueError(f"Expected a {ROBOT_STATE_DIM}-dimensional robot action: {metadata!r}")

    devices = jax.local_devices()
    if variant.batch_size % len(devices):
        raise ValueError("batch_size must be divisible by the number of local JAX devices")
    shard_fn = partial(_shard_batch, sharding=jax.sharding.PositionalSharding(devices))

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
    variant.outputdir = os.path.join(os.environ.get("EXP", "./logs/dsrl_pi0_dobot"), expname)
    os.makedirs(variant.outputdir, exist_ok=True)
    logger = WandBLogger(
        True,
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=tempfile.mkdtemp(),
        group_name=variant.prefix + "_" + variant.launch_group_id,
    )

    observation_space = Dict({
        "pixels": Box(0, 255, shape=(variant.resize_image, variant.resize_image, 3 * CAMERA_COUNT, 1), dtype=np.uint8),
        "state": Box(-np.inf, np.inf, shape=(ROBOT_STATE_DIM + feature_dim, 1), dtype=np.float32),
    })
    action_space = Box(-1, 1, shape=(1, noise_dim), dtype=np.float32)
    sample_obs = add_batch_dim(observation_space.sample())
    sample_action = add_batch_dim(action_space.sample())
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)
    if variant.restore_path:
        agent.restore_checkpoint(variant.restore_path)

    capacity = max(1, 2 * variant.max_steps // variant.multi_grad_step)
    buffer = ReplayBuffer(observation_space, action_space, capacity)
    buffer.seed(variant.seed)
    robot = _robot_factory(variant.robot_factory)
    robot_config = {"action_horizon": horizon, "noise_dim": noise_dim, "feature_dim": feature_dim}
    try:
        if reward_plugin is not None and not callable(getattr(robot, "prepare_progress_frame", None)):
            raise TypeError("Shaped reward requires robot.prepare_progress_frame()")
        robot.reset()
        if reward_plugin is not None:
            initial_obs = _validated_observation(robot.get_observation())
            initial_frame = np.asarray(
                robot.prepare_progress_frame(initial_obs["external_rgb"])
            )
            if initial_frame.shape != (224, 224, 3) or initial_frame.dtype != np.uint8:
                raise ValueError(
                    "prepare_progress_frame() must return a 224x224 RGB uint8 frame, "
                    f"got {initial_frame.shape} {initial_frame.dtype}"
                )
        trajwise_alternating_training_loop(
            variant, agent, robot, robot, buffer, buffer, logger,
            shard_fn=shard_fn, agent_dp=client, robot_config=robot_config,
            collect_fn=collect_traj_dobot, reward_plugin=reward_plugin,
        )
    finally:
        if callable(getattr(robot, "close", None)):
            robot.close()


if __name__ == "__main__":
    raise SystemExit("Run examples/launch_train_real_dobot.py or examples/scripts/run_real_dobot.sh")
