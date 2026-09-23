#!/usr/bin/env python3
"""Serve a LeRobot PyTorch pi0 checkpoint to the Dobot DSRL client.

This server uses the checkpoint's saved LeRobot processors. The camera-key
arguments are the two image feature names recorded in its policy config.
"""

import argparse
import asyncio
import logging
import traceback
from pathlib import Path

import numpy as np
import torch
import websockets.asyncio.server
from openpi_client import msgpack_numpy


class TorchPi0Service:
    def __init__(self, policy, preprocessor, postprocessor, agentview_key, wrist_key):
        from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d

        self.policy = policy.eval()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.agentview_key = agentview_key
        self.wrist_key = wrist_key
        self.device = next(policy.parameters()).device
        self.horizon = int(policy.config.chunk_size)
        self.noise_dim = int(policy.config.max_action_dim)
        self.action_dim = int(policy.config.output_features[ACTION].shape[0])
        self.feature_dim = int(
            policy.model.paligemma_with_expert.paligemma.model.language_model.config.hidden_size
        )
        if self.action_dim != 7:
            raise ValueError(f"Dobot wrapper expects seven policy actions, got {self.action_dim}")
        state_feature = policy.config.input_features.get(OBS_STATE)
        if state_feature is None or tuple(state_feature.shape) != (7,):
            raise ValueError(f"Dobot wrapper expects a seven-dimensional checkpoint state, got {state_feature}")
        if self.noise_dim < self.action_dim or int(policy.config.max_state_dim) < 7:
            raise ValueError("pi0 padded state/action dimensions must be at least seven")
        if agentview_key == wrist_key or not {agentview_key, wrist_key}.issubset(policy.config.image_features):
            raise ValueError(f"Camera keys must name two distinct features in {list(policy.config.image_features)}")

        self.state_key = OBS_STATE
        self.tokens_key = OBS_LANGUAGE_TOKENS
        self.token_mask_key = OBS_LANGUAGE_ATTENTION_MASK
        self.make_att_2d_masks = make_att_2d_masks
        self.prepare_attention_masks_4d = prepare_attention_masks_4d

    @classmethod
    def from_checkpoint(cls, checkpoint, agentview_key, wrist_key):
        from safetensors.torch import load_file
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        checkpoint = Path(checkpoint)
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"LeRobot weights not found: {weights}")
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if config.type != "pi0":
            raise ValueError(f"Expected a LeRobot pi0 checkpoint, got {config.type}")

        # Load strictly: some LeRobot releases return a random model when their
        # from_pretrained weight load fails, which is unsuitable for robot control.
        policy = PI0Policy(config)
        state = load_file(str(weights), device="cpu")
        state = policy._fix_pytorch_state_dict_keys(state, config)
        state = {key if key.startswith("model.") else f"model.{key}": value for key, value in state.items()}
        policy.load_state_dict(state, strict=True)
        preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
        return cls(policy, preprocessor, postprocessor, agentview_key, wrist_key)

    @property
    def metadata(self):
        return {
            "model_type": "pi0_flow",
            "action_horizon": self.horizon,
            "noise_dim": self.noise_dim,
            "feature_dim": self.feature_dim,
            "robot_action_dim": self.action_dim,
        }

    @staticmethod
    def _image(raw, name):
        image = np.asarray(raw)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name} must be an HWC uint8 RGB image, got {image.shape} {image.dtype}")
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

    def _batch(self, obs):
        joint = np.asarray(obs["joint_position"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1)
        if joint.shape != (6,) or gripper.shape != (1,) or not np.isfinite(joint).all() or not np.isfinite(gripper).all():
            raise ValueError("Expected six finite joint positions and one finite gripper position")
        raw = {
            self.agentview_key: self._image(obs["external_rgb"], "external_rgb"),
            self.wrist_key: self._image(obs["wrist_rgb"], "wrist_rgb"),
            self.state_key: torch.from_numpy(np.concatenate((joint, gripper))),
            "task": str(obs["prompt"]),
        }
        batch = self.preprocessor(raw)
        # The saved preprocessor controls normalization. These tensors must also
        # live on the policy device when the checkpoint is served remotely.
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def _model_inputs(self, batch):
        images, masks = self.policy._preprocess_images(batch)
        return images, masks, batch[self.tokens_key], batch[self.token_mask_key], self.policy.prepare_state(batch)

    @torch.inference_mode()
    def get_prefix_rep(self, obs):
        images, masks, tokens, token_masks, _ = self._model_inputs(self._batch(obs))
        core = self.policy.model
        embeddings, pad_masks, att_masks = core.embed_prefix(images, masks, tokens, token_masks)
        attention = self.prepare_attention_masks_4d(self.make_att_2d_masks(pad_masks, att_masks))
        positions = torch.cumsum(pad_masks, dim=1) - 1
        core.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        (hidden, _), _ = core.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=False,
        )
        feature = hidden[:, -1, :].float().cpu().numpy()
        if feature.shape != (1, self.feature_dim):
            raise ValueError(f"Unexpected pi0 prefix shape: {feature.shape}")
        return feature

    @torch.inference_mode()
    def infer(self, obs, noise):
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (1, self.horizon, self.noise_dim) or not np.isfinite(noise).all():
            raise ValueError(f"Expected finite noise (1, {self.horizon}, {self.noise_dim}), got {noise.shape}")
        images, masks, tokens, token_masks, state = self._model_inputs(self._batch(obs))
        noise_tensor = torch.from_numpy(noise).to(self.device)
        actions = self.policy.model.sample_actions(
            images, masks, tokens, token_masks, state, noise=noise_tensor
        )[:, :, : self.action_dim]
        actions = self.postprocessor(actions)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (1, self.horizon, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"Unexpected pi0 action shape: {actions.shape}")
        return {"actions": actions[0]}


async def serve(service, host, port):
    async def handler(websocket):
        await websocket.send(msgpack_numpy.packb(service.metadata))
        async for packet in websocket:
            try:
                message = msgpack_numpy.unpackb(packet)
                obs = dict(message["obs"])
                if message["method"] == "infer":
                    result = service.infer(obs, obs.pop("noise"))
                elif message["method"] == "get_prefix_rep":
                    result = service.get_prefix_rep(obs)
                else:
                    raise ValueError(f"Unknown policy method: {message['method']}")
                await websocket.send(msgpack_numpy.packb(result))
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close()
                break

    async with websockets.asyncio.server.serve(handler, host, port, compression=None, max_size=None):
        logging.info("Serving LeRobot pi0 on %s:%s", host, port)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local LeRobot SFT checkpoint directory")
    parser.add_argument("--agentview-key", required=True, help="Agent-view image key in checkpoint config")
    parser.add_argument("--wrist-key", required=True, help="Right-wrist image key in checkpoint config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    service = TorchPi0Service.from_checkpoint(args.checkpoint, args.agentview_key, args.wrist_key)
    asyncio.run(serve(service, args.host, args.port))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
