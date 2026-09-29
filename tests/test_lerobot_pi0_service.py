"""Real LeRobot 0.6.1 API smoke tests with a deliberately tiny random policy.

Run in the isolated LeRobot service environment. Set DSRL_TEST_PI0_DEVICE=cuda
to exercise CUDA. These tests validate software interfaces, not robot behavior.
Only model dimensions and the tokenizer vocabulary are reduced; checkpoint
serialization, processors, transformer layers, and flow inference are real.
"""

import asyncio
from contextlib import suppress
import json
import os
import socket

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot", minversion="0.6.1")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi0 import modeling_pi0, processor_pi0
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.processor import DeviceProcessorStep, TokenizerProcessorStep
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GemmaConfig, PaliGemmaConfig, PreTrainedTokenizerFast, SiglipVisionConfig
from openpi_client import msgpack_numpy
from websockets.asyncio.client import connect

from examples.serve_lerobot_pi0_dobot import TorchPi0Service, serve


AGENT_KEY = "observation.images.external"
WRIST_KEY = "observation.images.wrist"


def create_test_checkpoint(tmp_path, monkeypatch, dtype="float32", horizon=3):
    """Create a random, reduced-size checkpoint for interface tests only."""
    # Preserve the upstream model implementation while making the real forward
    # pass fast enough for an interface test without downloading a 3B checkpoint.
    monkeypatch.setattr(
        modeling_pi0,
        "get_gemma_config",
        lambda _: modeling_pi0.GemmaConfig(32, 1, 64, 1, 1, 32),
    )
    monkeypatch.setattr(
        modeling_pi0,
        "CONFIG_MAPPING",
        {
            "gemma": GemmaConfig,
            "paligemma": lambda: PaliGemmaConfig(
                projection_dim=32,
                vision_config=SiglipVisionConfig(
                    hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                    num_attention_heads=1, image_size=28, patch_size=14,
                ),
                text_config=GemmaConfig(
                    hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                    num_attention_heads=1, num_key_value_heads=1, head_dim=32,
                ),
            ),
        },
    )
    paligemma_model = modeling_pi0.PaliGemmaForConditionalGenerationWithPiGemma

    def tiny_paligemma(config):
        # PI0 hardcodes this projection to the production VLM width after
        # constructing its config, so resize it alongside the other dimensions.
        config.vision_config.projection_dim = 32
        return paligemma_model(config)

    monkeypatch.setattr(modeling_pi0, "PaliGemmaForConditionalGenerationWithPiGemma", tiny_paligemma)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "pick": 2, "mango": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    ).save_pretrained(tokenizer_dir)
    monkeypatch.setattr(
        processor_pi0,
        "TokenizerProcessorStep",
        lambda **kwargs: TokenizerProcessorStep(**{**kwargs, "tokenizer_name": str(tokenizer_dir)}),
    )
    config = PI0Config(
        device="cpu", dtype=dtype, chunk_size=horizon, n_action_steps=horizon,
        max_state_dim=8, max_action_dim=8, num_inference_steps=2,
        image_resolution=(28, 28), tokenizer_max_length=4,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            AGENT_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
            WRIST_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    torch.manual_seed(7)
    policy = modeling_pi0.PI0Policy(config).eval()
    checkpoint_dir = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint_dir)
    stats = {
        key: {"mean": torch.arange(7).float() / 10, "std": torch.linspace(0.5, 1.1, 7)}
        for key in ("observation.state", "action")
    }
    pre, post = processor_pi0.make_pi0_pre_post_processors(config, stats)
    pre.save_pretrained(checkpoint_dir)
    post.save_pretrained(checkpoint_dir)
    return checkpoint_dir, policy, stats


@pytest.fixture(params=["float32", "bfloat16"])
def checkpoint(tmp_path, monkeypatch, request):
    return create_test_checkpoint(tmp_path, monkeypatch, dtype=request.param)


def observation():
    return {
        "external_rgb": np.full((28, 28, 3), 100, dtype=np.uint8),
        "wrist_rgb": np.full((28, 28, 3), 180, dtype=np.uint8),
        "joint_position": np.linspace(0, 0.5, 6, dtype=np.float32),
        "gripper_position": np.array([0.2], dtype=np.float32),
        "prompt": "pick mango",
    }


async def websocket_roundtrip(service, expected_feature, expected_actions):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_task = asyncio.create_task(serve(service, "127.0.0.1", port))
    try:
        for attempt in range(100):
            try:
                client = await connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
                break
            except OSError:
                if server_task.done():
                    await server_task
                if attempt == 99:
                    raise
                await asyncio.sleep(0.01)
        async with client:
            assert msgpack_numpy.unpackb(await client.recv()) == service.metadata
            await client.send(msgpack_numpy.packb({"method": "get_prefix_rep", "obs": observation()}))
            np.testing.assert_allclose(msgpack_numpy.unpackb(await client.recv()), expected_feature)
            obs = {**observation(), "noise": np.zeros((1, 3, 8), dtype=np.float32)}
            await client.send(msgpack_numpy.packb({"method": "infer", "obs": obs}))
            np.testing.assert_allclose(msgpack_numpy.unpackb(await client.recv())["actions"], expected_actions)
            obs["noise"] = np.zeros((1, 3, 7), dtype=np.float32)
            await client.send(msgpack_numpy.packb({"method": "infer", "obs": obs}))
            assert "Expected finite noise" in await client.recv()
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task


def test_real_checkpoint_processors_and_inference(checkpoint):
    checkpoint_dir, original_policy, stats = checkpoint
    device = os.environ.get("DSRL_TEST_PI0_DEVICE", "cpu")
    # Both saved processors may contain the SFT machine's now-invalid device.
    # Reloading must honor the inference device without changing normalization.
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = checkpoint_dir / name
        data = json.loads(path.read_text())
        for step in data["steps"]:
            if step.get("registry_name") == "device_processor":
                step["config"]["device"] = "cuda:99"
        path.write_text(json.dumps(data))
    service = TorchPi0Service.from_checkpoint(checkpoint_dir, AGENT_KEY, WRIST_KEY, device=device)
    assert service.device.type == torch.device(device).type
    assert service.policy.config.device == device
    assert service.metadata == {
        "model_type": "pi0_flow", "action_horizon": 3, "noise_dim": 8,
        "feature_dim": 32, "robot_action_dim": 7,
    }
    for key, tensor in original_policy.state_dict().items():
        torch.testing.assert_close(service.policy.state_dict()[key].cpu(), tensor)
    batch = service._batch(observation())
    assert all(value.device == service.device for value in batch.values() if isinstance(value, torch.Tensor))
    assert next(step for step in service.postprocessor.steps if isinstance(step, DeviceProcessorStep)).tensor_device.type == "cpu"
    zeros = torch.zeros((1, 3, 7), device=service.device)
    unnormalized = service.postprocessor(zeros)
    torch.testing.assert_close(unnormalized, stats["action"]["mean"].expand(1, 3, 7))
    feature = service.get_prefix_rep(observation())
    noise = np.zeros((1, 3, 8), dtype=np.float32)
    actions = service.infer(observation(), noise)["actions"]
    assert feature.shape == (1, 32) and np.isfinite(feature).all()
    assert actions.shape == (3, 7) and np.isfinite(actions).all()
    assert actions.dtype == np.float32
    np.testing.assert_allclose(service.infer(observation(), noise)["actions"], actions)
    with pytest.raises(ValueError, match="Expected finite noise"):
        service.infer(observation(), np.zeros((1, 3, 7), dtype=np.float32))
    asyncio.run(websocket_roundtrip(service, feature, actions))


def test_corrupt_checkpoint_is_rejected(checkpoint):
    checkpoint_dir, _, _ = checkpoint
    weights = checkpoint_dir / "model.safetensors"
    state = load_file(str(weights))
    del state["model.action_out_proj.weight"]
    save_file(state, str(weights))
    with pytest.raises(RuntimeError, match="Missing key"):
        TorchPi0Service.from_checkpoint(checkpoint_dir, AGENT_KEY, WRIST_KEY, device="cpu")


def test_explicit_cuda_never_silently_falls_back(checkpoint, monkeypatch):
    checkpoint_dir, _, _ = checkpoint
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="PyTorch CUDA is unavailable"):
        TorchPi0Service.from_checkpoint(checkpoint_dir, AGENT_KEY, WRIST_KEY, device="cuda")
