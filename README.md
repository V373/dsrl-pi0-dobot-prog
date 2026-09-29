# DSRL for π₀: Diffusion Steering via Reinforcement Learning

[项目主页](https://diffusion-steering.github.io) · [论文（CoRL 2025）](https://arxiv.org/abs/2506.15799)

以下为 Dobot 部署流程：训练端运行 JAX SAC 和 PyTorch progress reward，独立 LeRobot π₀ 服务提供推理。两端可部署在同一台或不同机器，分别使用独立环境。

## Installation

需要 Linux x86_64、NVIDIA GPU、兼容 CUDA 12.8 的驱动及 Conda。分机部署时两端均需克隆仓库，后续命令均在仓库根目录执行。

```bash
git clone https://github.com/V373/dsrl-pi0-dobot-prog.git --recurse-submodules
cd dsrl-pi0-dobot-prog
```

在训练机创建环境（Python 3.11、JAX 0.5.1、Torch 2.7.1+cu128），并下载 Git LFS 模型资产（约 235 MiB）：

```bash
conda env create -f environment.dobot-sac-gpu.yml
conda activate dsrl_dobot_gpu
git lfs install --local
git lfs pull
python -m pip check
```

依赖由 [requirements.dobot-sac-gpu.txt](requirements.dobot-sac-gpu.txt) 安装，包含本地 `jaxrl2` 和 `openpi-client`。

## π₀ Server (Dobot)

在推理机创建环境并启动服务。替换 checkpoint 路径和两路相机键；checkpoint 需包含 LeRobot 的 `model.safetensors`、config 和保存的 processors，状态/动作须为与 SFT 一致的 7 维格式。

```bash
conda create -n dobot_pi0_server python=3.12 pip -y
conda activate dobot_pi0_server
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
  'lerobot[pi]==0.6.1' 'transformers==5.5.4' 'numpy==2.2.6' \
  'websockets==17.1' -e ./openpi/packages/openpi-client
python -m pip check

CUDA_VISIBLE_DEVICES=0 python examples/serve_lerobot_pi0_dobot.py \
  --checkpoint /path/to/dobot_lerobot_pi0_checkpoint \
  --agentview-key 'SFT_AGENTVIEW_KEY' --wrist-key 'SFT_RIGHT_WRIST_KEY' \
  --device cuda:0 --host 0.0.0.0 --port 8000
```

## Training (Real Dobot)

启动前完成以下准备：

- 补全 [DobotDataWrapper](examples/dobot_data_wrapper.py) 的 SDK 连接、相机采集、动作发送和启停接口；当前模板会抛出 `NotImplementedError`。相机预处理、状态/动作顺序与单位须匹配 SFT。
- 将配套的 encoder、Gaussian 和 calibration 文件放入 `shaped_reward/assets/pick_mango/ctx10/`，详见 [资产说明](shaped_reward/assets/pick_mango/README.md)。Progress 输入为原始 `480×640 RGB uint8 topFullImg`，默认裁剪区域为 `[168:392, 256:480]`。
- 确认训练机可访问 π₀ 服务的 `8000` 端口。下方 `pi0_host` 改为推理机 IP，同机部署用 `127.0.0.1`。

在训练机的交互终端执行：

```bash
conda activate dsrl_dobot_gpu
wandb login
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pi0_host=127.0.0.1
reward_ctx=10
reward_assets="$PWD/shaped_reward/assets/pick_mango/ctx${reward_ctx}"
python examples/launch_train_real_dobot.py \
  --robot_factory examples.dobot_data_wrapper:create_robot \
  --remote_host "$pi0_host" --remote_port 8000 \
  --instruction 'pick mango' --query_freq 10 \
  --reward_type pbrs --reward_device cuda:0 \
  --reward_context_stride "$reward_ctx" \
  --reward_checkpoint "$reward_assets/encoder_epoch020000.pt" \
  --reward_gaussian_h5 "$reward_assets/gaussian_progress_model.h5" \
  --reward_calibration_h5 "$reward_assets/calibration_embeddings.h5" \
  --reward_ood_threshold 0.5
```

- 默认 25 Hz、每条轨迹最多 200 个控制步。按 `q` 提前结束采集，输入成功 `1` / 失败 `0`，再在 `(Pdb)` 输入 `c` 继续训练。
- 使用 ctx20 时，将 `reward_ctx=20` 并准备 `ctx20/` 的三份配套文件，`query_freq` 保持 10。改为 `--reward_type sparse` 可关闭 progress reward，无需奖励模型文件。
- 其他参数见 `python examples/launch_train_real_dobot.py --help`，或编辑 [启动脚本](examples/scripts/run_real_dobot.sh)。

## Original OpenPI / Simulation

Franka / LIBERO / Aloha 需另外配置 [OpenPI](openpi/README.md#installation) 和 [原版依赖](requirements.txt)。入口分别为 [run_real.sh](examples/scripts/run_real.sh)、[run_libero.sh](examples/scripts/run_libero.sh)、[run_aloha.sh](examples/scripts/run_aloha.sh)；Franka 另需 [DROID](https://github.com/droid-dataset/droid)。

## Credits

基于 [jaxrl2](https://github.com/ikostrikov/jaxrl2) 和 [PTR](https://github.com/Asap7772/PTR)。原版训练日志见 [W&B](https://wandb.ai/mitsuhiko/DSRL_pi0_public)。

<details>
<summary>论文引用</summary>

```bibtex
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

</details>
