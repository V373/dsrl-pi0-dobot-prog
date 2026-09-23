<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), across various environments, including:

- **Simulation:** Libero, Aloha  
- **Real Robot:** Franka

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Installation
1. Create a conda environment:
```
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules
```
git clone git@github.com:nakamotoo/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies
```
pip install -e .
pip install -r requirements.txt

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO

# RTX 5090: pin compatible versions after installing openpi
pip install "jax[cuda12]==0.5.1" "wandb[media]==0.19.9" "protobuf>=3.20.3,<6"
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero
```

## Training (Simulation)
Libero
```
bash examples/scripts/run_libero.sh
```
Aloha
```
bash examples/scripts/run_aloha.sh
```
### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)]

1. [On the remote server] Host pi0 droid model on your remote server
```
cd openpi && python scripts/serve_policy.py --env=DROID
```
2. [On your robot client machine] Run DSRL
```
bash examples/scripts/run_real.sh
```


## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 

## Dobot 单臂真机训练（LeRobot π₀）

先在推理机启动与 Dobot SFT 数据格式一致的 LeRobot PyTorch π₀ checkpoint（两路相机键填写 checkpoint 中的实际名称）：

```bash
python3 examples/serve_lerobot_pi0_dobot.py \
  --checkpoint /path/to/lerobot-pi0-sft \
  --agentview-key 'SFT_AGENTVIEW_KEY' --wrist-key 'SFT_RIGHT_WRIST_KEY' \
  --host 0.0.0.0 --port 8000
```

在机器人端修改 `examples/scripts/run_real_dobot.sh`，再运行 `bash examples/scripts/run_real_dobot.sh`。主要参数都在该脚本中：

- 接口与任务：`robot_factory`（默认 `examples.dobot_data_wrapper:create_robot`）、`remote_host`/`remote_port`、`instruction`，以及本机 GPU 的 `device_id`。
- 采集：Dobot 启动脚本和 launcher 默认都使用 `control_hz=25`；`max_timesteps=200` 为**每条轨迹最多控制步数**，不是实际时间超时；`query_freq=10` 为 π₀ 查询间隔，须不大于 checkpoint 的动作序列长度。
- SAC：`resize_image=128`、`batch_size=256`、`discount=0.99`、`multi_grad_step=30`；`max_steps=500000` 计 **SAC 梯度更新次数**，不是机器人步数。

一轮 online 训练：

1. 每个控制步读取外部视角、右腕视角和 6 维关节 + 1 维夹爪状态。每隔 `query_freq` 步，用两路图像拼接的像素（默认 `(1, 128, 128, 6, 1)`）和 `(1, 7+F, 1)` 状态（含 π₀ 前缀特征）生成 `(1, D)` 噪声：首条轨迹随机采样，随后由 SAC 输出。噪声扩展为 `(1, H, D)` 后交给 π₀，得到 `(H, 7)` 动作序列并逐步执行。`H/D/F` 由服务从 checkpoint 读取。
2. 达到步数上限或按 `q`（第 0 步忽略）后，机器人停止；操作员输入 `1`/`0`，采集末尾观测、保存视频并重置。回放池每次 π₀ 查询存一条 SAC 噪声转移，折扣为 `discount ** 该次查询实际控制步数`；成功轨迹末步的奖励/mask 为 `(0, 0)`，其余为 `(-1, 1)`。在 `pdb` 中输入 `c` 才继续入池和训练。
3. 首条轨迹后执行 5000 次 SAC 更新；此后每条轨迹执行 `转移数 × multi_grad_step` 次，随机采样 `batch_size` 条转移更新 SAC。完整 200 控制步通常对应 20 条转移、后续 600 次更新。

SDK 接口集中在 `examples/dobot_data_wrapper.py`：补全连接与相机初始化、`_receive_observation()`、`_send_action()`、`reset()`、`halt()`、`close()`。观测需提供 `external_rgb`、`wrist_rgb`（HWC、`uint8`、RGB）、`joint_position`（6 维）和 `gripper_position`（1 维）；`step(action)` 接收单条 7 维 π₀ 动作。关节/夹爪顺序与单位、动作含义、相机预处理必须匹配 SFT checkpoint，发送动作时还需校验硬件限位与命令回执。当前模板未接入 SDK，不能直接运行真机训练。
