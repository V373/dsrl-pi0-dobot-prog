#!/usr/bin/env bash
set -euo pipefail

# Start serve_lerobot_pi0_dobot.py on the remote host before this client.
proj_name=DSRL_pi0_Dobot
device_id=0
robot_factory=examples.dobot_data_wrapper:create_robot
remote_host=127.0.0.1
remote_port=8000
instruction="TODO: replace with the SFT task instruction"
control_hz=25
max_timesteps=200

export EXP="./logs/$proj_name"
export CUDA_VISIBLE_DEVICES="$device_id"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 examples/launch_train_real_dobot.py \
  --robot_factory "$robot_factory" \
  --remote_host "$remote_host" \
  --remote_port "$remote_port" \
  --instruction "$instruction" \
  --control_hz "$control_hz" \
  --max_timesteps "$max_timesteps" \
  --prefix dsrl_pi0_dobot \
  --wandb_project "$proj_name" \
  --batch_size 256 \
  --discount 0.99 \
  --seed 0 \
  --max_steps 500000 \
  --eval_interval 2000 \
  --log_interval 100 \
  --multi_grad_step 30 \
  --resize_image 128 \
  --action_magnitude 2.5 \
  --query_freq 10 \
  --hidden_dims 1024 \
  --num_qs 2
