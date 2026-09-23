#!/usr/bin/env python3
"""Parse Dobot run settings and start the real-robot training loop."""

import argparse
import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.train_real_dobot import main
from jaxrl2.utils.general_utils import AttrDict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_factory", "--robot-factory", default="examples.dobot_data_wrapper:create_robot")
    parser.add_argument("--remote_host", "--remote-host", default="127.0.0.1")
    parser.add_argument("--remote_port", "--remote-port", type=int, default=8000)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--control_hz", "--control-hz", type=float, default=25.0)
    parser.add_argument("--max_timesteps", "--max-timesteps", type=int, default=200,
                        help="Maximum robot control steps in one rollout, not a wall-clock timeout")
    parser.add_argument("--query_freq", "--query-freq", type=int, default=10)
    parser.add_argument("--resize_image", "--resize-image", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=256)
    parser.add_argument("--max_steps", "--max-steps", type=int, default=500000)
    parser.add_argument("--multi_grad_step", "--multi-grad-step", type=int, default=30)
    parser.add_argument("--log_interval", "--log-interval", type=int, default=100)
    parser.add_argument("--eval_interval", "--eval-interval", type=int, default=2000)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--action_magnitude", "--action-magnitude", type=float, default=2.5)
    parser.add_argument("--hidden_dims", "--hidden-dims", type=int, nargs="+", default=[1024])
    parser.add_argument("--num_qs", "--num-qs", type=int, default=2)
    parser.add_argument("--prefix", default="dsrl_pi0_dobot")
    parser.add_argument("--wandb_project", "--wandb-project", default="DSRL_pi0_Dobot")
    return AttrDict(vars(parser.parse_args(argv)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args())
