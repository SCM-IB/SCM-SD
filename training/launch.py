# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import os
import multiprocessing

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trainer import Trainer

# Set multiprocessing start method for Windows compatibility
if sys.platform == 'win32':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="colmap_dataset",
        help="Name of the config file (without .yaml extension, default: colmap_dataset)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    # Reduce num_workers on Windows to avoid CUDA errors
    if sys.platform == 'win32':
        if 'data' in cfg:
            if 'train' in cfg['data'] and 'num_workers' in cfg['data']['train']:
                cfg['data']['train']['num_workers'] = 0
                print(f"Reduced train num_workers to 0 for Windows compatibility")
            if 'val' in cfg['data'] and 'num_workers' in cfg['data']['val']:
                cfg['data']['val']['num_workers'] = 0
                print(f"Reduced val num_workers to 0 for Windows compatibility")

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


