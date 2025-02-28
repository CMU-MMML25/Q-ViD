"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # Add the subset_ratio parameter with default value of 0.2 (20%)
    parser.add_argument(
        "--subset-ratio", 
        type=float, 
        default=0.2,
        help="Percentage of validation set to use for evaluation (0.0-1.0)"
    )

    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# Modify the RunnerBase class to use a custom create_loaders method
class EvalRunnerBase(RunnerBase):
    """Extension of RunnerBase with custom dataset subsetting"""
    
    def __init__(self, cfg, task, model, datasets, job_id, subset_ratio=0.2):
        super().__init__(cfg, task, model, datasets, job_id)
        self.subset_ratio = subset_ratio
        # Force dataloaders to be recreated with subset
        self._dataloaders = None
        
    def create_loaders(self, datasets, num_workers, batch_sizes, is_trains, collate_fns, dataset_ratios=None):
        """
        Create dataloaders with dataset subsetting for evaluation.
        """
        from torch.utils.data import Subset, DataLoader, DistributedSampler
        from torch.utils.data.dataset import ChainDataset
        import webdataset as wds
        from lavis.common.dist_utils import get_world_size, get_rank
        from lavis.datasets.datasets.dataloader_utils import (
            IterLoader, MultiIterLoader, PrefetchLoader
        )
        
        print(f"Creating evaluation dataloaders with subset_ratio={self.subset_ratio}")
        
        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # Create subset for non-train datasets
            if not is_train and hasattr(dataset, "__len__"):
                total_samples = len(dataset)
                # Generate indices with fixed seed for reproducibility
                indices = list(range(total_samples))
                rng = random.Random(42)  # Fixed seed
                rng.shuffle(indices)
                subset_size = max(1, int(total_samples * self.subset_ratio))
                subset_indices = indices[:subset_size]
                dataset = Subset(dataset, subset_indices)
                print(f"Created subset with {subset_size}/{total_samples} samples ({self.subset_ratio*100:.1f}%)")

            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    
    torch.cuda.empty_cache()

    # Build model with memory optimizations
    model = task.build_model(cfg)
    
    print("Finish building model")

    # Free up memory again
    torch.cuda.empty_cache()

    # Create our custom runner that will use subsets for evaluation
    runner = EvalRunnerBase(
        cfg=cfg, 
        job_id=job_id, 
        task=task, 
        model=model, 
        datasets=datasets,
        subset_ratio=args.subset_ratio
    )
    
    print("Start evaluating with subset_ratio =", args.subset_ratio)
    
    # Add explicit no_grad context
    with torch.no_grad():
        runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
