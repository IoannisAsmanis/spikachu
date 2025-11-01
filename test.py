import sys

sys.path.append("poyo/")

import argparse
from torch.utils.data import DataLoader, SequentialSampler, Sampler, RandomSampler
import torch
from poyo.utils import seed_everything
from poyo.data.sampler import (
    SequentialFixedWindowSampler,
    RandomFixedWindowSampler,
)
from poyo.data.dataset import Dataset, DatasetIndex
from poyo.transforms import UnitDropout, Compose
from poyo.data.collate import collate, chain, pad
import torch_optimizer as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torcheval.metrics import R2Score, Mean
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import wandb
from models.model_wrapper import model_wrapper, POYOTokenizer
from poyo.models.poyo import POYO

from utils import train, evaluate
import matplotlib.pyplot as plt
import os
import yaml
from omegaconf import OmegaConf
import multiprocessing as mp
from functools import partial
import random
from datetime import date

os.environ["WANDB_SILENT"] = "true"

# SLURM execution context
import os

# Get SLURM Job ID
job_id = os.getenv("SLURM_JOB_ID")

# Get SLURM Array Job ID (if this is part of an array job)
array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")

# Get SLURM Task ID within the array job
task_id = os.getenv("SLURM_ARRAY_TASK_ID")

# Get the process ID for the task within a job allocation
proc_id = os.getenv("SLURM_PROCID")

SLURM_CTX = {
    "job_id": job_id,
    "array_job_id": array_job_id,
    "task_id": task_id,
    "proc_id": proc_id,
}

# refactor current working directory to avoid errors on cluster
CWD_PROXY = ""  # os.getcwd() # leave empty on cluster


def main(checkpoint_path, data_idcs, uid, cfg_path, device, cfg_override=None):

    with open(cfg_path) as f:
        config_dict = yaml.safe_load(f)
        base_config = OmegaConf.create(config_dict)

    curr_dir = CWD_PROXY
    datasets = sorted(
        [
            x[:-3]
            for x in os.listdir(
                curr_dir + base_config.data.data_path + base_config.data.dandiset
            )
            if ".h5" == x[-3:]
        ]
    )
    num_electrodes = pd.read_csv(
        curr_dir
        + base_config.data.data_path
        + base_config.data.dandiset
        + "/num_electrodes.csv",
        index_col=0,
    )
    datasets = [datasets[di] for di in data_idcs]

    # Initialize a dataframe that will store the results
    results_r2 = pd.DataFrame(
        data=np.zeros((len(datasets), 4)),
        index=datasets,
        columns=["train_R2_x", "train_R2_y", "valid_R2_x", "valid_R2_y"],
    )
    results_mse = pd.DataFrame(
        data=np.zeros((len(datasets), 4)),
        index=datasets,
        columns=["train_MSE_x", "train_MSE_y", "valid_MSE_x", "valid_MSE_y"],
    )
    results_sign = pd.DataFrame(
        data=np.zeros((len(datasets), 4)),
        index=datasets,
        columns=["train_SGN_x", "train_SGN_y", "valid_SGN_x", "valid_SGN_y"],
    )

    # for session_id in datasets:
    # session_cfg = base_config
    base_config.data.session = datasets  # session_id
    seed_everything(base_config.seed)

    # %% Initialize models
    wandb.login()
    with wandb.init(
        project="spikachu",
        name=base_config.model.type
        + "_"
        + "_AND_".join([x for x in base_config.data.session]),
        config=OmegaConf.to_container(base_config, resolve=True),
        mode="disabled",
    ):
        cfg = base_config  # wandb.config
        if cfg.model.type == "POYO":
            model = POYO(
                dim=cfg.model.setup.dim,
                dim_head=cfg.model.setup.dim_head,
                num_latents=cfg.model.setup.num_latents,
                depth=cfg.model.setup.depth,
                cross_heads=cfg.model.setup.cross_heads,
                self_heads=cfg.model.setup.self_heads,
                ffn_dropout=cfg.model.setup.ffn_dropout,
                lin_dropout=cfg.model.setup.lin_dropout,
                atn_dropout=cfg.model.setup.atn_dropout,
                emb_init_scale=cfg.model.setup.emb_init_scale,
                use_memory_efficient_attn=cfg.model.setup.use_memory_efficient_attn,
                init_embeddings=False,
            ).to(device)
        else:
            model = model_wrapper(
                num_neurons=(num_electrodes.loc[cfg.data.session, "num_electrodes"]),
                cfg=cfg,
                init_embeddings=False,
            ).to(device)

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                weights_only=False,
                map_location=device,
            )
        )

        # train_tokenizer = POYOTokenizer(
        #     model.unit_tokenizer,
        #     model.session_tokenizer,
        #     latent_step=cfg.tokenizer.latent_step,
        #     num_latents_per_step=cfg.tokenizer.num_latents_per_step,
        #     using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
        #     eval=False,
        # )

        # unit_dropout = UnitDropout(
        #     min_units=cfg.dropout.min_units,
        #     mode_units=cfg.dropout.mode_units,
        #     max_units=cfg.dropout.max_units,
        #     peak=cfg.dropout.peak,
        #     M=cfg.dropout.M,
        #     max_attempts=cfg.dropout.max_attempts,
        # )

        valid_tokenizer = POYOTokenizer(
            model.unit_tokenizer,
            model.session_tokenizer,
            latent_step=cfg.tokenizer.latent_step,
            num_latents_per_step=cfg.tokenizer.num_latents_per_step,
            using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
            eval=True,
        )

        # train_tokenizer = Compose([unit_dropout, train_tokenizer])
        #
        # train_set = Dataset(
        #     root=CWD_PROXY + cfg.data.data_path,
        #     split="train",
        #     include=[
        #         {
        #             "selection": [
        #                 {
        #                 "dandiset": cfg.data.dandiset, "session": cfg.data.session[i]
        #                 }
        #                 for i in range(len(datasets))
        #             ],
        #         }
        #     ],
        #     transform=train_tokenizer,
        # )

        valid_set = [
            Dataset(
                root=CWD_PROXY + cfg.data.data_path,
                split="valid",
                include=[
                    {
                        "selection": [
                            {
                                "dandiset": cfg.data.dandiset,
                                "session": cfg.data.session[i],
                            }
                        ],
                    }
                ],
                transform=valid_tokenizer,
            )
            for i in range(len(datasets))
        ]

        # train_sampler = RandomFixedWindowSampler(
        #     interval_dict=train_set.get_sampling_intervals(),
        #     window_length=cfg.sampler.window_length,
        #     generator=torch.Generator().manual_seed(cfg.seed),
        # )
        # sampling_intervals = valid_set[0].get_sampling_intervals()
        # values = list(sampling_intervals.values())[0]
        # tot_time = sum(b - a for a, b in values)
        # print('******** ',  datasets[0] + ': ',  tot_time, ' ********')
        # return

        valid_sampler = [
            SequentialFixedWindowSampler(
                interval_dict=vs.get_sampling_intervals(),
                window_length=cfg.sampler.window_length,
                step=None,
            )
            for vs in valid_set
        ]

        # train_loader = DataLoader(
        #     dataset=train_set,
        #     batch_size=cfg.sampler.batch_size,
        #     sampler=train_sampler,
        #     collate_fn=collate,
        # )

        valid_loader = [
            DataLoader(
                dataset=vs,
                batch_size=32,
                sampler=sampler,
                collate_fn=collate,
            )
            for vs, sampler in zip(valid_set, valid_sampler)
        ]

        # optimizer = optim.Lamb(
        #     model.parameters(),
        #     lr=cfg.train.lr,
        #     weight_decay=cfg.train.weight_decay,
        #     eps=1e-8,
        # )
        #
        # scheduler = CosineAnnealingLR(
        #     optimizer, T_max=int(0.25 * cfg.train.num_epochs), eta_min=0
        # )

        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=cfg.train.lr,
        #     total_steps=cfg.train.num_epochs,
        #     pct_start=0.05
        #
        # )

        # %% Train the model & Save the results
        # try:
        # trained_net = train(
        #     model,
        #     train_loader,
        #     valid_loader,
        #     optimizer,
        #     scheduler,
        #     cfg.train.num_epochs,
        #     device,
        #     tag=getattr(cfg, "experiment_tag", "default_experiment"),
        #     session_tag="_".join([str(x) for x in data_idcs]),
        # )

        for vs in valid_loader:
            results_r2, results_mse, results_sign = evaluate(
                model,
                vs,
                "valid",
                results_r2,
                results_mse,
                results_sign,
            )

        # except Exception as error:
        #     print(f"Failed to train session {session_id}: {error}")
        #     results.loc[session_id, ["valid_R2_x", "valid_R2_y"]] = np.array(
        #         [-1, -1]
        #     )

    # # Save the model
    # torch.save(
    #     trained_net,
    #     cfg.save_path + f'{cfg.model.type}_MS_{date.today()}.pt'
    # )

    results_dir = os.path.join(cfg.save_path, "test_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_r2.to_csv(
        os.path.join(results_dir, f"{cfg.model.type}_results_r2_{uid}.csv")
    )
    results_mse.to_csv(
        os.path.join(results_dir, f"{cfg.model.type}_results_mse_{uid}.csv")
    )
    results_sign.to_csv(
        os.path.join(results_dir, f"{cfg.model.type}_results_sign_{uid}.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spikachu testing script")
    parser.add_argument(
        "--no_slurm", action="store_true", help="Disable SLURM execution context"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="Path to model weights to load"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/spikachu.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device == torch.device("cuda"):
        pass
    else:
        torch.set_num_threads(torch.get_num_threads())

    # use no-slurm to debug locally
    if args.no_slurm:
        data_idcs = [[23]]  # test on specific sessions
        groups = [
            (args.checkpoint_path, data_idcs[i], i, args.config, device)
            for i in range(len(data_idcs))
        ]
        for group in groups:
            main(*group)
    else:
        print(SLURM_CTX, flush=True)
        uid = int(SLURM_CTX["task_id"])
        if args.multi_subject:
            assert uid == 0, "Multi-subject only supports single task (uid=0)"
            all_idcs = [[x for x in range(99)]]  # | 99 session model
            print("Generating multisubject model with idcs: ", all_idcs)
            data_idcs = all_idcs[uid]
        else:
            all_idcs = [[x] for x in range(99)]  # [30, 40, 50, 62, 64, 70, 94, 95, 96]]
            data_idcs = all_idcs[uid * 10 : (uid + 1) * 10]  # all_idcs[uid]

        print(f"Data indices: {data_idcs}", flush=True)
        print(f"UID: {uid}", flush=True)
        print(
            f"Device detail: {torch.cuda.get_device_properties(0)}, {device.index}",
            flush=True,
        )

        for di in data_idcs:
            main(args.checkpoint_path, di, uid, args.config, device)
