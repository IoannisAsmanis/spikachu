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
from poyo.models.poyo import POYO  # , POYOTokenizer

from utils import train, evaluate, count_trainable_parameters
import matplotlib.pyplot as plt
import os
import yaml
from omegaconf import OmegaConf
import multiprocessing as mp
from functools import partial
import random
from datetime import date
from itertools import product
from collections import defaultdict

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


def main(data_idcs, uid, cfg_path, device, cfg_override=None):

    with open(cfg_path) as f:
        config_dict = yaml.safe_load(f)
        base_config = OmegaConf.create(config_dict)
        if cfg_override is not None:
            base_config = OmegaConf.merge(base_config, cfg_override)

    print(f"Base config retrieval complete: {base_config}", flush=True)

    curr_dir = CWD_PROXY
    datasets = sorted(
        [
            x[:-3]
            for x in os.listdir(
                curr_dir + base_config.data.data_path + base_config.data.dandiset
            )
            if ".h5" == x[-3:]
        ]
    )  # '/../bio_spikes/processed/000688'
    num_electrodes = pd.read_csv(
        curr_dir
        + base_config.data.data_path
        + base_config.data.dandiset
        + "/num_electrodes.csv",
        index_col=0,
    )

    if base_config.data.lock_data_idcs is not None:
        data_idcs = list(base_config.data.lock_data_idcs)

    datasets = [datasets[di] for di in data_idcs]

    print(f"Data indices: {data_idcs}", flush=True)
    print(f"Corresponding datasets: {datasets}", flush=True)

    # # Initialize a dataframe that will store the results
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
        project="poyo",
        name=base_config.model.type
        + "_"
        + "_AND_".join([x for x in base_config.data.session]),
        config=OmegaConf.to_container(base_config, resolve=True),
        mode=base_config.wandb_mode,
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
            ).to(device)
        else:
            if cfg.model.setup.pre_trained_path is None:
                model = model_wrapper(
                    num_neurons=(
                        num_electrodes.loc[cfg.data.session, "num_electrodes"]
                    ),
                    cfg=cfg,
                    init_embeddings=True,
                ).to(device)

                model.snn_parameter_analysis()

            else:
                model = model_wrapper(
                    num_neurons=(
                        num_electrodes.loc[cfg.data.session, "num_electrodes"]
                    ),
                    cfg=cfg,
                    init_embeddings=False,
                ).to(device)

                model.snn_parameter_analysis()

                print("Loading weights from: ", cfg.model.setup.pre_trained_path)
                model.load_state_dict(
                    torch.load(
                        cfg.model.setup.pre_trained_path,
                        weights_only=False,
                        map_location=device,
                    )
                )

                if cfg.model.setup.finetune_with_unit_identification:
                    assert cfg.model.setup.pre_trained_path is not None
                    print(
                        "Finetuning with unit identification: ",
                        cfg.model.setup.pre_trained_path,
                    )
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.unit_emb.parameters():
                        param.requires_grad = True

        print(
            f"Model initialized with {count_trainable_parameters(model)} trainable parameters."
        )

        train_tokenizer = POYOTokenizer(
            model.unit_tokenizer,
            model.session_tokenizer,
            latent_step=cfg.tokenizer.latent_step,
            num_latents_per_step=cfg.tokenizer.num_latents_per_step,
            using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
            eval=False,
        )

        unit_dropout = UnitDropout(
            min_units=cfg.dropout.min_units,
            mode_units=cfg.dropout.mode_units,
            max_units=cfg.dropout.max_units,
            peak=cfg.dropout.peak,
            M=cfg.dropout.M,
            max_attempts=cfg.dropout.max_attempts,
        )

        valid_tokenizer = POYOTokenizer(
            model.unit_tokenizer,
            model.session_tokenizer,
            latent_step=cfg.tokenizer.latent_step,
            num_latents_per_step=cfg.tokenizer.num_latents_per_step,
            using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
            eval=True,
        )

        test_tokenizer = POYOTokenizer(
            model.unit_tokenizer,
            model.session_tokenizer,
            latent_step=cfg.tokenizer.latent_step,
            num_latents_per_step=cfg.tokenizer.num_latents_per_step,
            using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
            eval=True,
        )

        train_tokenizer = Compose([unit_dropout, train_tokenizer])

        train_set = Dataset(
            root=CWD_PROXY + cfg.data.data_path,
            split="train",
            include=[
                {
                    "selection": [
                        {"dandiset": cfg.data.dandiset, "session": cfg.data.session[i]}
                        for i in range(len(datasets))
                    ],
                }
            ],
            transform=train_tokenizer,
        )

        valid_set = Dataset(
            root=CWD_PROXY + cfg.data.data_path,
            split="valid",
            include=[
                {
                    "selection": [
                        {"dandiset": cfg.data.dandiset, "session": cfg.data.session[i]}
                        for i in range(len(datasets))
                    ],
                }
            ],
            transform=valid_tokenizer,
        )

        test_set = Dataset(
            root=CWD_PROXY + cfg.data.data_path,
            split="test",
            include=[
                {
                    "selection": [
                        {"dandiset": cfg.data.dandiset, "session": cfg.data.session[i]}
                        for i in range(len(datasets))
                    ],
                }
            ],
            transform=test_tokenizer,
        )

        train_sampler = RandomFixedWindowSampler(
            interval_dict=train_set.get_sampling_intervals(),
            window_length=cfg.sampler.window_length,
            generator=torch.Generator().manual_seed(cfg.seed),
        )

        valid_sampler = SequentialFixedWindowSampler(
            interval_dict=valid_set.get_sampling_intervals(),
            window_length=cfg.sampler.window_length,
            step=None,
        )

        test_sampler = SequentialFixedWindowSampler(
            interval_dict=test_set.get_sampling_intervals(),
            window_length=cfg.sampler.window_length,
            step=None,
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=cfg.sampler.batch_size,
            sampler=train_sampler,
            collate_fn=collate,
        )

        # valid_loader = DataLoader(
        #     dataset=valid_set,
        #     batch_size=32,
        #     sampler=valid_sampler,
        #     collate_fn=collate,
        #     drop_last=True,
        # )

        # Switched to test set for validation
        valid_loader = DataLoader(
            dataset=test_set,
            batch_size=32,
            sampler=test_sampler,
            collate_fn=collate,
        )

        optimizer = optim.Lamb(
            filter(
                lambda p: p.requires_grad, model.parameters()
            ),  # model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            eps=1e-8,
        )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=int(0.25 * cfg.train.num_epochs), eta_min=0
        )

        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=cfg.train.lr,
        #     total_steps=cfg.train.num_epochs,
        #     pct_start=0.05
        #
        # )

        # %% Train the model & Save the results
        # try:
        trained_net = train(
            model,
            train_loader,
            valid_loader,
            optimizer,
            scheduler,
            cfg.train.num_epochs,
            device,
            tag=getattr(cfg, "experiment_tag", "default_experiment"),
            session_tag="_".join([str(x) for x in data_idcs]),
            plot_data=False,
        )

        results_r2, results_mse, results_sign = evaluate(
            trained_net,
            valid_loader,
            "valid",
            results_r2,
            results_mse,
            results_sign,
            plot_data=True,
        )

        # results_r2, results_mse, results_sign = evaluate(
        #     trained_net,
        #     test_loader,
        #     "test",
        #     results_r2,
        #     results_mse,
        #     results_sign,
        #     plot_data=True,
        # )

        # except Exception as error:
        #     print(f"Failed to train session {session_id}: {error}")
        #     results.loc[session_id, ["valid_R2_x", "valid_R2_y"]] = np.array(
        #         [-1, -1]
        #     )

    # results_dir = os.path.join(cfg.save_path, "results")
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    # results_r2.to_csv(
    #     os.path.join(results_dir, f"{cfg.model.type}_results_r2_{uid}.csv")
    # )
    # results_mse.to_csv(
    #     os.path.join(results_dir, f"{cfg.model.type}_results_mse_{uid}.csv")
    # )
    # results_sign.to_csv(
    #     os.path.join(results_dir, f"{cfg.model.type}_results_sign_{uid}.csv")
    # )


def nest_dict(flat_dict):
    """
    Convert a flat dictionary with dotted keys into a nested dictionary.
    """

    nested_dict = defaultdict(dict)
    for key, value in flat_dict.items():
        keys = key.split(".")
        d = nested_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested_dict


def multi_setting_manager(data_idcs, uid, cfg_path, device):
    """
    Run a gauntlet of experiments after overriding model configurations.
    """

    hidden_sizes = [256, 384, 512]
    n_branch_scales = [1, 3, 5, 7, 9]
    n_branch_layers = [1, 2, 3, 4, 5]
    n_layer_skips = [-1, 2]
    n_integrator_scales = [1, 3, 5, 7, 9]

    all_cfgs = product(
        hidden_sizes,
        n_branch_scales,
        n_branch_layers,
        n_layer_skips,
        n_integrator_scales,
    )

    for cfg in all_cfgs:

        # don't run if n_layer_skips > n_branch_layers
        if cfg[2] < cfg[3]:
            continue

        cfg_override_flat = {
            "model.setup.hidden_size": cfg[0],
            "model.setup.block_cfg.n_scales": cfg[1],
            "model.setup.block_cfg.branch_cfg.n_layers": cfg[2],
            "model.setup.block_cfg.branch_cfg.n_layer_skip": cfg[3],
            "model.setup.integrator_cfg.n_scales": cfg[4],
        }
        cfg_override_dict = nest_dict(cfg_override_flat)
        cfg_override = OmegaConf.create(dict(cfg_override_dict))

        print("*" * 80, flush=True)
        print("--- START ---", flush=True)
        print(f"Running experiment with config override: {cfg_override}", flush=True)
        main(data_idcs, uid, cfg_path, device, cfg_override)
        print("--- DONE ---", flush=True)
        print("*" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train POYO model")
    parser.add_argument(
        "--no_slurm", action="store_true", help="Disable SLURM execution context"
    )
    parser.add_argument(
        "--multi_subject", action="store_true", help="Run multi-subject model"
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

    if args.no_slurm:
        # mp.set_start_method("spawn")

        # Create a list of dataset indices and shuffle them randomly
        # num_datasets, num_processes = 99, 3
        # all_indices = list(range(num_datasets))
        # random.shuffle(all_indices)
        #
        # # Split the shuffled indices into approximately equal groups
        # data_idcs = [
        #     all_indices[
        #         i * (num_datasets // num_processes)
        #         + min(i, num_datasets % num_processes) : (i + 1)
        #         * (num_datasets // num_processes)
        #         + min(i + 1, num_datasets % num_processes)
        #     ]
        #     for i in range(num_processes)
        # ]

        # Keep only 3 datasets, one from each monkey
        # num_processes = 10
        # data_idcs = [[30, 40, 50, 62, 64, 70, 94, 95, 96]]

        # c_20150313_center_out_reaching
        data_idcs = [[23]]

        groups = [(data_idcs[i], i, args.config, device) for i in range(len(data_idcs))]

        # # Train the models on each dataset in parallel (as long as memory doesn't crash).
        # # res = main([0], 0, torch.device('cuda'))
        # with mp.Pool(processes=num_processes) as pool:
        #     # Use starmap_async and capture the result object
        #     async_results = pool.starmap_async(main, groups)
        #
        #     # Call .get() to ensure the main process waits for all async processes to finish
        #     async_results.get()
        for group in groups:
            main(*group)
    else:
        print(SLURM_CTX, flush=True)

        # N_FILES_TOTAL = 9
        # single_node_width = 1
        uid = int(SLURM_CTX["task_id"])
        print(uid)
        if args.multi_subject:
            assert uid == 0
            # all_idcs = [[2,5,11,19,23,27,30,34,38,41,50,53,57,60,64,70,72,80,85,96]]  # 8, 15, 45, 76, 90 | 20 subject model
            # all_idcs = [[x for x in range(0, 98, 2)]]  # | 50 subject model
            # all_idcs = [
            #     [x for x in range(99) if x not in [1, 3, 12, 25, 14, 16, 35, 38, 41, 43, 47, 50, 51, 60, 61, 65, 66, 73, 72, 78, 76, 85, 86, 89, 95]]
            # ]  # | 74 subject model
            all_idcs = [[x for x in range(99)]]  # | 99 session model
            print("Generating multisubject model with idcs: ", all_idcs)
            data_idcs = all_idcs[uid]
        else:
            all_idcs = [[x] for x in range(99)]  # [30, 40, 50, 62, 64, 70, 94, 95, 96]]
            data_idcs = all_idcs[uid * 10 : (uid + 1) * 10]  # all_idcs[uid]

            # data_idcs = [[30]]

        print(f"Data indices: {data_idcs}", flush=True)
        print(f"UID: {uid}", flush=True)
        print(
            f"Device detail: {torch.cuda.get_device_properties(0)}, {device.index}",
            flush=True,
        )

        for di in data_idcs:
            main(di, uid, args.config, device)
        # multi_setting_manager(data_idcs, uid, device)
