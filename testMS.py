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
import re

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


def main(data_idcs, uid, device):

    with open("./configs/configs_spikingjelly_snn.yaml") as f:
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
    )  # '/../bio_spikes/processed/000688'
    num_electrodes = pd.read_csv(
        curr_dir
        + base_config.data.data_path
        + base_config.data.dandiset
        + "/num_electrodes.csv",
        index_col=0,
    )

    datasets = [datasets[di] for [di] in data_idcs]

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
    cfg = base_config  # wandb.config

    folder_with_models = "/home/gment/poyo/trained_models/spikachu_1000_epochs"
    split = "test"

    state_dicts = [
        folder_with_models + "/" + x
        for x in sorted(
            os.listdir(folder_with_models),
            key=lambda s: int(re.search(r"SpikingJellySNN_(\d+)", s).group(1)),
        )
    ]

    assert len(datasets) == len(
        state_dicts
    ), "Lenght of datasets and models is not the same - check for duplicate saved models"
    # datasets, state_dicts = [datasets[i] for i in [86, 87]], [state_dicts[i] for i in [86, 87]]
    for dataset, state_dict in zip(datasets, state_dicts):
        try:
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
                    num_neurons=(num_electrodes.loc[[dataset], "num_electrodes"]),
                    cfg=cfg,
                    init_embeddings=False,
                ).to(device)

            model.load_state_dict(
                torch.load(state_dict, map_location=torch.device(device))
            )

            valid_tokenizer = POYOTokenizer(
                model.unit_tokenizer,
                model.session_tokenizer,
                latent_step=cfg.tokenizer.latent_step,
                num_latents_per_step=cfg.tokenizer.num_latents_per_step,
                using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
                eval=True,
            )

            valid_set = Dataset(
                root=CWD_PROXY + cfg.data.data_path,
                split=split,
                include=[
                    {
                        "selection": [
                            {"dandiset": cfg.data.dandiset, "session": dataset}
                        ],
                    }
                ],
                transform=valid_tokenizer,
            )

            valid_sampler = SequentialFixedWindowSampler(
                interval_dict=valid_set.get_sampling_intervals(),
                window_length=cfg.sampler.window_length,
                step=None,
            )

            valid_loader = DataLoader(
                dataset=valid_set,
                batch_size=128,
                sampler=valid_sampler,
                collate_fn=collate,
            )

            results_r2, results_mse, results_sign = evaluate(
                model,
                valid_loader,
                "valid",
                results_r2,
                results_mse,
                results_sign,
            )
        except:
            print(
                f"\n ******************** Skipping: {dataset}. *********************************** \n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train POYO model")
    parser.add_argument(
        "--no_slurm", action="store_true", help="Disable SLURM execution context"
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

        # data_idcs = [[i] for i in range(99)]
        # data_idcs = [[2], [5], [11], [19], [23], [27], [30], [34], [38], [41], [50], [53], [57], [60], [64], [70], [72], [80], [85], [96]]
        # data_idcs = [[0], [2], [4], [6], [8], [10], [12], [14], [16], [18], [20], [22], [24], [26], [28], [30], [32], [34], [36], [38], [40], [42], [44], [46], [48], [50], [52], [54], [56], [58], [60], [62], [64], [66], [68], [70], [72], [74], [76], [78], [80], [82], [84], [86], [88], [90], [92], [94], [96]]
        # data_idcs = [[0], [2], [4], [5], [6], [7], [8], [9], [10], [11], [13], [15], [17], [18], [19], [20], [21], [22], [23], [24], [26], [27], [28], [29], [30], [31], [32], [33], [34], [36], [37], [39], [40], [42], [44], [45], [46], [48], [49], [52], [53], [54], [55], [56], [57], [58], [59], [62], [63], [64], [67], [68], [69], [70], [71], [74], [75], [77], [79], [80], [81], [82], [83], [84], [87], [88], [90], [91], [92], [93], [94], [96], [97], [98]]
        # data_idcs = [[i] for i in range(12)]
        data_idcs = [[i] for i in range(99)]

        # groups = [(data_idcs[i], i, device) for i in range(len(data_idcs))]

        groups = [(data_idcs, 0, device)]

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
        # data_idcs = np.arange(
        #     uid * single_node_width,
        #     min((uid + 1) * single_node_width, N_FILES_TOTAL),
        # )
        # all_idcs = [[30], [40], [50], [62], [64], [70], [94], [95], [96]]
        all_idcs = [[30, 31], [40, 41]]

        data_idcs = all_idcs[uid]

        print(f"Data indices: {data_idcs}", flush=True)
        print(f"UID: {uid}", flush=True)
        print(
            f"Device detail: {torch.cuda.get_device_properties(0)}, {device.index}",
            flush=True,
        )

        main(data_idcs, uid, device)
