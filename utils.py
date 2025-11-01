import time
import pickle
import torch
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

from collections import defaultdict
import wandb
from torcheval.metrics import R2Score, Mean
import os
from pathlib import Path
import copy
from datetime import date
import sys
from spikingjelly.activation_based import functional
from matplotlib.ticker import MultipleLocator


def sign_agreement_metric(gt, pred, reduction="mean"):
    gt_signs = np.sign(gt)
    pred_signs = np.sign(pred)
    aggreed_signs = gt_signs == pred_signs
    if reduction == "mean":
        return aggreed_signs.astype(float).mean()
    elif reduction == "sum":
        return aggreed_signs.sum()
    elif reduction == "none":
        return aggreed_signs
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")


def evaluate(
    net, loader, split, results_r2, results_loss, results_sign, plot_data=True
):
    # TODO: Session ID incorrect when training on data from multiple subjects.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = Mean(device=device)
    total_r2 = R2Score(device=device, multioutput="raw_values")
    with torch.no_grad():
        net.eval()
        total_r2.reset()
        total_loss.reset()
        OnlyWithinTrials = True
        for i, data in enumerate(loader):
            # Load the data to device
            session_id = data["session_id"][0][7:]
            data = {
                key: data[key]
                for key in data
                if key not in ["session_id", "absolute_start", "output_subtask_index"]
            }
            data = {
                k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in data.items()
            }

            # Forward Pass
            predicted_values, loss, r2 = net(**data)

            # Make some plots, for when data is within trials only.

            is_within_trial = torch.where(data["output_weights"] == 5)[0]
            if is_within_trial.numel() == 0:
                is_within_trial = torch.where(
                    torch.logical_or(
                        data["output_weights"] == 5, data["output_weights"] == 1
                    )
                )[0]
                OnlyWithinTrials = False

            # Update the R2 score & Loss
            total_r2.update(
                predicted_values[is_within_trial],
                data["output_values"][is_within_trial],
            )
            total_loss.update(loss)

            if hasattr(net, "cfg") and "snn" in net.cfg.model.type.lower():
                # need to reset the membrane potentials
                if "jelly" in net.cfg.model.type.lower():
                    functional.reset_net(net.model)
                else:
                    net.model.reset_mem()

            if hasattr(net, "cfg") and "spikinghomogenizer" in net.cfg.model.setup:
                functional.reset_net(net.homogenizer)

            if plot_data:
                # Group data into "reaching movements"
                diffs = torch.diff(
                    is_within_trial, prepend=is_within_trial[:1] - 1
                )  # Compute differences
                group_starts = torch.where(diffs > 1)[0]  # Start of new groups
                group_ends = torch.cat(
                    (
                        group_starts[1:],
                        torch.tensor(
                            [len(is_within_trial)], device=group_starts.device
                        ),
                    )
                )  # End of groups
                subgroups = [
                    is_within_trial[start:end]
                    for start, end in zip(group_starts, group_ends)
                ]

                if not subgroups:
                    # subgroups = [is_within_trial]
                    subgroups = [
                        is_within_trial[i : i + 100]
                        for i in range(0, len(is_within_trial), 100)
                    ]

                # Determine subplot grid dimensions
                # num_subgroups = len(subgroups)
                # cols = 4  # Maximum columns per row
                # rows = max((num_subgroups + cols - 1) // cols, 1)  # Calculate number of rows needed

                # if rows <= 0:
                #     continue

                # # Set up subplots
                # fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True, dpi=300)

                # # Flatten axes for easy iteration (if only one row, make axes iterable)
                # axes = axes.flatten()

                # Plot each subgroup
                for ii, subgroup in enumerate(subgroups):
                    true_ = data["output_values"][subgroup].detach().cpu().numpy()
                    pred_ = predicted_values[subgroup].detach().cpu().numpy()
                    # discontinuities = np.where(np.abs(np.diff(true_, axis=0)) > 0.1)[0]

                    fig, ax = plt.subplots(
                        2, 1, figsize=(6.4, 3.2), sharex=True, dpi=300
                    )
                    axes = ax.flatten()
                    # subgroup_points = predicted_values[subgroup]  # Extract points in this subgroup

                    ax[0].plot(
                        [x / 100 for x in range(len(true_[:, 0]))],
                        true_[:, 0],
                        "k",
                        label="Ground Truth",
                        linewidth=2.5,
                    )
                    ax[1].plot(
                        [x / 100 for x in range(len(true_[:, 0]))],
                        true_[:, 1],
                        "k",
                        linewidth=2.5,
                    )
                    ax[0].plot(
                        [x / 100 for x in range(len(true_[:, 0]))],
                        pred_[:, 0],
                        "teal",
                        alpha=0.7,
                        label="Spikachu",
                        linewidth=2.5,
                    )
                    ax[1].plot(
                        [x / 100 for x in range(len(true_[:, 0]))],
                        pred_[:, 1],
                        "teal",
                        alpha=0.7,
                        linewidth=2.5,
                    )
                    plt.suptitle(
                        f"Batch: {i} | R2: {max(0, r2_score(true_, pred_)):.2f}",
                        fontsize=6,
                    )
                    # ax.set_title(f"Batch: {i} | R2: {max(0, r2_score(true_, pred_)):.2f}", fontsize=6)
                    ax[1].set_xlabel("Time (sec)")
                    ax[0].set_ylabel("$v_{x}$", fontsize=14)
                    ax[0].tick_params(labelbottom=False)
                    ax[1].set_ylabel("$v_{y}$", fontsize=14)

                    # # Hide unused subplots
                    # for ax in axes[num_subgroups:]:
                    #     ax.axis("off")

                    for ax_ in ax:

                        ax_.spines["top"].set_visible(False)
                        ax_.spines["right"].set_visible(False)
                        ax_.spines["bottom"].set_visible(False)
                        ax_.spines["left"].set_visible(True)
                        ax_.yaxis.set_ticks_position("left")
                        ax_.yaxis.set_major_locator(MultipleLocator(1))
                        ax_.tick_params(axis="y", labelsize=14)
                        ax_.patch.set_alpha(0)
                        ax_.tick_params(axis="y", labelsize=14)
                        ax_.yaxis.set_major_locator(MultipleLocator(1))
                        ax_.tick_params(axis="x", labelsize=14)
                        ax_.tick_params(axis="y", labelsize=14)

                    # for ax_ in ax:
                    #     ax_.spines['top'].set_visible(False)
                    #     ax_.spines['right'].set_visible(False)
                    #     ax_.spines['bottom'].set_visible(False)
                    #     ax_.spines['left'].set_visible(True)  # keep y-axis spine
                    #     ax_.xaxis.set_visible(False)  # hide x-axis line
                    #     ax_.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    #     ax_.yaxis.set_ticks_position('left')  # only left ticks
                    #     ax_.yaxis.set_tick_params(width=1)

                    ax[0].xaxis.set_visible(False)  # hide x-axis line
                    ax[0].tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )
                    # ax[0].legend()

                    ax[1].spines["bottom"].set_visible(True)
                    ax[1].tick_params(axis="x", labelsize=14)

                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.subplots_adjust(hspace=0.1)
                    fig.patch.set_alpha(0)

                    os.makedirs("vis_logs", exist_ok=True)
                    fig.savefig(
                        f"vis_logs/{session_id}_{split}_batch_{i}_subgroup_{ii}_r2_score_{100*np.round(r2_score(true_, pred_), 2)}.png"
                    )
                    plt.close(fig)

                # # Set up subplots
                # fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=300)

                # # Flatten axes for easy iteration (if only one row, make axes iterable)
                # axes = axes.flatten()

                # # Plot each subgroup
                # for _, (subgroup, ax) in enumerate(zip(subgroups, axes)):
                #     # subgroup_points = predicted_values[subgroup]  # Extract points in this subgroup
                #     true_ = data["output_values"][subgroup].detach().cpu().numpy()
                #     pred_ = predicted_values[subgroup].detach().cpu().numpy()
                #     ax.plot(true_[:, 0], 'b')
                #     ax.plot(true_[:, 1], 'm')
                #     ax.plot(pred_[:, 0], 'k', alpha=0.7)
                #     ax.plot(pred_[:, 1], 'k', alpha=0.7)
                #     ax.set_title(f"Batch: {i} | R2: {max(0, r2_score(true_, pred_)):.2f}", fontsize=6)
                #     ax.set_xlabel("Sample")
                #     ax.set_ylabel("Vel (a.u.)")

                # # Hide unused subplots
                # for ax in axes[num_subgroups:]:
                #     ax.axis("off")

                # plt.tight_layout()
                # # plt.show()

                # os.makedirs('vis_logs', exist_ok=True)
                # fig.savefig(f"vis_logs/{session_id}_{split}_batch_{i}")
                # plt.close(fig)

        # Show the validation Loss
        print(
            f"[{session_id}], {split}, Loss: {total_loss.compute():.4f}, r2: {total_r2.compute().mean():.2f}, OnlyWithinTrials: {OnlyWithinTrials}"
        )
        results_r2.loc[session_id, ["valid_R2_x", "valid_R2_y"]] = (
            total_r2.compute().detach().cpu().numpy()
        )
        results_loss.loc[session_id, ["valid_MSE_x", "valid_MSE_y"]] = (
            total_loss.compute().detach().cpu().numpy()
        )
        results_sign.loc[session_id, ["valid_SGN_x", "valid_SGN_y"]] = (
            sign_agreement_metric(
                data["output_values"][:, 0].detach().cpu().numpy(),
                predicted_values[:, 0].detach().cpu().numpy(),
            ),
            sign_agreement_metric(
                data["output_values"][:, 1].detach().cpu().numpy(),
                predicted_values[:, 1].detach().cpu().numpy(),
            ),
        )
        return results_r2, results_loss, results_sign


def train(
    net,
    trainloader,
    validloader,
    optimizer,
    scheduler,
    epochs,
    device,
    tag=None,
    session_tag="default_session",
    plot_data=True,
):

    steps = 0
    best_val_loss = None
    spike_params = defaultdict(list)
    for epoch in range(epochs):
        # r2_train = R2Score(device=device, multioutput="raw_values")
        # train_loss = Mean(device=device)
        # for i, data in enumerate(trainloader):
        #     net.train()
        #     data = {
        #         k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
        #         for k, v in data.items()
        #     }

        #     # Forward Pass
        #     predicted_values, loss, r2 = net(**data)

        #     # Recover shape information
        #     B = data["spike_unit_index"].shape[0]
        #     BT, C = data["output_values"].shape
        #     T = BT // B

        #     # Compute the R2 score across the training data
        #     r2_train.update(
        #         predicted_values, data["output_values"][: predicted_values.shape[0]]
        #     )
        #     train_loss.update(loss)

        #     # Backward Pass
        #     optimizer.zero_grad()
        #     loss.backward(retain_graph=False)
        #     optimizer.step()
        #     steps += 1

        #     if hasattr(net, "cfg") and "snn" in net.cfg.model.type.lower():
        #         # need to reset the membrane potentials
        #         if "jelly" in net.cfg.model.type.lower():
        #             functional.reset_net(net.model)
        #         else:
        #             net.model.reset_mem()

        # if hasattr(net, "get_spike_params"):
        #     all_sp = net.get_spike_params()  # ffn_sp, ms_sp, int_sp
        #     spike_params["ffn_sp"].append(all_sp[0])
        #     spike_params["ms_sp"].append(all_sp[1])
        #     spike_params["int_sp"].append(all_sp[2])

        # # try:
        # print(
        #     f"[TRAIN] Total Steps: {steps}, Epoch: {epoch}, Loss: {train_loss.compute():.4f},  r2: {r2_train.compute().mean():.2f}",
        #     flush=True,
        # )
        # wandb.log(
        #     {
        #         "train_Loss": train_loss.compute(),
        #         "train_R2": r2_train.compute().mean(),
        #         "LR": optimizer.param_groups[0]["lr"],
        #     },
        #     step=steps,
        # )
        # # except:
        # #     print("Error in logging!", flush=True)

        if epoch % 25 == 0:
            if plot_data:
                if hasattr(net, "cfg") and "homogenizer" in net.cfg.model.setup:
                    data_in = net.identify_binned_spiked_units(
                        spike_unit_index=data["spike_unit_index"],
                        spike_timestamps=data["spike_timestamps"],
                        input_mask=data["input_mask"],
                        spike_type=data["spike_type"],
                    ).permute(1, 0, 2)
                elif hasattr(net, "cfg"):
                    data_in = net.create_spike_tensor(
                        spike_unit_index=data["spike_unit_index"],
                        spike_timestamps=data["spike_timestamps"],
                        input_mask=data["input_mask"],
                        spike_type=data["spike_type"],
                    ).permute(1, 0, 2)

                    actual_b = data_in.shape[1]
                    actual_t = data_in.shape[0]
                    labels = data["output_values"][
                        : actual_b * actual_t
                    ]  # off-by-one error?!
                else:
                    sys.exit("Please set plot_data=False when using POYO.")
                # more off-by-one ridiculousness
                if labels.shape[0] < actual_b * actual_t:
                    labels = torch.cat(
                        [
                            labels,
                            torch.zeros(actual_b * actual_t - labels.shape[0], C).to(
                                labels.device
                            ),
                        ]
                    )
                if predicted_values.shape[0] < actual_b * actual_t:
                    predicted_values = torch.cat(
                        [
                            predicted_values,
                            torch.zeros(
                                actual_b * actual_t - predicted_values.shape[0], C
                            ).to(predicted_values.device),
                        ]
                    )
                # plot data
                labels = labels.reshape(actual_b, actual_t, C).permute(1, 0, 2)
                plot_inputs(
                    data_in,
                    labels,
                    "train",
                    file_tag=tag + f"_{epoch}",
                    session_tag=session_tag,
                )
                plot_intermediate(
                    [
                        labels.permute(1, 2, 0),
                        predicted_values[: actual_b * actual_t]
                        .reshape(actual_b, actual_t, C)
                        .permute(0, 2, 1),
                    ],
                    ["target", "output"],
                    epoch,
                    "train",
                    i,
                    exp_tag=tag,
                    session_tag=session_tag,
                )

            net.eval()
            with torch.no_grad():
                r2_valid = R2Score(device=device, multioutput="raw_values").reset()
                valid_loss = Mean(device=device).reset()
                OnlyWithinTrials = True
                for i, data in enumerate(validloader):
                    # Load the data to device
                    data = {
                        key: data[key]
                        for key in data
                        if key
                        not in ["session_id", "absolute_start", "output_subtask_index"]
                    }
                    data = {
                        k: (
                            v.to(device=device, non_blocking=True)
                            if hasattr(v, "to")
                            else v
                        )
                        for k, v in data.items()
                    }

                    # Forward Pass
                    predicted_values, loss_, r2 = net(**data)

                    if hasattr(net, "cfg") and "snn" in net.cfg.model.type.lower():
                        # need to reset the membrane potentials
                        if "jelly" in net.cfg.model.type.lower():
                            functional.reset_net(net.model)
                        else:
                            net.model.reset_mem()
                    if (
                        hasattr(net, "cfg")
                        and "spikinghomogenizer" in net.cfg.model.setup
                    ):
                        functional.reset_net(net.homogenizer)

                    # Update the R2 score & Loss only for samples that are within the "reaching" or other task period
                    is_within_trial = torch.where(data["output_weights"] == 5)[0]
                    if is_within_trial.numel() == 0:
                        is_within_trial = torch.where(
                            torch.logical_or(
                                data["output_weights"] == 5, data["output_weights"] == 1
                            )
                        )[0]
                        OnlyWithinTrials = False

                    if is_within_trial.numel() == 0:
                        is_within_trial = torch.ones_like(
                            data["output_weights"], dtype=torch.bool
                        )
                        OnlyWithinTrials = False

                    # is_within_trial = torch.where(torch.logical_or(data['output_weights'] == 5, data['output_weights'] == 1))[0]
                    r2_valid.update(
                        predicted_values[is_within_trial],
                        data["output_values"][is_within_trial],
                    )
                    valid_loss.update(loss_)

                    if plot_data:
                        actual_b = data_in.shape[1]
                        actual_t = data_in.shape[0]
                        labels = data["output_values"][
                            : actual_b * actual_t
                        ]  # off-by-one error?!
                        # more off-by-one ridiculousness
                        if labels.shape[0] < actual_b * actual_t:
                            labels = torch.cat(
                                [
                                    labels,
                                    torch.zeros(
                                        actual_b * actual_t - labels.shape[0], C
                                    ).to(labels.device),
                                ]
                            )
                        if predicted_values.shape[0] < actual_b * actual_t:
                            predicted_values = torch.cat(
                                [
                                    predicted_values,
                                    torch.zeros(
                                        actual_b * actual_t - predicted_values.shape[0],
                                        C,
                                    ).to(predicted_values.device),
                                ]
                            )
                        labels = labels.reshape(actual_b, actual_t, C).permute(1, 0, 2)
                        plot_intermediate(
                            [
                                labels.permute(1, 2, 0),
                                predicted_values[: actual_b * actual_t]
                                .reshape(actual_b, actual_t, C)
                                .permute(0, 2, 1),
                            ],
                            ["target", "output"],
                            epoch,
                            "valid",
                            i,
                            exp_tag=tag,
                            session_tag=session_tag,
                        )

                if best_val_loss is None or valid_loss.compute() < best_val_loss:
                    best_weights = copy.deepcopy(net.state_dict())
                    best_val_loss = valid_loss.compute()
                    try:
                        path_to_write = (
                            net.cfg.save_path
                            + f"{net.cfg.model.type}_{session_tag}_{date.today()}.pt"
                            if hasattr(net, "cfg")
                            else "./trained_models/"
                            + f"POYO_{session_tag}_{date.today()}.pt"
                        )
                        print(f"[train - try] Saving weights to {path_to_write}")
                        if not os.path.exists(Path(path_to_write).parent):
                            os.makedirs(Path(path_to_write).parent, exist_ok=True)
                        torch.save(best_weights, path_to_write)
                    except:
                        path_to_write = (
                            net.cfg.save_path
                            + f"{net.cfg.model.type}_multi_subject_{date.today()}.pt"
                            if hasattr(net, "cfg")
                            else "./trained_models/"
                            + f"POYO_{session_tag}_{date.today()}.pt"
                        )
                        print(f"[train - except] Saving weights to {path_to_write}")
                        torch.save(best_weights, path_to_write)

                # Show the validation Loss
                print(
                    f"[VALID] Total Steps: {steps}, Epoch: {epoch}, Loss: {valid_loss.compute():.4f}, r2: {r2_valid.compute().mean():.2f}, OnlyWithinTrials: {OnlyWithinTrials}"
                )
                wandb.log(
                    {
                        "valid_Loss": valid_loss.compute(),
                        "valid_R2": r2_valid.compute().mean(),
                    },
                    step=steps,
                )

        r2_train = R2Score(device=device, multioutput="raw_values")
        train_loss = Mean(device=device)
        for i, data in enumerate(trainloader):
            net.train()
            data = {
                k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in data.items()
            }

            dawn = time.time()

            # Forward Pass
            predicted_values, loss, r2 = net(**data)

            # Recover shape information
            B = data["spike_unit_index"].shape[0]
            BT, C = data["output_values"].shape
            T = BT // B

            # Compute the R2 score across the training data
            r2_train.update(
                predicted_values, data["output_values"][: predicted_values.shape[0]]
            )
            train_loss.update(loss)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            steps += 1

            dusk = time.time()
            print(f"f/b pass: {dusk - dawn:.2f} seconds")

            if hasattr(net, "cfg") and "snn" in net.cfg.model.type.lower():
                # need to reset the membrane potentials
                if "jelly" in net.cfg.model.type.lower():
                    functional.reset_net(net.model)
                else:
                    net.model.reset_mem()

        if hasattr(net, "get_spike_params"):
            all_sp = net.get_spike_params()  # ffn_sp, ms_sp, int_sp
            spike_params["ffn_sp"].append(all_sp[0])
            spike_params["ms_sp"].append(all_sp[1])
            spike_params["int_sp"].append(all_sp[2])
            print(f"Spike parameters: {all_sp[1]}")

        # try:
        print(
            f"[TRAIN] Total Steps: {steps}, Epoch: {epoch}, Loss: {train_loss.compute():.4f},  r2: {r2_train.compute().mean():.2f}",
            flush=True,
        )
        wandb.log(
            {
                "train_Loss": train_loss.compute(),
                "train_R2": r2_train.compute().mean(),
                "LR": optimizer.param_groups[0]["lr"],
            },
            step=steps,
        )
        # except:
        #     print("Error in logging!", flush=True)

        if epoch > 0.75 * epochs:
            scheduler.step()
        # scheduler.step()

    ## Uncomment to save spiking params
    # if hasattr(net, "get_spike_params"):
    #     spike_param_path = "spike_params_logical_init.pkl"
    #     with open(spike_param_path, "wb") as fout:
    #         # list of tuples of arrays, each list element (ffn_sp, ms_sp, int_sp)
    #         # length of list is equal to number of epochs
    #         pickle.dump(spike_params, fout)

    net.load_state_dict(best_weights)  # Returns the weights of the best network
    return net


def generate_twinx_param_plots(sp_path_1, sp_path_2, group_key, param_index):
    with open(sp_path_1, "rb") as fin:
        spike_params_1 = pickle.load(fin)
    with open(sp_path_2, "rb") as fin:
        spike_params_2 = pickle.load(fin)

    data_1 = spike_params_1[group_key]
    data_2 = spike_params_2[group_key]
    data_1 = np.stack(data_1, axis=0)  # shape: (num_epochs, *param_shape)
    data_2 = np.stack(data_2, axis=0)  # shape: (num_epochs, *param_shape)

    epoch_cutoff = min(data_1.shape[0], data_2.shape[0])

    if group_key == "ffn_sp":
        # process ffn_sp
        plot_data_1 = data_1[..., param_index]  # (E, n_layers,)
        plot_data_2 = data_2[..., param_index]  # (E, n_layers,)
        deviations = None
        labels_1 = [f"ffn_sp_{i}_{param_index}" for i in range(plot_data_1.shape[1])]
        labels_2 = labels_1.copy()
    elif group_key == "ms_sp":
        # process ms_sp
        raw_data_1 = data_1[..., param_index]  # (E, n_taus, n_layers)
        raw_data_2 = data_2[..., param_index]  # (E, n_taus, n_layers)
        plot_data_1 = raw_data_1.mean(axis=2)  # (E, n_taus)
        plot_data_2 = raw_data_2.mean(axis=2)  # (E, n_taus)
        deviations = raw_data_1.std(axis=2)  # (E, n_taus)
        labels_1 = [f"$\\tau_{i}^A$" for i in range(plot_data_1.shape[1])]
        labels_2 = [f"$\\tau_{i}^B$" for i in range(plot_data_2.shape[1])]
    elif group_key == "int_sp":
        # process int_sp
        raw_data_1 = data_1[..., param_index]  # (E, n_taus, n_layers)
        raw_data_2 = data_2[..., param_index]  # (E, n_taus, n_layers)
        plot_data_1 = raw_data_1.mean(axis=2)  # (E, n_taus)
        plot_data_2 = raw_data_2.mean(axis=2)  # (E, n_taus)
        deviations = raw_data_1.std(axis=2)  # (E, n_taus)
        labels_1 = [f"int_sp_{i}_{param_index}" for i in range(plot_data_1.shape[1])]
        labels_2 = [f"int_sp_{i}_{param_index}" for i in range(plot_data_2.shape[1])]
    else:
        raise ValueError("Invalid parameter index")

    # Input data should be a list of arrays: (num_epochs,) x num_params_of_interest
    # Accompanying data: deviations (num_epochs,) x num_params_of_interest
    # Also labels x num_params_of_interest
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis

    for i in range(plot_data_1.shape[1]):
        param_data_1 = plot_data_1[:epoch_cutoff, i]
        param_data_2 = plot_data_2[:epoch_cutoff, i]
        dev = deviations[:epoch_cutoff, i] if deviations is not None else None
        label1 = labels_1[i]
        label2 = labels_2[i]

        # Plot the first dataset on ax1
        ax1.plot(param_data_1, label=label1, color=f"C{i}")
        if dev is not None:
            ax1.fill_between(
                range(len(param_data_1)),
                param_data_1 - dev,
                param_data_1 + dev,
                alpha=0.2,
                color=f"C{i}",
            )

        # Plot the second dataset on ax2
        ax2.plot(param_data_2, label=label2, linestyle="--", color=f"C{i}")

    # ax1.set_yscale("log")  # Set y-axis to log scale for ax1
    ax1.set_title(f"$\\tau$ Parameter Evolution")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dataset 1 Value")
    ax2.set_ylabel("Dataset 2 Value")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    ax1.grid(True)
    # Save figure with transparent background (everywhere except plot elements)
    plt.savefig(
        f"twinx_high_res_{group_key}_{param_index}.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )


def generate_spike_param_plots(spike_param_path, group_key, param_index):
    with open(spike_param_path, "rb") as fin:
        spike_params = pickle.load(fin)

    data = spike_params[group_key]
    data = np.stack(data, axis=0)  # shape: (num_epochs, *param_shape)

    if group_key == "ffn_sp":
        # process ffn_sp
        plot_data = data[..., param_index]  # (E, n_layers,)
        deviations = None
        labels = [f"ffn_sp_{i}_{param_index}" for i in range(plot_data.shape[1])]
    elif group_key == "ms_sp":
        # process ms_sp
        raw_data = data[..., param_index]  # (E, n_taus, n_layers)
        plot_data = raw_data.mean(axis=2)  # (E, n_taus)
        deviations = raw_data.std(axis=2)  # (E, n_taus)
        # labels = [f"ms_sp_{i}_{param_index}" for i in range(plot_data.shape[1])]
        labels = [f"$\\tau_{i}$" for i in range(plot_data.shape[1])]

        # if param_index == 1:
        #     plot_data[:, -1] /= 10
        #     deviations[:, -1] /= 100
        #     labels[-1] = f"ms_sp_{plot_data.shape[1] - 1}_{param_index} / 10"
    elif group_key == "int_sp":
        # process int_sp
        raw_data = data[..., param_index]  # (E, n_taus, n_layers)
        plot_data = raw_data.mean(axis=2)  # (E, n_taus)
        deviations = raw_data.std(axis=2)  # (E, n_taus)
        labels = [f"int_sp_{i}_{param_index}" for i in range(plot_data.shape[1])]
    else:
        raise ValueError("Invalid parameter index")

    # Input data should be a list of arrays: (num_epochs,) x num_params_of_interest
    # Accompanying data: deviations (num_epochs,) x num_params_of_interest
    # Also labels x num_params_of_interest
    plt.figure()
    for i in range(plot_data.shape[1]):
        param_data = plot_data[:, i]
        dev = deviations[:, i] if deviations is not None else None
        label = labels[i]
        plt.plot(param_data, label=label)
        if dev is not None:
            plt.fill_between(
                range(len(param_data)), param_data - dev, param_data + dev, alpha=0.2
            )

    plt.yscale("log")  # Set y-axis to log scale
    plt.title(f"$\\tau$ Parameter Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Value (log scale)")
    plt.legend()
    plt.grid(True)
    # plt.savefig(f"logical_init_spike_params_{group_key}_{param_index}.png")
    # Save figure with transparent background (everywhere except plot elements)
    plt.savefig(
        f"high_res_{group_key}_{param_index}.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )


def reset_weights(model):
    """
    Try resetting model weights to avoid weight leakage.
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


### Spline interpolation code based on: https://stackoverflow.com/a/64872885
def h_poly(t):
    tt = t[None, :] ** torch.arange(4, device=t.device)[:, None]
    A = torch.tensor(
        [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
        dtype=t.dtype,
        device=t.device,
    )
    return A @ tt


def interp(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = x[idxs + 1] - x[idxs]
    hh = h_poly((xs - x[idxs]) / dx)
    return (
        hh[0] * y[idxs]
        + hh[1] * m[idxs] * dx
        + hh[2] * y[idxs + 1]
        + hh[3] * m[idxs + 1] * dx
    )


def spline_upsample_tensor(samples, target_size, indices, dim=0):
    """
    Upsample a tensor using spline interpolation.

    Args:
        samples (torch.Tensor): The tensor to upsample.
        target_size (int): The target size of the upsampled tensor.
        indices (torch.Tensor): The indices of the samples.
        dim (int): The dimension to upsample.

    Returns:
        torch.Tensor: The upsampled tensor.
    """
    interp_lambda = lambda y: interp(
        x=indices, y=y, xs=torch.linspace(indices[0], indices[-1], target_size)
    )
    interp_vec = torch.vmap(
        interp_lambda,
        in_dims=dim,
        out_dims=dim,
    )
    ret = interp_vec(samples)
    return ret


def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def vis_data(spikes, vels, save_path=None):
    from matplotlib import pyplot as plt
    from snntorch import spikeplot as splt

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    #  s: size of scatter points; c: color of scatter points
    splt.raster(spikes, axes[0], s=1.5, c="black")
    axes[0].set_title("Spike raster plot")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Neuron Number")

    axes[1].plot(vels[:, 0].detach().cpu().numpy(), label="x velocity")
    axes[1].plot(vels[:, 1].detach().cpu().numpy(), label="y velocity")
    axes[1].set_title("Cursor velocity")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Velocity")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_inputs(
    spike_tensor, label_tensor, split, file_tag, session_tag, batch_index=0
):
    output_dir = Path(f"vis/input_vis/{session_tag}/")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fname = f"{split}_{file_tag}.png"
    fpath = output_dir / fname
    vis_data(
        spike_tensor[:, batch_index, :],
        label_tensor[:, batch_index, :],
        save_path=fpath,
    )


def plot_intermediate(
    data_list,
    label_list,
    epoch,
    split,
    batch,
    exp_tag="exp_tag_default",
    session_tag="default_session",
    file_tag="output_vs_target",
):
    """Plot outputs and labels at an intermediate step of training.

    Args:
        data_list (List(torch.tensor)): List of (B, C, T) tensors to plot
        label_list (List(str)): List of labels for the data
        epoch (int): Epoch training number
        split (str): Name of split (e.g. train, val, or test)
        batch (int): Batch index in the dataloader
        exp_tag (str): Experiment tag for the visualization - will be used for naming the destination folder
        session_tag (str): Session tag for the visualization - will be used for naming the destination folder
        file_tag (str): File tag for the visualization - will be used for naming the output file
    """
    out_dir = Path(f"vis/dtag_vis/{exp_tag}/{session_tag}/")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    N_lines = len(data_list)
    B, C, T = data_list[0].shape

    for batch_idx in [0]:  # range(outputs.shape[0]):
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        xy_tag = ["x", "y"]
        for data, label in zip(data_list, label_list):
            for i in range(C):  # C should be 2
                data_ = data[batch_idx, i, :].detach().cpu().numpy()
                axes[i].plot(data_, label=label)
                axes[i].set_title(
                    f"$V_{xy_tag[i]}$, Epoch: {epoch}, Split: {split}, Batch: {batch}, Batch Index: {batch_idx}"
                )
                axes[i].legend()
        # for i in range(C):
        #     net_out = outputs[batch_idx, i, :].detach().cpu().numpy()
        #     tar = labels[batch_idx, i, :].detach().cpu().numpy()
        #     axes[i].plot(net_out, label="output")
        #     axes[i].plot(tar, label="target")
        #     axes[i].set_title(
        #         f"$V_{xy_tag[i]}$, Epoch: {epoch}, Split: {split}, Batch: {batch}, Batch Index: {batch_idx}"
        #     )
        #     axes[i].legend()

        out_path = out_dir / f"{file_tag}_{split}_{epoch}_{batch}.png"
        plt.savefig(out_path)
        plt.close()


def main():
    test_num = 3

    if test_num == 0:
        x = torch.linspace(0, 10, 10)
        y = torch.linspace(3, 20, x.shape[0])  # torch.randn(10)
        xs = torch.linspace(0, 10, 100)
        ys = interp(x, y, xs)

        fig = plt.figure()
        plt.plot(x, y, "o", label="points")
        plt.plot(xs, ys, label="spline")
        plt.title("Spline check")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif test_num == 1:
        samples = torch.randn(10, 7)
        target_size = 170
        indices = torch.linspace(0, 14, samples.shape[-1]).long()

        # dim is the dimension we are *preserving* (i.e. the dimension that is not being upsampled)
        upsampled = spline_upsample_tensor(samples, target_size, indices, dim=0)
        print(f"Upsampling shapes: {samples.shape} -> {upsampled.shape}")

        plot_idx = 1
        plt.figure()
        plt.plot(indices, samples[plot_idx], label="original")
        plt.plot(
            torch.linspace(indices[0], indices[-1], target_size),
            upsampled[plot_idx],
            label="upsampled",
        )
        plt.title("Upsampling check")
        plt.show()

    elif test_num == 2:
        # test spike params plotting
        from itertools import product

        spike_param_path = "spike_params_4-8-25.pkl"  # "/home/ioannis/poyo/spike_params_data_logs/othhheeee_one.pkl"
        group_keys = ["ms_sp", "ffn_sp"]
        param_idxs = [0, 1]
        combos = list(product(group_keys, param_idxs))
        for group_key, param_index in combos:
            generate_spike_param_plots(spike_param_path, group_key, param_index)

    elif test_num == 3:
        # spike params twinx plotting
        from itertools import product

        sp_path_1 = "spike_params_4-8-25.pkl"
        sp_path_2 = "spike_params_data_logs/othhheeee_one.pkl"
        group_keys = ["ms_sp", "ffn_sp"]
        param_idxs = [0, 1]
        combos = list(product(group_keys, param_idxs))
        for group_key, param_index in combos:
            generate_twinx_param_plots(sp_path_1, sp_path_2, group_key, param_index)


if __name__ == "__main__":
    main()
