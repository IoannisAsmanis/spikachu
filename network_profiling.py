from collections import OrderedDict
from pprint import pprint
import os
import glob
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import thop
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler, Sampler, RandomSampler
from torch.profiler import profile, record_function, ProfilerActivity

from models.spiking_attention import LTLIFNode
from poyo.data.dataset import Dataset, DatasetIndex
from poyo.data.sampler import (
    SequentialFixedWindowSampler,
    RandomFixedWindowSampler,
)
from poyo.data.collate import collate, chain, pad
from models.model_wrapper import model_wrapper, POYOTokenizer
from poyo.models.poyo import POYO

# import syops

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import surrogate
import spikingjelly.activation_based.functional as sf
from models.baselines import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pynvml
import time


def gpu_energy_usage_joules(model, inputs, n_tries=10):
    print(f"Running estimation on {device}")
    model.eval()

    energies_joules = []
    for i in range(n_tries):
        with torch.no_grad():
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                power_idle = pynvml.nvmlDeviceGetPowerUsage(handle)  # mW

                # Warm-up the GPU (CUDA lazy initialization)
                _ = model(**inputs)
                torch.cuda.synchronize()

                # Start energy measurement
                start_time = time.perf_counter()
                _ = model(**inputs)
                torch.cuda.synchronize()

                end_time = time.perf_counter()
                power_after = pynvml.nvmlDeviceGetPowerUsage(handle)  # mW

                power_mw = power_after - power_idle
                power_w = power_mw / 1000.0  # Convert mW to W
                elapsed_time = end_time - start_time  # in seconds

                energy_joules = power_w * elapsed_time
                energies_joules.append(energy_joules)

                print(
                    f"Used power: {power_w:.4f} W ({power_idle=} mW to {power_after=} mW)"
                )
                print(f"Elapsed time: {elapsed_time:.6f} seconds")
                print(f"Energy used: {energy_joules:.6f} J")
            finally:
                pynvml.nvmlShutdown()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(5.0)

    ret = torch.tensor(energies_joules, dtype=float)

    print("-" * 50)
    print(f"Average GPU Energy Usage: {ret.mean():.6f} Joules")
    print(f"Standard Deviation: {ret.std():.6f} Joules")
    print("+" * 50)

    return ret


# def gpu_energy_usage_joules(model, inputs, n_tries=10):
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     energies_joules = []

#     for i in range(n_tries):
#         with torch.no_grad():
#             _ = model(**inputs)  # warm-up
#             torch.cuda.synchronize()

#             start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
#             _ = model(**inputs)
#             torch.cuda.synchronize()
#             end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
#             energy_joules = (end_energy - start_energy) / 1000.0
#             print(f"Energy used: {energy_joules:.6f} J")
#             energies_joules.append(energy_joules)
#         torch.cuda.empty_cache()
#         time.sleep(1.0)

#     pynvml.nvmlShutdown()
#     ret = torch.tensor(energies_joules, dtype=float)
#     print("-" * 50)
#     print(f"Average GPU Energy Usage: {ret.mean():.6f} Joules")
#     print(f"Standard Deviation: {ret.std():.6f} Joules")
#     print("+" * 50)
#     return ret


def profile_model(prof_type, model, inputs, device="cuda"):
    model.to(device)

    if prof_type == "thop":
        flops, params = thop.profile(model, inputs=(inputs,))
        return (flops, params)
    elif prof_type == "torch":
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("model_inference"):
                _ = model(inputs)
            return prof  # .key_averages().table(sort_by="self_cpu_time_total")


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_full_model(cfg):
    num_electrodes = pd.read_csv(
        cfg.data.data_path + cfg.data.dandiset + "/num_electrodes.csv",
        index_col=0,
    )

    # session 10
    session = "t_20130830_random_target_reaching"

    num_neurons = num_electrodes.loc[session, "num_electrodes"]

    if os.path.isfile(cfg.save_path):
        model = model_wrapper(
            num_neurons=num_neurons,
            cfg=cfg,
            init_embeddings=False,  # TODO: figure out how to load this propeprly
        )
        trained_state_dict = torch.load(
            cfg.save_path, map_location=device, weights_only=False
        )
        model.load_state_dict(trained_state_dict)
    elif cfg.model.type.lower() == "poyo":
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
            causality_enabled=cfg.model.setup.causality_enabled,
        ).to(device)
    else:
        model = model_wrapper(num_neurons=num_neurons, cfg=cfg, init_embeddings=True)

    model.to(device)

    train_tokenizer = POYOTokenizer(
        model.unit_tokenizer,
        model.session_tokenizer,
        latent_step=cfg.tokenizer.latent_step,
        num_latents_per_step=cfg.tokenizer.num_latents_per_step,
        using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
        eval=False,
    )

    dandiset = cfg.data.dandiset
    dataset = Dataset(
        root=cfg.data.data_path,
        split="train",
        include=[{"selection": [{"dandiset": dandiset, "session": session}]}],
        transform=train_tokenizer,
    )

    train_sampler = RandomFixedWindowSampler(
        interval_dict=dataset.get_sampling_intervals(),
        window_length=cfg.sampler.window_length,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.sampler.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
    )

    inputs = next(iter(dataloader))
    inputs = {
        k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    return model, inputs


def generate_targeted_model(cfg, model_load_path, session_num):
    num_electrodes = pd.read_csv(
        cfg.data.data_path + cfg.data.dandiset + "/num_electrodes.csv",
        index_col=0,
    )

    session = num_electrodes.index[session_num]

    num_neurons = num_electrodes.loc[session, "num_electrodes"]

    if model_load_path is not None:
        model = model_wrapper(
            num_neurons=num_neurons,
            cfg=cfg,
            init_embeddings=False,  # TODO: figure out how to load this propeprly
        )
        trained_state_dict = torch.load(
            model_load_path, map_location=device, weights_only=False
        )
        model.load_state_dict(trained_state_dict)
    elif cfg.model.type.lower() == "poyo":
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
            causality_enabled=cfg.model.setup.causality_enabled,
        ).to(device)
    else:
        model = model_wrapper(num_neurons=num_neurons, cfg=cfg, init_embeddings=True)

    model.to(device)

    train_tokenizer = POYOTokenizer(
        model.unit_tokenizer,
        model.session_tokenizer,
        latent_step=cfg.tokenizer.latent_step,
        num_latents_per_step=cfg.tokenizer.num_latents_per_step,
        using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
        eval=False,
    )

    dandiset = cfg.data.dandiset
    dataset = Dataset(
        root=cfg.data.data_path,
        split="train",
        include=[{"selection": [{"dandiset": dandiset, "session": session}]}],
        transform=train_tokenizer,
    )

    train_sampler = RandomFixedWindowSampler(
        interval_dict=dataset.get_sampling_intervals(),
        window_length=cfg.sampler.window_length,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,  # cfg.sampler.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
    )

    inputs = next(iter(dataloader))
    inputs = {
        k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    return model, inputs


def model_profiling(cfg, model, inputs):
    results = profile_model(cfg.prof_type, model, inputs=inputs, device=device)

    # NOTE: basically only thop gives a rough estimate of torch.nn-related FLOPs

    if cfg.prof_type == "syops":
        pass  # syops.
    elif cfg.prof_type == "thop":
        flops, params = results
        sample_flops = flops / cfg.sampler.batch_size
        print(f"FLOPs per sample: {sample_flops}, Params: {params}")
        train_params = count_trainable_parameters(model)
        print("Trainable parameters: ", train_params)

        num_lif_neurons = (train_params - params) // 2
        print(f"Number of LIF neurons: {num_lif_neurons}")

        full_op_count = sample_flops + num_lif_neurons
        print(f"Full operation count: {full_op_count}")

        if isinstance(inputs, dict):
            print(f"Input shape: {inputs['spike_timestamps'].shape}")
        else:
            print(f"Input shape: {inputs.shape}")
    elif cfg.prof_type == "torch":
        keys = list(filter(lambda x: not x.startswith("_"), dir(results)))
        print(results)


def estimate_spiking_energy(
    neuron_counts_per_module, spiking_rate_per_module, energy_per_spike=9e-13
):
    """
    Estimate the energy consumption of a spiking neural network.
    :param neuron_counts_per_layer: List of neuron counts for each layer.
    :param spiking_rate_per_layer: List of spiking rates for each layer.
    :param energy_per_spike: Energy consumed per spike in Joules (default is 0.9 pJ).
    :return: Estimated energy consumption in Joules.
    """
    total_energy = 0.0
    sops = 0
    for neuron_count, spiking_rate in zip(
        neuron_counts_per_module, spiking_rate_per_module
    ):
        sops += neuron_count * spiking_rate
        total_energy += sops * energy_per_spike
    return total_energy, sops


def estimate_energy_snn(
    module_energy_dict, energy_per_spike=9e-13, energy_per_flop=4.6e-12
):
    spiking_energy_dict = {}
    flop_energy_dict = {}
    sop_dict = {}
    flop_dict = {}
    for mod_name, mod_specs in module_energy_dict.items():
        neuron_count = mod_specs.get("neuron_count", -1)
        spiking_rate = mod_specs.get("spiking_rate", -1)
        # flops = mod_specs.get("flops", -1)
        maccs = mod_specs.get("maccs", -1)
        accs = mod_specs.get("accs", -1)

        energy_spiking, sops = estimate_spiking_energy(
            neuron_counts_per_module=[neuron_count],
            spiking_rate_per_module=[spiking_rate],
            energy_per_spike=energy_per_spike,
        )

        energy_maccs = maccs * energy_per_flop if maccs else 0.0
        energy_accs = accs * energy_per_spike if accs else 0.0
        energy_flops = energy_maccs + energy_accs

        spiking_energy_dict[mod_name] = energy_spiking
        flop_energy_dict[mod_name] = energy_flops
        sop_dict[mod_name] = sops
        flop_dict[mod_name] = (maccs, accs)

    return spiking_energy_dict, flop_energy_dict, sop_dict, flop_dict


def single_recurrent_cell_flops(rec_type, input_size, hidden_size, bias):
    # see here: https://github.com/ultralytics/thop/blob/main/thop/rnn_hooks.py
    # gate_flops are only matrix multiplies and additions of their results
    # bias terms apply to all matrix multiplies
    # hadamard terms/elementwise terms (included elem-wise tanh) counted separately
    maccs = 0
    accs = 0
    if rec_type.lower() == "rnn":
        bias_term = 2 * hidden_size if bias else 0
        gate_flops = hidden_size * (input_size + hidden_size) + hidden_size
        # flops = gate_flops + bias_term
        maccs = gate_flops
        accs = bias_term
    elif rec_type.lower() == "gru":
        bias_term = 2 * hidden_size if bias else 0
        gate_flops = hidden_size * (input_size + hidden_size) + hidden_size
        elemwise_flops = hidden_size * 4
        # flops = 3 * (gate_flops + bias_term) + elemwise_flops
        maccs = 3 * gate_flops + elemwise_flops
        accs = 3 * bias_term
    elif rec_type.lower() == "lstm":
        bias_term = 2 * hidden_size if bias else 0
        gate_flops = hidden_size * (input_size + hidden_size) + hidden_size
        hadamard_flops = hidden_size * 3
        # flops = 4 * (gate_flops + bias_term) + hadamard_flops + hidden_size
        maccs = 4 * gate_flops + hadamard_flops
        accs = 4 * bias_term + hidden_size
    else:
        raise ValueError(f"Unknown recurrent type: {rec_type}")
    return maccs, accs


def full_recurrent_module_flops(rec_type, module):

    # check if we are doing bidirectional
    dir_factor = 2 if module.bidirectional else 1

    # flops for the first layer: (input -> hidden) x dir_factor
    internal_maccs, internal_accs = (
        single_recurrent_cell_flops(
            rec_type,
            module.input_size,
            module.hidden_size,
            module.bias,
        )
        * dir_factor
    )

    # flops for each subsequent layer: (hidden * dir_factor -> hidden) x dir_factor
    layer_maccs, layer_accs = (
        single_recurrent_cell_flops(
            rec_type,
            module.hidden_size * dir_factor,
            module.hidden_size,
            module.bias,
        )
        * dir_factor
    )
    maccs = internal_maccs + layer_maccs * (module.num_layers - 1)
    accs = internal_accs + layer_accs * (module.num_layers - 1)
    return maccs, accs


def construct_module_specs(model):
    module_spec_dict = {}
    last_module = None
    last_module_name = None
    for name, module in model.named_modules():
        if isinstance(module, LTLIFNode):
            neuron_count = module.get_explicit_neuron_count()
            spiking_rate = module.spike_rate
            maccs = 0
            accs = 0
        elif isinstance(module, (layer.Linear, torch.nn.Linear)):
            neuron_count = module.weight.shape[0]
            spiking_rate = 0.0
            flops = module.weight.shape[0] * module.weight.shape[1]
            accs = maccs = 0
            if isinstance(last_module, LTLIFNode):
                accs += flops * last_module.spike_rate
                print(
                    f"ASSOCIATION:\n{last_module_name}, {last_module}\n{last_module.v.shape}-> {name}, {module} @ {last_module.spike_rate}"
                )
            else:
                maccs += flops
            if module.bias is not None:
                accs += module.weight.shape[0]
        elif isinstance(module, (layer.BatchNorm1d, torch.nn.BatchNorm1d)):
            maccs = module.num_features * 6

        # recurrent blocks
        # only reports FLOPS for a single time step, with batch size 1
        elif isinstance(module, torch.nn.RNN):
            maccs, accs = full_recurrent_module_flops("rnn", module)
            spiking_rate = 0.0
            neuron_count = module.hidden_size
        elif isinstance(module, torch.nn.GRU):
            maccs, accs = full_recurrent_module_flops("gru", module)
            spiking_rate = 0.0
            neuron_count = module.hidden_size
        elif isinstance(module, torch.nn.LSTM):
            maccs, accs = full_recurrent_module_flops("lstm", module)
            spiking_rate = 0.0
            neuron_count = module.hidden_size

        elif "attention" in module.__class__.__name__.lower():
            try:
                print(f"Module {name} is ATTENTION!")
                neuron_count = 0
                spiking_rate = 0.0
                maccs = module.maccs
                accs = module.accs or 0
            except:
                print(f"Module {name} is ATTENTION but no maccs/accs!")
                neuron_count = 0
                spiking_rate = 0.0
                maccs = 0
                accs = 0

        else:
            # TODO: need to count exact FLOPs for MultiHeadAttention
            # not the Linear() ops, only the extras in the forward function
            # print(f"Module {name} not supported yet!")
            # print(f"Module type: {type(module)}")
            continue
        last_module = module
        last_module_name = name

        module_spec_dict[name] = {
            "neuron_count": neuron_count,
            "spiking_rate": spiking_rate,
            "maccs": maccs.item() if isinstance(maccs, torch.Tensor) else maccs,
            "accs": accs.item() if isinstance(accs, torch.Tensor) else accs,
        }
    return module_spec_dict


def backprop_unit_test(input_data, labels, criterion, optimizer, neuron, n_epochs):
    neuron.train()
    spike_params = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        yhat = neuron(input_data)
        loss = criterion(yhat, labels)
        loss.backward()
        optimizer.step()
        sf.reset_net(neuron)
        spike_params.append(neuron[1].get_spike_params())

    plot_data = np.stack(spike_params, axis=0)

    plt.figure()
    for param_index in range(plot_data.shape[1]):
        plt.plot(plot_data[:, param_index], label=f"Parameter {param_index}")

    plt.title(f"Unit test of Spike Parameters Over {n_epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("spike_params_unit_test.png")

    return plot_data


def count_lif_neurons(model):
    """
    Count the number of LIF neurons in the model.
    :param model: The model to analyze.
    :return: The number of LIF neurons.
    """
    lif_neuron_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LTLIFNode):
            lif_neuron_count += module.get_explicit_neuron_count()
    return lif_neuron_count


def single_snn_energy_bench(cfg, model_load_path, session_id, run_model=True):

    # TODO: recover taus, flops etc

    ## Background stuff ##
    # all_ret = None
    # for i in range(10):
    #     model, inputs = generate_targeted_model(cfg, model_load_path, session_id)
    #     ret = np.array(base_power_test(20))
    #     ret = ret[ret > 60]
    #     all_ret = np.concatenate((all_ret, ret), axis=0) if all_ret is not None else ret

    # print(f"Average power: {np.mean(all_ret)} W")
    # print(f"Standard deviation: {np.std(all_ret)} W")
    # print(f"Min/max power: {np.min(all_ret)} W / {np.max(all_ret)} W")

    model, inputs = generate_targeted_model(cfg, model_load_path, session_id)

    # gpu_energy = gpu_energy_usage_joules(model, inputs)

    # print(f"GPU ENERGY TOTAL: {sum(gpu_energy)} Joules")
    # pprint(gpu_energy)

    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params}")

    if run_model:
        with torch.no_grad():
            out = model(**inputs)
    med = construct_module_specs(model)

    spike_rates = np.array(
        [
            v["spiking_rate"].detach().cpu()
            for k, v in med.items()
            if v["spiking_rate"] > 0
        ]
    )
    print(f"LTLIF neurons: {count_lif_neurons(model)}")

    print(f"Mean spike rate: {np.mean(spike_rates)}")
    print(f"Std spike rate: {np.std(spike_rates)}")

    pprint(f"Module energy dict: {med}")
    spike_energy, regular_energy, sops, flops = estimate_energy_snn(med)

    pprint(f"Spike energy: {spike_energy}")
    se = sum(v for k, v in spike_energy.items())
    print(f"Spike energy sum: {se}")

    pprint(f"Regular energy: {regular_energy}")
    re = sum(v for k, v in regular_energy.items())
    print(f"Regular energy sum: {re}")

    # total_sops = sum(v for k, v in sops.items())
    # total_flops = sum(v for k, v in flops.items())
    # print(f"Total sops: {total_sops}")
    # print(f"Total flops: {total_flops}")

    return (
        spike_rates,
        spike_energy,
        regular_energy,
        sops,
        flops,
        trainable_params,
        # gpu_energy,
    )


OUTPUT_COLUMNS = [
    "session_id",
    "spike_rate_mean",
    "spike_rate_std",
    "energy_spike_sum",
    "energy_regular_sum",
    "MACCs",
    "ACCs",
    "SOPs",
    "trainable_params",
]


def baseline_benchmark(cfg, directories, tag_str="2025-04-30"):
    # for dir in directories:
    for i in range(1):
        model_type = cfg.model.type.lower()
        # models = glob.glob(
        #     os.path.join(dir, f"*{model_type}*{tag_str}*.pt")
        # ) + glob.glob(os.path.join(dir, f"*{model_type}*{tag_str}*.pth"))
        # models = sorted(models)
        # print(f"Working on {dir}, with models:")
        # pprint(models)
        results = OrderedDict()
        cnt = 0
        # for model_path in models:
        #     model_name = model_path.split("/")[-1]
        #     session_id = int(model_name.split("_")[1])
        #     print(f"Working on {model_name} for session {session_id}")
        for session_id in range(1):
            print(f"Working on session {session_id}, NO MODEL LOADING")
            # try:
            (
                spike_rates,
                spike_energy,
                regular_energy,
                sops,
                flops,
                trainable_params,
                gpu_energy,
            ) = single_snn_energy_bench(cfg, None, session_id, run_model=False)
            # except Exception as e:
            #     print(f"Error processing {session_id}: {e}")
            #     continue

            print(f"Spike rates: {spike_rates}")
            print(f"Spike energy: {spike_energy}")
            print(f"Regular energy: {regular_energy}")

            res_dict = {
                "session_id": session_id,
                "spike_rates": spike_rates,
                "spike_energy": spike_energy,
                "regular_energy": regular_energy,
                "sops": sops,
                "flops": flops,
                "params": trainable_params,
            }
            results[session_id] = res_dict
            cnt += 1

            # if cnt == 2:
            #     break
            # break
        # break
        fout = open(
            f"baseline_finalized_bench/MAZE_homog_baseline_post_train_benchmark_{model_type}.csv",
            "w",
        )
        fout.write(",".join(OUTPUT_COLUMNS) + "\n")
        for k, v in results.items():
            session_id = v["session_id"]
            spike_rates = v["spike_rates"]
            spike_energy = v["spike_energy"]
            regular_energy = v["regular_energy"]
            sops = v["sops"]
            flops = v["flops"]
            trainable_params = v["params"]

            spike_energy_sum = sum(spike_energy.values()) * 1e6
            if isinstance(spike_energy_sum, torch.Tensor):
                spike_energy_sum = spike_energy_sum.item()

            regular_energy_sum = sum(regular_energy.values()) * 1e6
            if isinstance(regular_energy_sum, torch.Tensor):
                regular_energy_sum = regular_energy_sum.item()

            sops = sum(sops.values())
            if isinstance(sops, torch.Tensor):
                sops = sops.item()

            outputs = [
                session_id,
                np.mean(spike_rates),
                np.std(spike_rates),
                spike_energy_sum,
                regular_energy_sum,
                sops,
                sum(maccs for maccs, _ in flops.values()),
                sum(accs for _, accs in flops.values()),
                trainable_params,
            ]
            output_strings = []
            for i in range(len(outputs)):
                if isinstance(outputs[i], str):
                    output_strings.append(outputs[i])
                elif "float" in str(type(outputs[i])):
                    output_strings.append(f"{outputs[i]:.4f}")
                elif isinstance(outputs[i], int):
                    output_strings.append(f"{outputs[i]}")
                else:
                    output_strings.append(str(outputs[i]))
            fout.write(",".join(output_strings) + "\n")
        fout.close()


def snn_benchmark(cfg, directories):

    # each directory contains all models trained with a setting
    for dir in directories:
        setting = dir.split("/")[-1]

        fout = open(
            f"snn_finalized_bench/MAZE_snn_post_train_benchmark_{setting}.csv", "w"
        )
        fout.write(",".join(OUTPUT_COLUMNS) + "\n")

        models = glob.glob(os.path.join(dir, "*.pt")) + glob.glob(
            os.path.join(dir, "*.pth")
        )
        models = sorted(models)

        print(f"Working on {setting}, with models:")
        pprint(models)

        results = OrderedDict()
        cnt = 0

        # iterate over all models
        for model_path in models:
            model_name = model_path.split("/")[-1]
            session_id = int(model_name.split("_")[1])
            print(f"Working on {model_name} for session {session_id}")

            # single_snn_energy_bench(cfg, model_path, session_id)
            # continue

            (
                spike_rates,
                spike_energy,
                regular_energy,
                sops,
                flops,
                trainable_params,
                # gpu_energy,
            ) = single_snn_energy_bench(cfg, model_path, session_id)

            print(f"Spike rates: {spike_rates}")
            print(f"Spike energy: {spike_energy}")
            print(f"Regular energy: {regular_energy}")

            res_dict = {
                "session_id": session_id,
                "spike_rates": spike_rates,
                "spike_energy": spike_energy,
                "regular_energy": regular_energy,
                "sops": sops,
                "flops": flops,
                "params": trainable_params,
                # "gpu_energy": gpu_energy,
            }
            results[session_id] = res_dict
            cnt += 1

            # if cnt == 2:
            #     break
        # break

        # OUTPUT_COLUMNS = [
        #     "session_id",
        #     "spike_rate_mean",
        #     "spike_rate_std",
        #     "energy_spike_sum",
        #     "energy_regular_sum",
        #     "MACCs",
        #     "ACCs",
        #     "SOPs",
        #     "trainable_params",
        # ]

        for k, v in results.items():
            session_id = v["session_id"]
            spike_rates = v["spike_rates"]
            spike_energy = v["spike_energy"]
            regular_energy = v["regular_energy"]
            sops = v["sops"]
            flops = v["flops"]
            trainable_params = v["params"]
            # gpu_energy = v["gpu_energy"]

            outputs = [
                session_id,
                np.mean(spike_rates),
                np.std(spike_rates),
                sum(spike_energy.values()).item() * 1e6,
                sum(regular_energy.values()) * 1e6,
                sum(sops.values()).item(),
                sum(maccs for maccs, _ in flops.values()),
                sum(accs for _, accs in flops.values()),
                trainable_params,
                # gpu_energy,
            ]
            output_strings = []
            for i in range(len(outputs)):
                if isinstance(outputs[i], str):
                    output_strings.append(outputs[i])
                elif "float" in str(type(outputs[i])):
                    output_strings.append(f"{outputs[i]:.4f}")
                elif isinstance(outputs[i], int):
                    output_strings.append(f"{outputs[i]}")
                else:
                    output_strings.append(str(outputs[i]))
            fout.write(",".join(output_strings) + "\n")
        fout.close()


def base_power_test(n_tries=10):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    ret = []
    for _ in range(n_tries):
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_w = power_mw / 1000
        ret.append(power_w)
        print(f"Power: {power_w:.2f} W")
        time.sleep(1.0)

    pynvml.nvmlShutdown()
    return ret


@hydra.main(
    config_path="configs",
    config_name="multisubject_root_config_profiler",
    version_base=None,
)
def main(cfg: DictConfig):
    root_path = "/home/gment/poyo/trained_models/"  # TODO: set this to your path
    directories = os.listdir(root_path)
    directories = [
        os.path.join(root_path, dir)
        for dir in directories
        if os.path.isdir(os.path.join(root_path, dir))
    ]
    directories = sorted(
        list(
            filter(
                lambda x: (
                    "spikachu_1000_epochs" in x and "delete" not in x and "maze" in x
                ),
                directories,
            )
        ),
        reverse=True,
    )

    print("About to work on:")
    pprint(directories)

    # TODO: change according to what you are profiling
    snn_benchmark(cfg, directories)
    # baseline_benchmark(cfg, directories)

    print("Done with benchmarking")

    # batch_size = 32
    # time_steps = 10
    # dim_feats = 15
    # input_data = torch.rand(batch_size, time_steps, dim_feats)
    # labels = torch.rand_like(input_data)
    # criterion = torch.nn.MSELoss()

    # tau = 1.1
    # surr = surrogate.ATan()
    # backend = "torch"
    # v_threshold = 0.75
    # v_reset = 0.0
    # ltlif = LTLIFNode(
    #     init_tau=tau,
    #     surrogate_function=surr,
    #     step_mode="m",
    #     backend=backend,
    #     v_threshold=v_threshold,
    #     v_reset=v_reset,
    #     decay_input=False,
    # )

    # neuron = nn.Sequential(
    #     layer.Linear(dim_feats, dim_feats),
    #     ltlif,
    #     layer.BatchNorm1d(time_steps),
    # ).to(device)

    # neuron.train()
    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, neuron.parameters()),
    #     lr=0.002,
    #     weight_decay=0.0001,
    # )
    # n_epochs = 100  # Number of epochs for the unit test
    # spike_params = backprop_unit_test(
    #     input_data.to(device),
    #     labels.to(device),
    #     criterion,
    #     optimizer,
    #     neuron,
    #     n_epochs,
    # )

    # print("Done with UT")

    # get full model and real sample inputs
    # model, inputs = generate_full_model(cfg)

    # print(f"Trainable parameters: {count_trainable_parameters(model)}")

    # # warmup first
    # warmup_tries = 10
    # with torch.no_grad():
    #     # warmup_time = 0.0
    #     # for _ in range(warmup_tries):
    #     #     out = model(**inputs)
    #     #     warmup_time += model.delta_t_raw_forward
    #     # warmup_time /= warmup_tries
    #     # print(f"AVG warmup time: {warmup_time} sec")

    #     # # decompose model into energy dict
    #     # times = []
    #     # for _ in range(10):
    #     #     out = model(**inputs)
    #     #     times.append(model.delta_t_raw_forward)
    #     # times = np.array(times)
    #     out = model(**inputs)
    # med = construct_module_specs(model)

    # spike_rates = np.array(
    #     [v["spiking_rate"] for k, v in med.items() if v["spiking_rate"] > 0]
    # )
    # print(f"Mean spike rate: {np.mean(spike_rates)}")
    # print(f"Std spike rate: {np.std(spike_rates)}")

    # print(f"Module energy dict: {med}")
    # spike_energy, regular_energy = estimate_energy_snn(med)

    # print(f"Spike energy: {spike_energy}")
    # se = sum(v for k, v in spike_energy.items())
    # print(f"Spike energy sum: {se}")

    # print(f"Regular energy: {regular_energy}")
    # re = sum(v for k, v in regular_energy.items())
    # print(f"Regular energy sum: {re}")

    # times *= 1000.0  # back to ms
    # print(f"Forward compute time: {times.mean():.3f} +/- {times.std():.3f} ms")
    # print("Done!")

    # import pdb

    # pdb.set_trace()

    # rec_model = ConfigurableRecurrentModel(
    #     n_blocks=4,
    #     block_cfg={
    #         "rec_type": "RNN",
    #         "input_dim": 100,
    #         "hidden_dim": 212,
    #         "n_rec_layers": 8,
    #         "output_dim": 2,
    #         "p_drop": 0.1,
    #         "bidirectional": False,
    #     },
    # )
    # rec_model = model_wrapper(
    #     num_neurons=100,
    #     cfg=cfg,
    #     init_embeddings=False,  # TODO: figure out how to load this propeprly
    # )

    # rec_model, inputs = generate_full_model(cfg)
    # print(rec_model)

    # !!!! TOSAVE - MUST RUN IN LOOP OVER !!!!
    # .CSV with index=session_id, columns=
    #   average spike rate
    #   std spike rate
    #   energy total / energy spike / energy regular
    #   FLOPs
    #   params

    # rec_model = ConfigurableRecurrentModel(
    #     n_blocks=4,
    #     block_cfg={
    #         "rec_type": "GRU",
    #         "input_dim": 100,
    #         "hidden_dim": 184,
    #         "n_rec_layers": 4,
    #         "output_dim": 2,
    #         "p_drop": 0.1,
    #         "bidirectional": False,
    #     },
    # )

    # rec_model = ConfigurableRecurrentModel(
    #     n_blocks=4,
    #     block_cfg={
    #         "rec_type": "LSTM",
    #         "input_dim": 100,
    #         "hidden_dim": 162,
    #         "n_rec_layers": 4,
    #         "output_dim": 2,
    #         "p_drop": 0.1,
    #         "bidirectional": False,
    #     },
    # )

    # print(f"Trainable parameters: {count_trainable_parameters(rec_model)}")

    # rec_model.to(device)
    # rec_model.eval()

    # med = construct_module_specs(rec_model)

    # # spike_rates = np.array(
    # #     [v["spiking_rate"] for k, v in med.items() if v["spiking_rate"] > 0]
    # # )
    # # print(f"Mean spike rate: {np.mean(spike_rates)}")
    # # print(f"Std spike rate: {np.std(spike_rates)}")

    # print(f"Module energy dict: {med}")
    # spike_energy, regular_energy = estimate_energy_snn(med)

    # print(f"Spike energy: {spike_energy}")
    # se = sum(v for k, v in spike_energy.items())
    # print(f"Spike energy sum: {se}")

    # print(f"Regular energy: {regular_energy}")
    # re = sum(v for k, v in regular_energy.items())
    # print(f"Regular energy sum: {re}")

    # # model.to(device)
    # # model_profiling(cfg, model, inputs)

    # ## get spiking parameters over time
    # # spike_params = model.get_spike_params()
    # # print(f"Initial spike params: {spike_params.shape}")

    print("Done!")


if __name__ == "__main__":
    main()

# def main(args):
#     model = model_wrapper()
#     trained_state_dict = torch.load(args.trained_model_path)
#     model.load_state_dict(trained_state_dict)

#     train_tokenizer = POYOTokenizer(
#         model.unit_tokenizer,
#         model.session_tokenizer,
#         latent_step=cfg.tokenizer.latent_step,
#         num_latents_per_step=cfg.tokenizer.num_latents_per_step,
#         using_memory_efficient_attn=cfg.tokenizer.use_memory_efficient_attn,
#         eval=False,
#     )

#     dandiset = "000688"
#     session = "c_20150311_center_out_reaching"
#     dataset = Dataset(
#         root=f".../all_h5_files/",
#         split="train",
#         include=[
#             {
#                 "selection": [
#                     {"dandiset": dandiset, "session": session}
#                 ],
#             }
#         ],
#         transform=train_tokenizer,
#     )

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Profile a Spiking Neural Network model.')
#     parser.add_argument('trained_model_path', type=str, help='Path to the model file')
#     parser.add_argument('dataset_sequence', type=str, help='Dataset sequence to use')
#     parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (default: cuda)')
#     args = parser.parse_args()
#     main(args)
