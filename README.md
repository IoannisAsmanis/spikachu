# Spikachu

This is the official implementation repository for the NIPS '25 paper: **A Scalable, Causal, and Energy Efficient Framework for Neural Decoding with Spiking Neural Networks**

# Instructions

## Setting up

To replicate our results, please follow the instructions below.
Note that all scripts should be executed from the **root** directory of this repository.

### Assumptions

This pipeline has been developed and tested on a Linux Ubuntu 22.04.5 LTS GPU cluster, with SLURM.
The recommended Python version is 3.10.12, with PyTorch 2.6.0+cu124.
To run training, an NVIDIA GPU must be available on your system, with proprietary drivers and CUDA already installed and configured.

### Configuring Python

We recommend using a virtual environment for your Python configuration.
To quickly set up the environment, use:
```bash
$ pip install -r requirements.txt
```
To verify that your GPU is properly detected and usable by PyTorch, run:
```python
import torch
print(torch.__version__) # should be version with +cuXXX
print(torch.cuda.is_available()) # should be true
x = torch.tensor([0, 1, 2, 3]).cuda()
x *= 2
print(x) # should print 0, 2, 4, 6, with device cuda:0 or similar
```


## Data pre-processing

### Setting your paths

Ensure you have the absolute system path of the directory containing your copy of the Perich et al. 2018 dataset.

### Preprocessing

We provide a script in the `data_preprocessing/` directory, corresponding to the main dataset used in our work.
You can run this script as follows:
```bash
$ python3 data_preprocessing/prepare_data.py \
    --input_dir "/PATH/TO/ORIGINAL/DATA/" \
    --output_dir "/PATH/TO/PROCESSED/DATA/"
```
Now update the `data:data_path` key in the desired YAML configuration file for your training, in the `configs/` directory, to be the output directory of the previous step.

## Training

### Baseline methods
To train baseline model architectures:
- open `configs/baselines.yaml`, and select your desired model with the `model:type` key (options are: mlp, rnn, gru, lstm)
- in case your GPU memory is limited, consider reducing the value of `sampler:batch_size` to reduce memory requirements of training
- finally, ensure that the `data:data_path` key is set correctly
- then, for single-session training, execute:
```bash
$ python3 train.py \
    --config configs/baselines.yaml \
    --no_slurm
```

If you have access to a multi-GPU SLURM system, remove the `--no_slurm` flag to utilize it.
To switch to multi-subject training, add the `--multi_subject` flag.

### Spikachu training
Similar to the previous script, update the config file with the dataset paths and set your batch size according to your system.
Then run:
```bash
$ python3 train.py \
    --config configs/spikachu.yaml \
    --no_slurm
```
If you have access to a multi-GPU SLURM system, remove the `--no_slurm` flag to utilize it.
To switch to multi-subject training, add the `--multi_subject` flag.

#### Notes:
- For debugging purposes, you may want to use a CPU instead of a GPU. If so, set `model:setup:backend` to `torch` instead of `cupy` in `configs/spikachu.yaml`.
- To train on specific indices from the datasets, set the `data:lock_idcs` to a list of the desired indices in the configuration file.

## Evaluation

By default, trained model weights are stored under `trained_models/`.
To test a trained model, you can run the following:
```bash
$ python3 test.py \
    --config configs/spikachu.yaml \
    --no_slurm \
    --checkpoint_path trained_models/YOUR_CHECKPOINT.pt
```
After the evaluation completes, you can find visualizations under the `vis_logs/` directory, and metrics under `trained_models/test_results/`.
If you have access to a multi-GPU SLURM system, remove the `--no_slurm` flag to utilize it.
For multi-subject evaluation, add the `--multi_subject` flag.

# Acknowledgments

This work uses [SpikingJelly](https://github.com/fangwei123456/spikingjelly) as its spiking neural network framework.
It also uses code from [POYO](https://github.com/neuro-galaxy/poyo).


# Citation

If you use any part of this project, please consider citing our work as follows:
```bibtex
@inproceedings{
    mentzelopoulos2025,
    title={A Scalable, Causal, and Energy Efficient Framework for Neural Decoding with Spiking Neural Networks},
    author={Mentzelopoulos, Georgios and Asmanis, Ioannis and Kording, Konrad and Dyer, Eva L and Daniilidis, Kostas and Vitale, Flavia},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=oAbaGU9N1X}
}
```