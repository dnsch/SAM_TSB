# Sharpness-Aware Minimization Time Series Benchmark (SAM_TSB)

This repository contains code for experiments exploring the relationship between [Sharpness Aware Minimization (SAM)](https://arxiv.org/abs/2010.01412) methods and generalization error. The code allows you to test multiple model architectures, regularization techniques, and datasets to evaluate the effectiveness of SAM and its variants.

---

## Instructions

#### Cloning the repository

To clone this repository with the necessary modified submodules use the
following command in your terminal:

```bash
git clone --recurse-submodules [https://github.com/dnsch/sam_experiments.git](https://github.com/dnsch/SAM_TSB.git)
```

#### Replicating Results

The easiest way to replicate results is to set up a docker container on a
machine with Cuda support via the Dockerfile and the following command:

```bash
docker build  -t samtsb_gpu . 
```

This will build a Docker container with the same configurations we used to
conduct our experiments and download all datasets. 

Once it is set up, you can then run scripts via:

```bash
docker run --rm --gpus all   --user $(id -u):$(id -g)   -v $(pwd):/app   samtsb_gpu   bash experiments/single_split/patchtst/run_selection.sh
```

for example. You can find all scripts in the respective multi_split or
single_split model folders. The above command would launch the patchtst model
with all SAM variants on our tested datasets ETTh1, ETTh2, exchange_rate and
national_illness.

The results of these runs are then saved under the respective folders in the
/results directory.

You can also run the models individually with many arguments for hyperparemeter
tuning.

```bash
docker run --rm --gpus all   --user $(id -u):$(id -g)   -v $(pwd):/app   samtsb_gpu  python experiments/single_split/patchtst/patchtst.py --dataset ETTh1
```

for example launches the base patchtst model with default arguments using the
ETTh1 dataset. The help flag at the end of the command:

```bash
docker run --rm --gpus all   --user $(id -u):$(id -g)   -v $(pwd):/app   samtsb_gpu  python experiments/single_split/patchtst/patchtst.py --dataset ETTh1 --help
```

will display all possible arguments for the specific model.

To plot the loss landscape 

run for example:


```bash
docker run --rm --gpus all   --user $(id -u):$(id -g)   -v $(pwd):/app   samtsb_gpu   python third_party/utils/loss_landscape/plot_surface.py --x=-1:1:30 --y=-1:1:30 --vmax=2 --vlevel=0.001 --model_file path/to/final_model_s1.pt --dir_type weights --xnorm filter --ynorm filter  --plot --model patchtst --ts_dataset_name ETTh1 -c --loss_name mse
```


to plot the loss landscape of the patchtst model using the ETTh1 dataset. For
details check the original repository (see below).


#### Setting Up the Virtual Environment

Alternatively, you can run the code using a virtual environment.

To run the code, you must first set up a Python environment and install the project via:

To set up a python virtual environment with the necessary packages, you can use the `venv` command inside the cloned SAM_TSB directory:

```bash
python -m venv .venv # or python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project includes packages that help with downloading the datasets
and those that are needed to plot the loss surface using [loss-landscape](https://github.com/tomgoldstein/loss-landscape).

You can also use [pyenv](https://github.com/pyenv/pyenv) or [uv](https://github.com/astral-sh/uv) to set up the virtual environment, which also allow you to specify a Python version and might be faster.

---

#### Downloading the Data

To download the autoformer datasets, run the following command inside the virtual environment:

```bash
python scripts/download_autoformer_dataset.py
```

If the scripts do not work, you can download the datasets manually using the following links:

[Autoformer datasets (Google Drive)](https://drive.google.com/uc?id=1alE33S1GmP5wACMXaLu50rDIoVzBM4ik)

Once downloaded, place the Autoformer ```.csv``` files into the 📁 ```data/autoformer_datasets```.

---

#### Running the Code

Once the data is downloaded and placed in the correct directories, you can run the experiments.

The general command to train and test a model is:

```bash
python code/experiments/[type]/[model]/main.py
```

where `[model]` refers to the model you want to use and `[type]` is the scenario
(multi split/single split)

For example, to run the **SAMformer** model using the single split scenario with the **ETTh1** dataset and seed `1`, use:

```bash
python code/experiments/single_split/samformer/main.py --dataset ETTh1 --seed 1
```

To run the same experiment *SAM*, append the `sam = True` flag:

```bash
python code/experiments/single_split/samformer/main.py --dataset ETTh1 --seed 1 --sam True
```

The script runs on the CPU by default. If you have a CUDA compatible GPU, append your CUDA device via the `--device` flag:

```bash
python code/experiments/single_split/samformer/main.py --dataset ETTh1 --seed 1 --sam True --device cuda:0
```

This script will generate training and testing statistics plots. The output includes:

* A log file containing the training configuration and progress.
* A 📁 `saved_models/` folder containing the trained model for each epoch.
* A 📄 `final_model_s[seed].pt` file – the best model (i.e., the one with the lowest validation error).

All outputs are saved in the 📁 [`results/`](results/) directory. Files are named according to model type and arguments to help distinguish between experiment configurations.

---

##### Viewing Available Arguments

To view all available options for a given experiment, use the `--help` flag:

```bash
python code/experiments/[type]/[model]/main.py --help
```

---

#### Exploring the Loss Landscape

This repository includes a modified version of [loss-landscape](https://github.com/tomgoldstein/loss-landscape) for approximations of the loss surface.

To generate a loss surface plot in a `1x1` region around a trained model’s local minimum with a resolution of `20x20`, run:

```bash
python third_party/utils/loss_landscape/plot_surface.py --mpi --cuda --x=-1:1:20 --y=-1:1:20 \
--vmax=0.5 --vlevel=0.01 \
--model_file results/samformer/ETTh1/seq_len_512_pred_len_96_bs_256_rho_0.5/final_model_s1.pt \
--dir_type weights --xnorm filter --xignore biasbn \
--ynorm filter --yignore biasbn --plot \
--dataset samformer_datasets --model samformer \
--loss_name mse --dataset_name ETTh1
```

The resulting plot will be saved in the 📁 `plots/loss_surface/` directory of the model.

For usage of CUDA, append the `--cuda` flag.

Use the `--vmax` flag to cap the maximum loss value, which is helpful when comparing different landscapes. For an explanation of all available options, use the `--help` flag.

All argument parsing is handled in 📄 [`third_party/utils/loss_landscape/plot_surface.py`](third_party/utils/loss_landscape/plot_surface.py), which is a modified version of the original. Outdated functions (mostly related to MPI) were updated to ensure compatibility. However, this version was **not tested** with MPI or multi-GPU setups, so some original functionality may not be preserved.

---

## Repository Structure

```
.
├── 📁 src/        # Training routines, model code  
├── 📁 experiments/# Experiment logic
├── 📁 data/       # Downloaded datasets  
├── 📁 third_party/# External repositories (submodules)
├── 📁 results/    # Trained models and experiment outputs  
└── 📁 scripts/    # Dataset download scripts  
```

---

### 📁 [`src/`](src/)

Inspired by [LargeST](https://github.com/liuxu77/LargeST), this folder includes:

  * 📁 [`src/base/`](src/base/): includes base classes:
    * 📄 [`engine.py`](src/base/engine.py)
    * 📄 [`model.py`](src/base/model.py)
  * 📁 [`src/engines/`](src/engines/): contains custom training/test logic for each model.
  * 📁 [`src/models/`](src/models/): defines the model architectures.

The files [`samformer_engine.py`](src/engines/samformer_engine.py) and [`samformer.py`](src/models/samformer.py) are modified versions from the original [SAMformer PyTorch implementation](https://github.com/romilbert/samformer/tree/main/samformer_pytorch).

To add a new model:

1. Create a custom 📄 `[model]_engine.py` in [`engines/`](src/engines/) that inherits from [`base/engine.py`](src/base/engine.py).
2. Add a 📄 `[model].py` in [`models/`](src/models/) that inherits from [`base/model.py`](src/base/model.py).
3. Create a 📁 `experiments/[type]/[model]/` folder with a 📄 `model.py`.

You might want to add an argument group to [`src/utils/args.py`](src/utils/args.py).

---

### 📁 [`data/`](data/)

This folder stores the downloaded datasets:

* [SAMformer Datasets (Google Drive)](https://drive.google.com/uc?id=1alE33S1GmP5wACMXaLu50rDIoVzBM4ik)

---

### 📁 [`third_party/`](third_party/)

Contains additional repositories (forks) used in this project:

* [loss-landscape](https://github.com/dnsch/loss_landscape)
* [pyhessian](https://github.com/dnsch/PyHessian)

---

### 📁 [`results/`](results/)

This directory stores:

* Trained model checkpoints
* Plots generated from training/testing runs

Each run is saved in a separate folder named after the model and its configuration.

---

### 📁 [`scripts/`](scripts/)

Contains helper scripts to download the SAMformer and datasets automatically.

---
