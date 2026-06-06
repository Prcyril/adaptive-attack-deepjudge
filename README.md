# Adaptive Attack DeepJudge

Adversarial machine learning project for CIFAR-10 focused on four security tasks:

- Module backdoor attack
- Reverse engineering of a hidden trigger
- Data-free adaptive attack against DeepJudge-style fingerprinting
- Invisible image watermarking

This repository contains the code, notebook, and experiment assets used for a CSIT375 assignment submission.

## Overview

The project explores how image classifiers can be attacked, analyzed, and protected in different ways:

- `train_bad_module.py` trains a backdoor module that forces a target prediction when a trigger is present.
- `reverse_engineer.py` reconstructs a hidden trigger and mask for a chosen target class.
- `adaptive_attack.py` defines an input transformation pipeline for an adaptive attack against model fingerprinting.
- `watermark.py` trains an autoencoder-based watermarking pipeline that embeds and recovers invisible marks in CIFAR-10 images.

The codebase also includes reusable model, dataset, attack, and training utilities under `codebase/`.

## Repository Structure

```text
.
├── adaptive_attack.py
├── assignment2_CSIT375.ipynb
├── assignment2_CSIT375.pdf
├── reverse_engineer.py
├── train_bad_module.py
├── watermark.py
├── codebase/
│   ├── attacks/
│   ├── autoencoder/
│   ├── classifiers/
│   ├── datasets/
│   ├── visualization/
│   ├── exp_setup.py
│   ├── model_trainer.py
│   ├── setup.py
│   └── utils.py
├── images/
│   ├── acceptable_wm_*.png
│   └── reversed_trigger_*.png
├── data/
└── out/
```

## Main Components

### 1. Module Backdoor Attack

`train_bad_module.py` trains a malicious module on CIFAR-10 using a red square trigger placed at the bottom-right corner of the image. The goal is to redirect predictions to a fixed target label while keeping clean accuracy high.

Key details:

- Trigger: red `5x5` patch
- Trigger location: bottom-right corner
- Poison target: class `5`
- Poison rate: `0.5`

### 2. Reverse Engineering

`reverse_engineer.py` optimizes a learnable trigger and mask to recover a pattern that causes a model to predict a selected target class.

Key details:

- Learns both trigger content and trigger mask
- Uses cross-entropy plus L1 regularization on the mask
- Runs on CIFAR-10 samples excluding the target class

### 3. Adaptive Attack

`adaptive_attack.py` implements an augmentation pipeline intended to weaken fingerprint-based model verification by perturbing inputs while preserving their semantic content.

Current transformations:

- Small random affine transform
- Strong horizontal flip probability
- Gaussian noise injection

### 4. Image Watermarking

`watermark.py` trains an autoencoder and decoder jointly to embed an invisible watermark into images and recover it later.

Key details:

- Joint training with `AdamW`
- Mixed clean and watermarked batches
- Binary cross-entropy for watermark recovery
- Mean squared error penalty to keep perturbations subtle

## Requirements

This project is Python-based and depends primarily on PyTorch and common scientific Python libraries.

Suggested packages:

```bash
pip install torch torchvision numpy matplotlib scipy seaborn scikit-learn tqdm librosa
```

If you are using Jupyter or Colab, also install:

```bash
pip install notebook ipykernel
```

## Setup

### Local Setup

1. Clone the repository:

```bash
git clone https://github.com/Prcyril/adaptive-attack-deepjudge.git
cd adaptive-attack-deepjudge
```

2. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install torch torchvision numpy matplotlib scipy seaborn scikit-learn tqdm librosa notebook ipykernel
```

### Data and Output Files

The notebook expects:

- CIFAR-10 data under `data/`
- pretrained assignment assets under `out/`

Based on the assignment notebook, the following files are expected in `out/`:

- `task1_model/`
- `task2_model/`
- `task3_model/`
- `fingerprints.pt`

Note:

- `data/` and `out/` are excluded from Git in `.gitignore`
- CIFAR-10 may be downloaded automatically by `torchvision`
- pretrained model checkpoints are not included in the repo unless you add them manually

## Running the Project

### Option 1: Use the Notebook

The easiest way to reproduce the assignment workflow is:

```bash
jupyter notebook assignment2_CSIT375.ipynb
```

The notebook walks through:

- loading pretrained task models
- evaluating clean and poisoned performance
- visualizing triggers and outputs
- running the reverse engineering workflow
- evaluating the adaptive attack
- training and testing the watermarking pipeline

### Option 2: Use Individual Scripts

The main scripts define the core training and attack functions and are intended to be imported or executed from a larger workflow such as the notebook.

Examples:

```python
from train_bad_module import train as train_bad_module
from reverse_engineer import reverse_trigger
from watermark import train as train_watermark
from adaptive_attack import transform
```

Because the repository is assignment-oriented, the notebook is currently the best entry point for end-to-end execution.

## Project Outputs

The repository includes example output artifacts such as:

- reversed trigger visualizations in `images/`
- acceptable watermark examples in `images/`
- training curves in `out/task*_model/`

Common output files:

- `train_status.png`
- `acc_trends.png`
- `fingerprints.pt`

## Models and Utilities

The reusable code under `codebase/` includes:

- `codebase/classifiers/resnet.py`: ResNet models for CIFAR-10
- `codebase/classifiers/vgg.py`: VGG variants used by the assignment
- `codebase/datasets/poisoned.py`: dataset wrapper for backdoor poisoning
- `codebase/model_trainer.py`: generic training and evaluation helper
- `codebase/attacks/`: attack implementations including PGD and CW-L2

## Important Notes

- `codebase/setup.py` and `codebase/exp_setup.py` contain default paths that appear to be from the original assignment environment. If you run the code outside Colab, update `data_dir`, `out_dir`, and device settings as needed.
- The config helpers default to `cuda:0`, so CPU-only environments may need small changes.
- Some utility code supports broader experiments than this assignment, so not every helper is used directly by the four main scripts.

## Suggested GitHub About

Short description:

`Adversarial ML project on backdoor attacks, trigger reverse engineering, adaptive DeepJudge attacks, and image watermarking.`

## Author

Prayas Cyril Dulal

GitHub: [Prcyril](https://github.com/Prcyril)

## License

No license has been added yet. If you want others to reuse this project, add a license such as MIT.
