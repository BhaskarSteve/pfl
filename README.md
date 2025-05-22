# Optimal Strategies for Federated Learning Maintaining Client Privacy

Implementation of a federated learning framework with local differential privacy guarantees for each client. This is the implementation of our [arXiv]() preprint. The implementation supports training on raw images as well as handcrafted scattering transform features.

## Implementations

- **Federated Learning**: Distributed training across multiple clients with FedAvg aggregation
- **Differential Privacy**: Privacy-preserving training using Opacus library
- **Scattering Features**: Support for training on ScatterNet features using Kymatio
- **Datasets**: MNIST, Fashion-MNIST, and CIFAR-10 support
- **Data Split**: Both IID and non-IID data distribution strategies

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management. Install uv first, then set up the project dependencies:

```bash
# Clone the repository
git clone https://github.com/BhaskarSteve/pfl
cd pfl

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip sync requirements.lock
```


## Usage

The framework consists of three main components:

### 1. Standard Federated Learning (`main.py`)

Trains CNN models directly on image datasets with federated learning and optional differential privacy.

**Basic Usage:**
```bash
# Train on MNIST with default settings
python main.py --dataset mnist

# Train on CIFAR-10 with non-IID data distribution
python main.py --dataset cifar --partition non-iid

# Enable differential privacy
python main.py --dataset mnist --epsilon 1.0 --delta 1e-5

# Custom federated learning setup
python main.py \
    --dataset mnist \
    --global_ep 10 \
    --num_users 100 \
    --frac 0.1 \
    --local_ep 5 \
    --local_bs 32 \
    --lr 0.01
```

**Key Arguments:**
- `--dataset`: Dataset to use (`mnist`, `fmnist`, `cifar`)
- `--global_ep`: Number of global communication rounds (default: 2)
- `--num_users`: Number of federated clients (default: 10)
- `--frac`: Fraction of clients participating per round (default: 1.0)
- `--partition`: Data distribution strategy (`iid`, `non-iid`)
- `--local_ep`: Local training epochs per client (default: 1)
- `--local_bs`: Local batch size (default: 64)
- `--lr`: Learning rate (default: 0.1)
- `--disable_dp`: Disable differential privacy
- `--epsilon`: DP epsilon parameter (default: 2.93)
- `--delta`: DP delta parameter (default: 1e-5)
- `--activation`: Activation function (`relu`, `tempered`)

### 2. Feature Extraction (`extract_features.py`)

Extracts scattering transform features from datasets and saves them for later use.

**Basic Usage:**
```bash
# Extract features from MNIST
python extract_features.py --dataset mnist

# Extract features from CIFAR-10 with custom scattering parameters
python extract_features.py \
    --dataset cifar \
    --depth 3 \
    --rotations 16 \
    --batch_size 128
```

**Key Arguments:**
- `--dataset`: Dataset to process (`mnist`, `fmnist`, `cifar`)
- `--batch_size`: Batch size for feature extraction (default: 64)
- `--depth`: Scattering transform depth (default: 2)
- `--rotations`: Number of rotations in scattering transform (default: 8)

### 3. Feature-based Federated Learning (`features_main.py`)

Trains models on pre-extracted scattering features instead of raw images.

**Basic Usage:**
```bash
# First extract features
python extract_features.py --dataset mnist

# Then train on features with linear model
python features_main.py --dataset mnist --model linear

# Train with CNN on scattering features
python features_main.py \
    --dataset mnist \
    --model cnn \
    --global_ep 5 \
    --lr 0.1 \
    --num_groups 9
```

**Key Arguments:**
- All arguments from `main.py` plus:
- `--model`: Model architecture (`linear`, `cnn`)
- `--norm`: Normalization type (`group`)
- `--num_groups`: Number of groups for group normalization (default: 27)

## Citation

```bibtex
@article{bhaskar2025optimal,
  title={Optimal Strategies for Federated Learning Maintaining Client Privacy},
  author={Bhaskar, Uday and Srivastava, Varul and Vummintala, Avyukta Manjunatha and Manwani, Naresh and Gujar, Sujit},
  journal={arXiv preprint arXiv:2501.14453},
  year={2025}
}
```
