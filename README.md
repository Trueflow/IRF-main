# IRF-main
Unity ML-Agents-based Multi-Agent Reinforcement Learning Tool

## Overview

This repository provides a multi-agent reinforcement learning framework built on a **pre-built Unity ML-Agents environment**.

Since the Unity environment has already been built, users do **not** need to install the Unity project or Unity-side ML-Agents package separately in order to run experiments. Only the Python-side dependencies are required.

## 1. Environment Setup

### 1.1. Python Version

This project is designed for **Python 3.10.12**.

Create a file named `.python-version` in the project root directory with the following content:

```text
3.10.12
````

### 1.2. Virtual Environment Setup with `uv`

Open a terminal such as Windows Command Prompt, PowerShell, or Anaconda Prompt, and move to the project root directory.

```bash
cd (IRF-main folder path)
uv venv --python 3.10.12
```

### 1.3. Install Base Dependencies

Install the base dependencies defined in `pyproject.toml`:

```bash
uv sync
```

The base dependencies include:

* `mlagents==1.1.0`
* `mlagents-envs==1.1.0`
* `python-box==7.3.2`

### 1.4. Install PyTorch

PyTorch should be installed according to the user's hardware environment.
Choose **one** of the following options.

#### Option 1. CPU-only

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
```

#### Option 2. NVIDIA GPU (CUDA 11.8)

This option is recommended for reproducing the original setup.

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
```

#### Option 3. NVIDIA GPU (CUDA 12.1)

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
```

> Note: Install **only one** PyTorch option.

---

## 2. Running the Code

### 2.1. Run `main.py`

After the environment setup is complete, run:

```bash
uv run python src/main.py
```

This command runs the default **LIIR vs. RSA** setting.

### 2.2. Available Frameworks and Algorithms

The following frameworks and algorithms are currently available.

#### Frameworks

* `marl`
  Used for general multi-agent environments.

* `rsa`
  An agent that selects actions uniformly at random.
  It is used as the opponent of the learning algorithm.

  * `rsa` should be assigned only to the Red team.

#### Algorithms

* `liir`: Learning Individual Intrinsic Reward in MARL (LIIR)
* `coma`: Counterfactual Multi-Agent Policy Gradients (COMA)
* `poca`: Multi-Agent Posthumous Credit Assignment (MA-POCA)
* `cds`: Celebrating Diversity in Shared Multi-Agent Reinforcement Learning (CDS)
* `emc`: Episodic Multi-Agent Reinforcement Learning with Curiosity-driven Exploration (EMC)

---

## 3. Optional Command-Line Arguments

The following command-line arguments are optional.

* `--workerid`
  Required when running multiple prompts or instances simultaneously.
  Set this value between `0` and `10`.
  Default: `0`

* `--graphic`
  Set this to `True` if you want to visually observe the training process.
  Default: `False`

Example:

```bash
uv run python src/main.py --workerid 2 --graphic False
```

---

## 4. Notes

* This repository uses a **pre-built Unity environment**, so Unity Editor installation is not required for running experiments.
* However, the Python-side ML-Agents packages must remain compatible with the built environment.
* The recommended configuration for reproduction is:

  * Python `3.10.12`
  * `mlagents==1.1.0`
  * `mlagents-envs==1.1.0`
  * `python-box==7.3.2`
  * PyTorch `2.2.1` with CUDA `11.8`

---

## 5. Repository Structure

A typical project structure is as follows:

```text
IRF-main/
├─ src/
│  └─ main.py
├─ pyproject.toml
├─ .python-version
├─ README.md
└─ ...
```

If your actual entry point is different from `src/main.py`, update the execution command accordingly.



