# Generation of TMA Landing Trajectories Using Diffusion Models

This repository contains the codebase for generating aircraft trajectories in the Terminal Maneuvering Area (TMA) using a Diffusion Model. The model operates in a Kinematic State Space to ensure location-agnostic generation and robust learning of flight dynamics.

## How It Works

The model learns to synthesize flights based on kinematic features like track angle, groundspeed, and altitude over time. It uses a **1D Convolutional U-Net** to iteratively denoise a random sequence of these parameters, shaping it into a realistic, flyable trajectory. For the sake of efficiency, this architecture directly injects a global absolute temporal coordinate axis into the data.

The core pipeline involves:
1. **Data Preprocessing**: Converting raw spatial trajectories into a continuous kinematic representation (`scripts/` and `pi_ldm/src/dataset.py`).
2. **Diffusion Process**: A forward process adds Gaussian noise to real trajectories, and a reverse process (the U-Net) learns to denoise them (`pi_ldm/src/model.py` and `pi_ldm/src/train.py`).
3. **Sampling & Evaluation**: Generating new synthetic flights and validating their realism, diversity, and fidelity (`pi_ldm/src/sample.py` and `pi_ldm/src/evaluation.py`).

## Physics-Informed Regularization (Experimental)

A physics-informed training option is available within this repository (`pi_ldm/src/physics.py`). This module attempts to enforce aerodynamic boundaries and physical constraints directly into the loss function during training. 

> **Note:** While the physics-informed architecture is fully implemented and available for experimentation, current tests indicate that it does not achieve great results compared to the purely data-driven approach. Explicit physical regularization often constrained the model's natural distributional learning, and it is left as a foundation for future fine-tuning.

## Repository Structure

- `pi_ldm/src/` - Core source code:
  - `model.py`: Defines the 1D U-Net and Diffusion components.
  - `train.py`: Contains the training loop for the diffusion model.
  - `sample.py`: Handles reverse diffusion to generate synthetic trajectories.
  - `evaluation.py`: Computes fidelity, diversity, and memorization metrics.
  - `physics.py`: Contains the experimental physical regularization functions.
  - `dataset.py`: Handles data loading and batching.
- `pi_ldm/bin/` - Execution scripts (PowerShell wrappers).
- `scripts/` - Preprocessing and plotting utilities.
- `data/` - Target directory for storing raw and processed flight datasets.

## Installation and Setup

1. Ensure you have Python installed.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can use the provided PowerShell wrappers in `pi_ldm/bin/` to easily start the processes:

### 1. Training the Model
To train the model from scratch, simply run:
```powershell
.\pi_ldm\bin\train.ps1
```
*(This sets up the `PYTHONPATH` correctly and executes `src/train.py`)*

### 2. Evaluating the Model
To sample from a trained model and generate evaluation metrics:
```powershell
.\pi_ldm\bin\eval.ps1
```

*(Alternatively, you can run the python scripts directly from the `pi_ldm/src/` folder provided your `PYTHONPATH` is set to the repository root).*
