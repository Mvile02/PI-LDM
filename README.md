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

## Usage / Replicating the Results

To fully replicate the results of the thesis from scratch, follow this step-by-step pipeline. 

> **Note:** Pre-trained model weights (`.pth` files) are already available in the `pi_ldm/models/` directory, so you can skip directly to step 4 (Sampling & Evaluation) if you do not want to retrain the model.

### 1. Extract the Data
First, extract and preprocess the raw flight data into a usable kinematic tensor using the dataset builder script:
```bash
python scripts/dataset_builder.py
```
*(This extracts kinematics such as track, groundspeed, and altitude, handles resampling, and saves `.npy` arrays for training).*

### 2. Cluster the Trajectories (Optional)
If you wish to analyze specific approach clusters or filter out anomalous paths before training, run the clustering script:
```powershell
python scripts/trajectory_clusterer.py
```

### 3. Training the Model
To train the model from scratch on the processed data, execute the training script from the root directory:
```bash
python pi_ldm/src/train.py
```
*(Ensure your `PYTHONPATH` is set to the repository root if you encounter import issues).*

### 4. Sampling & Evaluation
To sample synthetic trajectories from the trained model (or the pre-trained `.pth` models) and generate the evaluation metrics:
```bash
python pi_ldm/src/sample.py
```
*(You can also use `python pi_ldm/src/evaluation.py` to run additional fidelity, diversity, and memorization tests).*
