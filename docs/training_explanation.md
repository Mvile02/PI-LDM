# PI-LDM Training & Comparative Analysis

The document explains the mechanics of the current training script ([train.py](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/train.py)) and how it functionally compares to the reference implementation from SOKOLOV ([diffusion_trajectory.py](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/SOKOLOV-diffusion-trajectory-generation-main/diffusion/diffusion_trajectory.py)).

---

## How The Current Training Works

The training script establishes a **Physics-Informed Latent Diffusion Model (PI-LDM)** using a forward/reverse stochastic process driven by the [PILDMTrainer](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/train.py#10-81) class:

1. **Forward Diffusion (Adding Noise)**  
   During the [train_step()](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/train.py#49-81) method, a clean batch of 4D trajectories ($x_0$ of shape `[batch_size, 4, 200]`) is drawn from the dataset loader. A random timestep $t$ is selected, and statistical Gaussian noise $\epsilon$ is injected into the trajectory using the [add_noise()](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/train.py#36-42) function. It operates via the closed-form diffusion formula: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$. 
   Because the precomputed $\bar{\alpha}_t$ parameters are 1D arrays, the algorithm uses PyTorch's `.view(-1, 1, 1)` function. This reshapes the 1D time variable into a 3D tensor (e.g., `[batch_size, 1, 1]`) allowing it to smoothly broadcast (multiply across) the entire 3D trajectory matrix uniformly without dimension mismatch errors.

2. **Reverse Denoising (Neural Representation)**  
   The neural network—the **Conditional 1D U-Net** ([ConditionalUNet1D](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/model.py#20-91) in [model.py](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/model.py))—absorbs the noisy state $x_t$, the timestep $t$, and the context conditions $c$ (e.g., categorical mappings for Airport ID and Aircraft Type). Its objective is to predict the exact noise $\epsilon_{\theta}$ that was injected.
   
   Internally, the [ConditionalUNet1D](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/model.py#20-91) architecture relies on:
   - A `time_mlp` block utilizing [SinusoidalPositionEmbeddings](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/SOKOLOV-diffusion-trajectory-generation-main/diffusion/v1.5/diffusion_trajectory.py#156-169) to encode temporal diffusion steps.
   - A `cond_mlp` processing contextual constraints.
   - A **Simplified Encoder** (`enc1`, `enc2`): Composed of two downsampling `nn.Conv1d` layers extracting hierarchical temporal trajectory features.
   - A middle deep associative **Bottleneck** layer.
   - A **Simplified Decoder** (`dec1`, `dec2`): Upsampling via `nn.ConvTranspose1d` layers utilizing skip connections extracted from the encoder branch to accurately reconstruct the original sequence dimensionality.
   
   *(Why "Simplified"? Currently, the PI-LDM relies on a stripped-down Conv1D skeleton to rapidly prove math convergence on your 200-waypoint dataset. In stark contrast, SOKOLOV's [diffusion_trajectory.py](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/SOKOLOV-diffusion-trajectory-generation-main/diffusion/diffusion_trajectory.py) algorithm utilizes an exceptionally heavy [Unet1D](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/SOKOLOV-diffusion-trajectory-generation-main/diffusion/v1.5/diffusion_trajectory.py#212-279) architecture containing multi-stage [ResidualBlock](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/SOKOLOV-diffusion-trajectory-generation-main/diffusion/v1.5/diffusion_trajectory.py#170-211) modules, `GroupNorm`, and `SiLU` activations inherently engineered and heavily parameter-optimized for scaling large 512-length car trajectories on A100 server GPUs).*

3. **Dual-Loss Function (The Physics Integration)**  
   The [PILDMTrainer](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/train.py#10-81) evaluates gradients utilizing a robust two-part loss function:
   - **Diffusion Loss (MSE):** The base `nn.MSELoss()` calculates the objective Mean Squared Error between the true noise $\epsilon$ and predicted noise $\epsilon_{\theta}$.
   - **Physics Loss (BADA):** At *every single optimization step*, the network executes the [predict_x0()](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/train.py#43-48) method to algebraically reconstruct the hypothetical clean trajectory $\hat{x}_0$ from the predicted noise parameters. This $\hat{x}_0$ is dynamically passed into the [PhysicsLoss](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py#5-95) module ([physics.py](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py)), which computes structural residuals based on Aerodynamic Equations of Motion ([eom_loss](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py#28-43)), Power Demand limits ([energy_loss](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py#44-51)), and Flight Envelope boundaries ([envelope_loss](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py#52-62)). 
   
   The optimizer calculates the backward pass incorporating the summed weights: $\mathcal{L}_{Total} = \mathcal{L}_{Diff} + \lambda_1 \mathcal{L}_{EOM} + \lambda_2 \mathcal{L}_{Energy} + \lambda_3 \mathcal{L}_{Envelope}$.

---

## Comparison with SOKOLOV's Implementation

### 1. Similarities

- **Core Generative Approach:** Both architectures utilize 1D U-Nets to progressively reconstruct trajectories over dimensional sequences (e.g., `model.py:ConditionalUNet1D` vs SOKOLOV's `diffusion_trajectory.py:Unet1D`). They both inject sequential knowledge mimicking FiLM-style conditioning (passing context to the network hidden layers via time and label Scale/Shift embeddings).
- **The "Predict $x_0$" Mechanism:** In both algorithms, physics constraints cannot be geometrically evaluated on pure Gaussian distributions. Both models temporarily estimate the clean trajectory $\hat{x}_0$ inside the main step using the equation $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}} \cdot \epsilon_{\theta}}{\sqrt{\bar{\alpha}}}$ to evaluate kinematic boundary violations (visible in `PILDMTrainer.predict_x0()` vs SOKOLOV's [train_pid()](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/SOKOLOV-diffusion-trajectory-generation-main/diffusion/v1.5/diffusion_trajectory.py#542-697) block).

### 2. Key Operational Differences

#### A. Depth of Physical Accuracy
- **SOKOLOV:** Employs rudimentary 1D kinematic heuristics. Their internal loss functions penalize simple properties like absolute Maximum Acceleration thresholds (`loss_asym`), mathematical Jerk variance constraints (`loss_jerk`), and elementary Distance Consistency checks (`loss_dist`).
- **The PI-LDM:** Enforces strict multi-axial **Aerodynamic BADA constraints**. Because aircraft operate in 3D physics, the [PhysicsLoss](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py#5-95) module actively evaluates multi-dimensional variables (Altitude, Velocity, Path Angles), enforcing practical thrust-to-drag power curves and intrinsic sink/bank limits unavailable in simplistic vehicular 1D kinematics.

#### B. Architectural Separation of Concerns
- **SOKOLOV:** Computes physical penalty components utilizing raw, sprawling tensor math written directly inline internally within a massive monolithic 200-line [train_pid](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/SOKOLOV-diffusion-trajectory-generation-main/diffusion/v1.5/diffusion_trajectory.py#542-697) loop block (e.g., formulating jerk variance manually via `torch.var` and `torch.diff`). Their entire architectural U-Net structure is also profoundly nested within this single massive script file.
- **The PI-LDM:** The methodology is abstracted entirely into a highly cohesive, organized layout. The analytical [PhysicsLoss](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py#5-95) functions act as a designated PyTorch `nn.Module` class encapsulating separated functions for [eom_loss](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py#28-43) and [energy_loss](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/physics.py#44-51). The U-Net encoder/decoder resides securely inside [model.py](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/pi_ldm/src/model.py), establishing a mathematics-agnostic implementation rendering the overall training loop simple and easily refactored.

#### C. Systemic Role of Guidance
- **SOKOLOV:** Extremely reliant on *Inference-Time Control*. During generation processing ([sample()](file:///c:/Users/usuario/Desktop/Delft/TFM/Code/SOKOLOV-diffusion-trajectory-generation-main/diffusion/v1.5/diffusion_trajectory.py#403-498)), their sequence calculates gradient adjustments manually on the fly to artificially push the unconstrained vehicle vector backwards towards a reliable trajectory. Experimental losses programmed into their offline training phase mainly provide basic data regularization rather than absolute control limits.
- **The PI-LDM:** Aggressively integrates heavy aerodynamic limiters fundamentally into the neural structure's latent space through the Offline Training pipeline scaling via predefined parameters ($\lambda_{1,2,3}$). Due to heavy offline optimization, the network intrinsically produces flyably safe trajectory bounds, expediting sampling speeds whilst maximizing dependability.
