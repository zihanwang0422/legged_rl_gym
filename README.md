# AMP for Hardware — Adversarial Motion Priors for Quadruped Locomotion

Implementation of [Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions](https://bit.ly/3hpvbD6).  
Trains natural locomotion policies for quadruped robots using ~4.5 seconds of reference MoCap data, with full support for Isaac Gym training, Sim-to-Sim validation (MuJoCo), and Sim-to-Real deployment on Unitree Go1 / A1.

> **中文文档**: [README_CN.md](README_CN.md)

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training](#training)
- [Play](#play)
- [Sim-to-Sim (MuJoCo Validation)](#sim-to-sim-mujoco-validation)
- [Sim-to-Real (Hardware Deployment)](#sim-to-real-hardware-deployment)
- [AMP Algorithm Overview](#amp-algorithm-overview)
- [Reference Motion Data Format](#reference-motion-data-format)
- [Observation & Action Space](#observation--action-space)
- [Troubleshooting](#troubleshooting)

---

## Installation

### 1. Create Conda Environment

```bash
conda create -n amp_hw python=3.8
conda activate amp_hw
```

### 2. Install PyTorch (CUDA 11.3)

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    tensorboard==2.8.0 pybullet==3.2.1 opencv-python==4.5.5.64 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 3. Install Isaac Gym Preview 3

Download **Preview 3** (Preview 2 is not compatible) from [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym).

```bash
cd isaacgym/python && pip install -e .
# Verify installation
cd examples && python 1080_balls_of_solitude.py
```

### 4. Install rsl_rl (PPO backend)

```bash
cd rsl_rl && pip install -e .
```

### 5. Install legged_gym

```bash
cd ../ && pip install -e .
```

### 6. Set Library Path

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Project Structure

```
legged_rl_gym/
├── datasets/
│   ├── mocap_motions/          # Reference motion files (JSON) used by the AMP discriminator
│   │   ├── trot0.txt           # Trotting gait 0
│   │   ├── trot1.txt           # Trotting gait 1
│   │   ├── pace0.txt           # Pacing gait
│   │   ├── leftturn0.txt       # Left turn
│   │   └── rightturn0.txt      # Right turn
│   └── hopturn/
│       └── hopturn.txt         # Hop-turn motion
│
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── legged_robot.py         # Base env: Isaac Gym integration, reset, reward
│   │   │   └── legged_robot_config.py  # Base config classes
│   │   ├── go1/
│   │   │   ├── go1_amp_config.py       # Go1 AMP training config
│   │   │   └── go1_amp.py              # Go1 AMP environment (inherits LeggedRobot)
│   │   └── a1/
│   │       ├── a1_amp_config.py        # A1 AMP training config
│   │       └── a1_amp.py               # A1 AMP environment
│   └── scripts/
│       ├── train.py                    # Launch training
│       ├── play.py                     # Visualize trained policy
│       └── record_policy.py            # Record policy video
│
├── rsl_rl/
│   └── rsl_rl/
│       ├── algorithms/
│       │   ├── amp_ppo.py              # AMP-PPO: joint policy + discriminator optimization
│       │   └── amp_discriminator.py    # Discriminator MLP → style reward
│       ├── datasets/
│       │   └── motion_loader.py        # AMPLoader: load & sample reference motion frames
│       └── runners/
│           └── amp_on_policy_runner.py # Main training loop
│
├── deploy/
│   ├── sim2sim2real_keyboard.py        # Unified Sim2Sim / Sim2Real keyboard control
│   ├── sim2sim2real_joystick.py        # Unified Sim2Sim / Sim2Real joystick control
│   ├── config/
│   │   ├── go1.yaml                    # Go1 deployment config
│   │   └── go2.yaml
│   └── exported_policy/                # TorchScript exported policies
│
├── resources/robots/
│   ├── go1/urdf/                       # Go1 URDF model
│   └── a1/urdf/                        # A1 URDF model
│
└── logs/                               # Training logs and checkpoints
    └── go1_amp_example/
        └── <date_time>/
            └── model_<iter>.pt
```

---

## Training

```bash
# Train Go1 AMP policy (recommended)
python legged_gym/scripts/train.py --task=go1_amp --headless

# Train A1 AMP policy
python legged_gym/scripts/train.py --task=a1_amp --headless
```

**CLI arguments:**

| Argument | Description |
|----------|-------------|
| `--task` | Task name: `go1_amp`, `a1_amp` |
| `--headless` | Disable rendering (significantly faster) |
| `--num_envs 4096` | Number of parallel environments (Go1 default: 4096) |
| `--max_iterations 500000` | Maximum training iterations |
| `--resume` | Resume from checkpoint |
| `--load_run <name>` | Specify run to load (`-1` = latest) |
| `--checkpoint <iter>` | Specify checkpoint number (`-1` = latest) |
| `--sim_device cpu` | Use CPU simulation |

**Tips:**
- Press `V` after training starts to toggle rendering — disabling it greatly speeds up simulation.
- Checkpoints are saved to: `logs/<experiment_name>/<date_time>_<run_name>/model_<iter>.pt`

---

## Play

```bash
python legged_gym/scripts/play.py --task=go1_amp
```

Loads the latest model by default. Set `load_run` and `checkpoint` in the config to specify a version.

Record a video:
```bash
python legged_gym/scripts/record_policy.py --task=go1_amp
```

---

## Sim-to-Sim (MuJoCo Validation)

Validate the policy in MuJoCo before deploying to hardware, testing behavioral consistency across physics engines:

```bash
cd deploy/

# Keyboard control
python sim2sim2real_keyboard.py --mode sim --model exported_policy/go1/policy.pt

# Joystick control
python sim2sim2real_joystick.py --mode sim --model exported_policy/go1/policy.pt
```

**Keyboard mapping:**

| Key | Action |
|-----|--------|
| `I` / `K` or Numpad `8` / `2` | Forward / Backward (step: 0.1 m/s) |
| `U` / `O` or Numpad `7` / `9` | Strafe left / right (step: 0.05 m/s) |
| `J` / `L` or Numpad `4` / `6` | Turn left / right (step: 0.1 rad/s) |
| `Space` or Numpad `5` | Emergency stop |
| `Q` or `ESC` | Quit |

---

## Sim-to-Real (Hardware Deployment)

> Confirm stable Sim-to-Sim behavior before proceeding to real hardware.

**Prerequisites:**
- Install and configure `unitree_legged_sdk` with proper network connection.
- Place the robot in a safe standing position with damping control active.

```bash
cd deploy/

# Keyboard control on hardware
python sim2sim2real_keyboard.py --mode real --model exported_policy/go1/policy.pt

# Joystick control on hardware
python sim2sim2real_joystick.py --mode real --model exported_policy/go1/policy.pt
```

**Control parameters (`UnifiedConfig`):**

| Parameter | Value |
|-----------|-------|
| Control timestep | 5 ms (200 Hz) |
| Policy inference rate | 33 Hz |
| Action scale | 0.25 |
| Observation clip | 100.0 |

**Exporting a policy (Isaac Gym → TorchScript):**

`play.py` exports the policy via `torch.jit.script` to `deploy/exported_policy/`. The exported file runs without Isaac Gym dependencies.

---

## AMP Algorithm Overview

### Core Idea

AMP (Adversarial Motion Priors) trains a policy whose motion **style** matches reference MoCap data, while simultaneously tracking task objectives (velocity commands). The discriminator output **replaces** hand-crafted motion quality rewards.

```
Total Reward = α × AMP Style Reward  +  (1 − α) × Task Reward
```

`α` is controlled by `amp_task_reward_lerp` (default: 0.3 → 70% AMP + 30% task).

### System Overview

```
┌───────────────────────────────────────────────────────┐
│                     Training Loop                     │
│                                                       │
│  Isaac Gym Env ──→ AMP Observations (s_t, s_{t+1})   │
│        ↓                       ↓                     │
│  PPO Actor-Critic        Discriminator D(s, s')       │
│        ↓                       ↓                     │
│  Task reward r_task    Style reward r_AMP             │
│         ↘                    ↙                        │
│           Total reward → GAE → Policy update          │
│                                                       │
│  Expert replay buffer ──→ Discriminator supervision   │
└───────────────────────────────────────────────────────┘
```

### Discriminator (`amp_discriminator.py`)

- **Architecture**: MLP with hidden layers `[1024, 512]`, scalar output logit
- **Input**: Concatenation of adjacent state frames $(s_t, s_{t+1})$, dimension = `observation_dim × 2`
- **Training**: Distinguishes "expert (MoCap)" from "policy-generated" transitions using WGAN-style gradient penalty (`compute_grad_pen`)
- **Style reward**:

$$r_{AMP} = \text{coef} \times \max\!\left(0,\ 1 - \tfrac{1}{4}(D(s,s') - 1)^2\right)$$

### Reference State Initialization (RSI)

- `reference_state_initialization_prob = 0.85`: On each environment reset, 85% of episodes initialize from a randomly sampled MoCap frame rather than a fixed standing pose.
- Purpose: Broadens phase-space coverage, accelerating convergence to natural gaits.

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `amp_reward_coef` | 2.0 | Style reward scaling factor |
| `amp_task_reward_lerp` | 0.3 | Task reward interpolation weight |
| `amp_discr_hidden_dims` | [1024, 512] | Discriminator hidden layers |
| `amp_replay_buffer_size` | 1,000,000 | Policy-side replay buffer size |
| `amp_num_preload_transitions` | 200,000 | Number of preloaded reference frames |
| `entropy_coef` | 0.01 | Policy entropy regularization |

---

## Reference Motion Data Format

Files are located in `datasets/mocap_motions/`, stored as JSON with a `.txt` extension.

### File Structure

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.021,
  "EnableCycleOffsetPosition": true,
  "EnableCycleOffsetRotation": true,
  "MotionWeight": 0.5,
  "Frames": [
    [frame_0_values...],
    [frame_1_values...]
  ]
}
```

- `FrameDuration`: Time between frames (seconds).
- `MotionWeight`: Sampling weight when mixing multiple motion files.
- `LoopMode: "Wrap"`: Motion loops continuously.

### Per-Frame Layout (61 dims, PyBullet joint ordering)

| Field | Dims | Index | Description |
|-------|------|-------|-------------|
| `root_pos` | 3 | 0–2 | Root position (x, y, z) |
| `root_rot` | 4 | 3–6 | Root rotation quaternion (x,y,z,w) |
| `joint_pos` | 12 | 7–18 | 12 joint angles [FR,FL,RR,RL]×3 |
| `toe_pos_local` | 12 | 19–30 | 4 foot-end local positions |
| `linear_vel` | 3 | 31–33 | Root linear velocity |
| `angular_vel` | 3 | 34–36 | Root angular velocity |
| `joint_vel` | 12 | 37–48 | 12 joint velocities |
| `toe_vel_local` | 12 | 49–60 | 4 foot-end local velocities |

> **Note**: Raw data uses PyBullet joint order `[FR, FL, RR, RL]`. `AMPLoader.reorder_from_pybullet_to_isaac()` automatically converts to Isaac Gym order `[FL, FR, RL, RR]`.

### AMP Observations Used by the Discriminator

`AMPLoader` strips `root_pos` and `root_rot`, keeping only:

```
joint_pos(12) + toe_pos_local(12) + linear_vel(3) + angular_vel(3) + joint_vel(12) = 42 dims
```

Discriminator input is two consecutive frames concatenated: $(s_t, s_{t+1})$ = **84 dims**.

---

## Observation & Action Space

### Policy Observations (Go1, 45 dims)

| Index | Dims | Content | Scale |
|-------|------|---------|-------|
| 0–2 | 3 | Base angular velocity `base_ang_vel` (x,y,z) | × 0.25 |
| 3–5 | 3 | Projected gravity vector | — |
| 6–8 | 3 | Velocity commands `[lin_vel_x, lin_vel_y, ang_vel_yaw]` | × [2.0, 2.0, 0.25] |
| 9–20 | 12 | Joint position offset `dof_pos − default_dof_pos` | × 1.0 |
| 21–32 | 12 | Joint velocities `dof_vel` | × 0.05 |
| 33–44 | 12 | Last actions | — |

### Joint Order (Isaac Gym: FL → FR → RL → RR)

| # | Joint | # | Joint |
|---|-------|---|-------|
| 1 | FL_hip_joint | 7 | RL_hip_joint |
| 2 | FL_thigh_joint | 8 | RL_thigh_joint |
| 3 | FL_calf_joint | 9 | RL_calf_joint |
| 4 | FR_hip_joint | 10 | RR_hip_joint |
| 5 | FR_thigh_joint | 11 | RR_thigh_joint |
| 6 | FR_calf_joint | 12 | RR_calf_joint |

### Actions (12 dims)

Output is a residual on the default joint positions:

```
target_angle = action_scale × action + default_dof_pos
```

`action_scale = 0.25`, PD control: kp = 80 N·m/rad, kd = 1.0 N·m·s/rad.

---

## Troubleshooting

**`ImportError: libpython3.8m.so.1.0: cannot open shared object file`**
```bash
sudo apt install libpython3.8
# or
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

**Out of GPU memory during training**  
Reduce parallel environments: `--num_envs 1024`

**Unstable contact forces (GPU + triangle mesh)**  
Use flat terrain (`mesh_type = 'plane'`), or replace `net_contact_force_tensor` with foot force sensors.

**Motor jitter on real hardware**  
Ensure `default_dof_pos` matches training config exactly. Add `action_rate` penalty (`_reward_action_rate` penalizes `||a_t − a_{t-1}||²`) and enable smooth command transitions (`smooth_command_alpha = 0.99`) in training to reduce inter-frame action discontinuities.
