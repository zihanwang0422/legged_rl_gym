# AMP for Hardware — Adversarial Motion Priors for Quadruped Locomotion

Implementation of [Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions](https://bit.ly/3hpvbD6).  
Trains natural locomotion policies for quadruped robots using ~4.5 seconds of reference MoCap data, with full support for Isaac Gym training, Sim-to-Sim validation (MuJoCo), and Sim-to-Real deployment on Unitree Go1 / Go2.

> **中文文档**: [README_CN.md](README_CN.md)

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
│   ├── mocap_motions/          # Original A1 reference motion files (JSON)
│   │   ├── trot0.txt           # Trotting gait 0
│   │   ├── trot1.txt           # Trotting gait 1
│   │   ├── pace0.txt           # Pacing gait
│   │   ├── leftturn0.txt       # Left turn
│   │   └── rightturn0.txt      # Right turn
│   ├── hopturn/
│   │   └── hopturn.txt         # Hop-turn motion
│   ├── mocap_go1/              # GO1-retargeted motion files (auto-generated)
│   ├── mocap_go2/              # GO2-retargeted motion files (auto-generated)
│   ├── retarget_mocap.py       # FK-based retargeting tool (A1 → GO1/GO2)
│   ├── verify_mocap.py         # Static geometry verification script
│   └── replay_mocap.py         # Isaac Gym mocap playback (no policy needed)
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

## Motion Data & Retargeting

### Reference Motion File Format

Files are in `datasets/mocap_motions/`, stored as JSON with a `.txt` extension.

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.021,
  "EnableCycleOffsetPosition": true,
  "EnableCycleOffsetRotation": true,
  "MotionWeight": 0.5,
  "Frames": [[frame_0_values...], [frame_1_values...]]
}
```

**Per-frame layout (61 dims, PyBullet joint order)**

| Field | Dims | Index | Description |
|-------|------|-------|-------------|
| `root_pos` | 3 | 0–2 | Root position (x, y, z) |
| `root_rot` | 4 | 3–6 | Root rotation quaternion (x,y,z,w) |
| `joint_pos` | 12 | 7–18 | Joint angles `[FR,FL,RR,RL]×3` |
| `toe_pos_local` | 12 | 19–30 | Foot-end local positions |
| `linear_vel` | 3 | 31–33 | Root linear velocity |
| `angular_vel` | 3 | 34–36 | Root angular velocity |
| `joint_vel` | 12 | 37–48 | Joint velocities |
| `toe_vel_local` | 12 | 49–60 | Foot-end local velocities |

> Raw data uses PyBullet order `[FR, FL, RR, RL]`. `AMPLoader.reorder_from_pybullet_to_isaac()` converts to Isaac Gym order `[FL, FR, RL, RR]` at load time.

**AMP discriminator observations**: `AMPLoader` strips `root_pos` and `root_rot`, keeping:

```
joint_pos(12) + toe_pos_local(12) + linear_vel(3) + angular_vel(3) + joint_vel(12) = 42 dims
```

Discriminator input = two consecutive frames: $(s_t, s_{t+1})$ = **84 dims**.

### Retargeting to a Different Robot

When training GO1/GO2 with A1 mocap data, limb proportions must be rescaled. `datasets/retarget_mocap.py` performs FK-based retargeting:

```bash
# Retarget A1 mocap → GO2 (writes to datasets/mocap_go2/)
python datasets/retarget_mocap.py --src a1 --tgt go2

# Retarget A1 mocap → GO1 (writes to datasets/mocap_go1/)
python datasets/retarget_mocap.py --src a1 --tgt go1
```

| Robot | Thigh | Calf | Total limb | Scale vs A1 |
|-------|-------|------|------------|-------------|
| A1    | 0.200 m | 0.200 m | 0.400 m | 1.000× |
| GO1   | 0.213 m | 0.213 m | 0.426 m | 1.065× |
| GO2   | 0.213 m | 0.213 m | 0.426 m | 1.065× |

After retargeting, point the AMP config at the new directory:

```python
# legged_gym/envs/go2/go2_amp_config.py
MOTION_FILES = glob.glob('datasets/mocap_go2/*')
```

### Verification & Visualization

**Static geometry check** (no GPU needed):
```bash
python datasets/verify_mocap.py --dir datasets/mocap_go2 --robot go2 --plot
```
Checks per file: frame dimension, `root_z` range, foot FK height, joint limits, quaternion norm, velocity magnitudes. Saves a figure to `datasets/verify_go2.png`.

**Isaac Gym playback** (replay data without any trained policy):
```bash
python datasets/replay_mocap.py --task go2_amp --speed 0.3
```
Iterates through all motion files, setting robot state frame-by-frame. Use `--speed 0.3` for slow-motion inspection.

---

## Train

```bash
# Train Go1 AMP policy
python legged_gym/scripts/train.py --task=go1_amp --headless

# Go2 AMP policy
python legged_gym/scripts/train.py --task=go2_amp --headless

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

---

## Play & Policy Export

```bash
python legged_gym/scripts/play.py --task=go1_amp

python legged_gym/scripts/play.py --task=go2_amp
```

Loads the latest checkpoint by default. Set `load_run` and `checkpoint` in the config to specify a version.

Record a video:
```bash
python legged_gym/scripts/record_policy.py --task=go1_amp
```

**Exporting for deployment**: `play.py` automatically exports the policy to `deploy/exported_policy/<robot>/` via `torch.jit.script`. The exported `.pt` file runs without Isaac Gym dependencies and can be loaded directly by the deploy scripts.

```bash
# After play.py exits, the exported policy is at:
# deploy/exported_policy/go2/policy.pt  (or policy_<iter>.pt)
```

---

## Sim-to-Sim (MuJoCo Validation)

Validate the policy in MuJoCo before deploying to hardware, testing behavioral consistency across physics engines.

All robot-specific parameters (xml path, policy path, joint names, default angles, gains, etc.) are read from the YAML config:

```bash
cd deploy/

# Go1 keyboard control
python sim2sim2real_keyboard.py --mode sim --config config/go1.yaml

# Go2 keyboard control
python sim2sim2real_keyboard.py --mode sim --config config/go2.yaml

# Override policy file:
python sim2sim2real_keyboard.py --mode sim --config config/go2.yaml --model policy.pt

# Joystick (uses the same --config flag)
python sim2sim2real_joystick.py --mode sim --config config/go2.yaml
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

# Go1 keyboard control on hardware
python sim2sim2real_keyboard.py --mode real --config config/go1.yaml

# Go2 keyboard control on hardware
python sim2sim2real_keyboard.py --mode real --config config/go2.yaml

# Joystick
python sim2sim2real_joystick.py --mode real --config config/go2.yaml
```

**Key parameters** (set per-robot in `deploy/config/<robot>.yaml`):

| Parameter | Description |
|-----------|-------------|
| `kp_walk` / `kd_walk` | PD gains during policy control |
| `default_dof_pos` | Must match training config exactly |
| `action_scale` | Must match training config exactly |
| `xml_path` | MuJoCo scene file for Sim2Sim |
| `policy_path` | Directory containing exported `.pt` policy |

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
