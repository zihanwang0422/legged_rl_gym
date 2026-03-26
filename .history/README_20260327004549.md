# AMP for Hardware — Adversarial Motion Priors 四足机器人运动控制

基于论文 [Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions](https://bit.ly/3hpvbD6) 的实现。  
使用少量 mocap 参考动作（约 4.5 秒），通过对抗式运动先验（AMP）训练自然运动策略，并支持部署到 Unitree Go1 / A1 真实机器人。

---

## 目录

- [安装](#安装)
- [项目结构](#项目结构)
- [训练 Train](#训练-train)
- [回放 Play](#回放-play)
- [Sim2Sim（MuJoCo 验证）](#sim2simmujoco-验证)
- [Sim2Real（真机部署）](#sim2real真机部署)
- [AMP 算法解读](#amp-算法解读)
- [参考数据格式](#参考数据格式)
- [观测与动作空间](#观测与动作空间)
- [常见问题](#常见问题)

---

## 安装

### 1. 创建 Conda 环境

```bash
conda create -n amp_hw python=3.8
conda activate amp_hw
```

### 2. 安装 PyTorch（cuda 11.3）

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    tensorboard==2.8.0 pybullet==3.2.1 opencv-python==4.5.5.64 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 3. 安装 Isaac Gym Preview 3

从 [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) 下载 Preview **3**（Preview 2 不兼容）。

```bash
cd isaacgym/python && pip install -e .
# 验证安装
cd examples && python 1080_balls_of_solitude.py
```

### 4. 安装 rsl_rl（PPO 实现）

```bash
cd rsl_rl && pip install -e .
```

### 5. 安装 legged_gym

```bash
cd ../ && pip install -e .
```

### 6. 设置环境变量

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## 项目结构

```
legged_rl_gym/
├── datasets/
│   ├── mocap_motions/          # 参考动作数据（JSON 格式，供 AMP 判别器使用）
│   │   ├── trot0.txt           # 小跑步态 0
│   │   ├── trot1.txt           # 小跑步态 1
│   │   ├── pace0.txt           # 溜步步态
│   │   ├── leftturn0.txt       # 左转
│   │   └── rightturn0.txt      # 右转
│   └── hopturn/
│       └── hopturn.txt         # 跳跃转弯
│
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── legged_robot.py         # 基础环境（IsaacGym 交互、reset、奖励计算）
│   │   │   └── legged_robot_config.py  # 基础配置类
│   │   ├── go1/
│   │   │   ├── go1_amp_config.py       # Go1 AMP 训练配置
│   │   │   └── go1_amp.py              # Go1 AMP 环境（继承自 LeggedRobot）
│   │   └── a1/
│   │       ├── a1_amp_config.py        # A1 AMP 训练配置
│   │       └── a1_amp.py              # A1 AMP 环境
│   └── scripts/
│       ├── train.py                    # 启动训练
│       ├── play.py                     # 回放已训练策略
│       └── record_policy.py            # 录制策略视频
│
├── rsl_rl/
│   └── rsl_rl/
│       ├── algorithms/
│       │   ├── amp_ppo.py              # AMP-PPO 算法（策略 + 判别器联合优化）
│       │   └── amp_discriminator.py    # 判别器网络（MLP，输出风格奖励）
│       ├── datasets/
│       │   └── motion_loader.py        # AMPLoader：读取并采样参考动作帧
│       └── runners/
│           └── amp_on_policy_runner.py # 训练主循环
│
├── deploy/
│   ├── sim2sim2real_keyboard.py        # 统一 Sim2Sim/Sim2Real 键盘控制脚本
│   ├── sim2sim2real_joystick.py        # 统一 Sim2Sim/Sim2Real 手柄控制脚本
│   ├── config/
│   │   ├── go1.yaml                    # Go1 部署配置
│   │   └── go2.yaml
│   └── exported_policy/                # 导出的 TorchScript 策略文件
│
├── resources/robots/
│   ├── go1/urdf/                       # Go1 URDF 模型
│   └── a1/urdf/                        # A1 URDF 模型
│
└── logs/                               # 训练日志与模型检查点
    └── go1_amp_example/
        └── <date_time>/
            ├── model_<iter>.pt
            └── ...
```

---

## 训练 Train

```bash
# 训练 Go1 AMP 策略（推荐）
python legged_gym/scripts/train.py --task=go1_amp --headless

# 训练 A1 AMP 策略
python legged_gym/scripts/train.py --task=a1_amp --headless
```

**常用参数：**

| 参数 | 说明 |
|------|------|
| `--task` | 任务名称，如 `go1_amp`、`a1_amp` |
| `--headless` | 无渲染模式（显著提速） |
| `--num_envs 4096` | 并行环境数（Go1 默认 4096） |
| `--max_iterations 500000` | 最大训练迭代次数 |
| `--resume` | 从已有检查点恢复训练 |
| `--load_run <run_name>` | 指定要加载的 run（`-1` 表示最新） |
| `--checkpoint <iter>` | 指定检查点编号（`-1` 表示最新） |
| `--sim_device cpu` | 使用 CPU 仿真 |

**训练技巧：**
- 训练开始后按 `V` 键可关闭渲染，大幅提升仿真速度，之后再按 `V` 恢复查看。
- 模型保存路径：`logs/<experiment_name>/<date_time>_<run_name>/model_<iter>.pt`

---

## 回放 Play

```bash
python legged_gym/scripts/play.py --task=go1_amp
```

默认加载最新 run 的最新模型，可通过修改配置文件中的 `load_run` 和 `checkpoint` 指定版本。

录制视频：
```bash
python legged_gym/scripts/record_policy.py --task=go1_amp
```

---

## Sim2Sim（MuJoCo 验证）

在 MuJoCo 中验证策略，确认部署前的行为一致性：

```bash
cd deploy/

# 键盘控制
python sim2sim2real_keyboard.py --mode sim --model exported_policy/go1/policy.pt

# 手柄控制
python sim2sim2real_joystick.py --mode sim --model exported_policy/go1/policy.pt
```

**键盘映射：**

| 按键 | 功能 |
|------|------|
| `I` / `K` 或 小键盘 `8` / `2` | 前进 / 后退（步进 0.1 m/s） |
| `U` / `O` 或 小键盘 `7` / `9` | 左移 / 右移（步进 0.05 m/s） |
| `J` / `L` 或 小键盘 `4` / `6` | 左转 / 右转（步进 0.1 rad/s） |
| `空格` 或 小键盘 `5` | 紧急停止（速度归零） |
| `Q` 或 `ESC` | 退出 |

---

## Sim2Real（真机部署）

> 确认 Sim2Sim 行为正常后再进行真机部署。

**前提条件：**
- 安装 `unitree_legged_sdk` 并配置网络连接。
- 机器人已趴下站稳，处于阻尼控制状态。

```bash
cd deploy/

# 键盘控制真机
python sim2sim2real_keyboard.py --mode real --model exported_policy/go1/policy.pt

# 手柄控制真机
python sim2sim2real_joystick.py --mode real --model exported_policy/go1/policy.pt
```

**控制频率（`UnifiedConfig`）：**

| 参数 | 值 |
|------|-----|
| 仿真/控制时间步长 | 5 ms（200 Hz） |
| 策略推理频率 | 33 Hz |
| 动作缩放 `action_scale` | 0.25 |

**导出策略（从 IsaacGym 到部署格式）：**

在 `play.py` 中，策略通过 `torch.jit.script` 导出为 TorchScript，保存于 `deploy/exported_policy/` 目录。

---

## AMP 算法解读

### 核心思想

AMP（Adversarial Motion Priors）的目标是让策略产生的运动**在风格上接近**参考 mocap 数据，同时完成速度跟踪等任务目标。它用**判别器的输出代替人工设计的运动质量奖励**。

```
总奖励 = α × AMP风格奖励  +  (1-α) × 任务奖励
```

其中 `α` 由 `amp_task_reward_lerp` 控制（默认 0.3，即 70% AMP + 30% 任务）。

### 系统组件

```
┌─────────────────────────────────────────────────────┐
│                    训练循环                          │
│                                                     │
│  IsaacGym 环境  ──→  AMP观测(s_t, s_{t+1})          │
│       ↓                      ↓                      │
│   PPO Actor-Critic      判别器 D(s,s')               │
│       ↓                      ↓                      │
│   任务奖励 r_task    风格奖励 r_AMP = clamp(1-¼(D-1)²)│
│         ↘                  ↙                        │
│           总奖励 → GAE → 策略更新                    │
│                                                     │
│  参考数据回放缓冲区  ──→  判别器监督信号              │
└─────────────────────────────────────────────────────┘
```

### 判别器（`amp_discriminator.py`）

- **网络结构**：MLP，隐藏层 `[1024, 512]`，输出 1 维 logit
- **输入**：拼接相邻两帧的 AMP 观测 $(s_t, s_{t+1})$，维度 = `observation_dim × 2`
- **训练目标**：区分"专家（mocap）"和"策略生成"的状态转移，使用 WGAN 风格的梯度惩罚（`compute_grad_pen`）
- **风格奖励计算**：

$$r_{AMP} = \text{coef} \times \max\left(0,\ 1 - \frac{1}{4}(D(s,s') - 1)^2\right)$$

### 参考状态初始化（RSI）

- 概率 `reference_state_initialization_prob = 0.85`：每次环境 reset 时，有 85% 概率从参考动作中随机采样一帧作为初始状态，而不是固定站姿。
- 目的：覆盖更广的运动相空间，加速学习自然步态。

### 训练超参数（`A1AMPCfgPPO` / `GO1AMPCfgPPO`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `amp_reward_coef` | 2.0 | 风格奖励系数 |
| `amp_task_reward_lerp` | 0.3 | 任务奖励插值比例 |
| `amp_discr_hidden_dims` | [1024, 512] | 判别器隐藏层 |
| `amp_replay_buffer_size` | 1,000,000 | 策略数据回放缓冲区大小 |
| `amp_num_preload_transitions` | 200,000 | 预加载参考帧数 |
| `entropy_coef` | 0.01 | 策略熵正则化 |

---

## 参考数据格式

数据文件位于 `datasets/mocap_motions/`，JSON 格式，扩展名为 `.txt`。

### 文件结构

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.021,
  "EnableCycleOffsetPosition": true,
  "EnableCycleOffsetRotation": true,
  "MotionWeight": 0.5,
  "Frames": [
    [frame_0_values...],
    [frame_1_values...],
    ...
  ]
}
```

- `FrameDuration`：帧间隔（秒），决定动作播放速度。
- `MotionWeight`：多个动作文件混合时的采样权重。
- `LoopMode: "Wrap"`：运动循环播放。

### 每帧数据维度（61 维，PyBullet 顺序，加载时会转换到 Isaac 顺序）

| 字段 | 维度 | 索引 | 说明 |
|------|------|------|------|
| `root_pos` | 3 | 0–2 | 根节点位置 (x, y, z) |
| `root_rot` | 4 | 3–6 | 根节点旋转四元数 (x,y,z,w) |
| `joint_pos` | 12 | 7–18 | 12 个关节角度 [FR,FL,RR,RL]×3 |
| `toe_pos_local` | 12 | 19–30 | 4 个足端局部位置 |
| `linear_vel` | 3 | 31–33 | 根节点线速度 |
| `angular_vel` | 3 | 34–36 | 根节点角速度 |
| `joint_vel` | 12 | 37–48 | 12 个关节速度 |
| `toe_vel_local` | 12 | 49–60 | 4 个足端局部速度 |

> **注意**：原始数据为 PyBullet 关节顺序 `[FR, FL, RR, RL]`，`AMPLoader.reorder_from_pybullet_to_isaac()` 会自动转换为 IsaacGym 顺序 `[FL, FR, RL, RR]`。

### AMP 观测（判别器实际使用）

`AMPLoader` 加载时会去除 `root_pos` 和 `root_rot`，只保留从 `ROOT_ROT_END_IDX` 到 `JOINT_VEL_END_IDX` 的部分：

```
joint_pos(12) + toe_pos_local(12) + linear_vel(3) + angular_vel(3) + joint_vel(12) = 42 维
```

判别器输入为相邻两帧拼接：$(s_t, s_{t+1})$ = **84 维**。

---

## 观测与动作空间

### 策略观测（Go1，45 维）

| 索引 | 维度 | 内容 | 缩放 |
|------|------|------|------|
| 0–2 | 3 | 机身角速度 `base_ang_vel` (x,y,z) | × 0.25 |
| 3–5 | 3 | 重力投影向量 `projected_gravity` | — |
| 6–8 | 3 | 速度指令 `[lin_vel_x, lin_vel_y, ang_vel_yaw]` | × [2.0, 2.0, 0.25] |
| 9–20 | 12 | 关节位置偏差 `dof_pos - default_dof_pos` | × 1.0 |
| 21–32 | 12 | 关节速度 `dof_vel` | × 0.05 |
| 33–44 | 12 | 上一时刻动作 `last_actions` | — |

### 关节顺序（IsaacGym，FL→FR→RL→RR）

| 序号 | 关节名 | 序号 | 关节名 |
|------|--------|------|--------|
| 1 | FL_hip_joint | 7 | RL_hip_joint |
| 2 | FL_thigh_joint | 8 | RL_thigh_joint |
| 3 | FL_calf_joint | 9 | RL_calf_joint |
| 4 | FR_hip_joint | 10 | RR_hip_joint |
| 5 | FR_thigh_joint | 11 | RR_thigh_joint |
| 6 | FR_calf_joint | 12 | RR_calf_joint |

### 动作（12 维）

输出为关节位置目标的残差，实际目标角度：

```
target_angle = action_scale × action + default_dof_pos
```

`action_scale = 0.25`，PD 控制：kp = 80 N·m/rad，kd = 1.0 N·m·s/rad。

---

## 常见问题

**`ImportError: libpython3.8m.so.1.0: cannot open shared object file`**
```bash
sudo apt install libpython3.8
# 或
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

**训练时 GPU 内存不足**  
减小并行环境数：`--num_envs 1024`

**接触力不稳定（GPU + triangle mesh）**  
使用平面地形（`mesh_type = 'plane'`），或改用足端力传感器替代 `net_contact_force_tensor`。

**Sim2Real 机器人抖动**  
检查 `default_dof_pos` 是否与训练配置完全一致；降低 `action_scale` 或增大 kd 阻尼。
 
