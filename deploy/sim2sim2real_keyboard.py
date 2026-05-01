#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
统一的 Sim2Sim (MuJoCo) 和 Sim2Real (Unitree SDK) 控制脚本

支持两种模式:
1. Sim2Sim: MuJoCo 仿真环境 (--mode sim)
2. Sim2Real: Unitree Go1 真实机器人 (--mode real)

共享配置参数和策略推理逻辑

用法:
  仿真模式: python sim2sim_sim2real_unified.py --mode sim
  真机模式: python sim2sim_sim2real_unified.py --mode real
"""

import time
import math
import numpy as np
import argparse
import torch
import threading
import sys
import termios
import tty
import select
import os
import yaml
from scipy.spatial.transform import Rotation as R

# ========== Config loader ==========

def load_config(config_path):
    """Load deployment configuration from a YAML file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    script_dir = os.path.dirname(os.path.abspath(config_path))
    legged_gym_root = os.path.abspath(os.path.join(script_dir, '..'))

    def resolve(p):
        p = p.replace('{LEGGED_GYM_ROOT_DIR}', legged_gym_root)
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    class Config:
        pass

    c = Config()

    c.sim_dt          = float(cfg['sim_dt'])
    c.policy_hz       = float(cfg['policy_hz'])
    c.policy_dt       = 1.0 / c.policy_hz

    c.kp_stand        = float(cfg['kp_stand'])
    c.kd_stand        = float(cfg['kd_stand'])
    c.kp_walk         = float(cfg['kp_walk'])
    c.kd_walk         = float(cfg['kd_walk'])

    c.default_dof_pos = np.array(cfg['default_dof_pos'], dtype=np.float32)

    scales = cfg['obs_scales']
    c.obs_scales = {
        'lin_vel':  float(scales['lin_vel']),
        'ang_vel':  float(scales['ang_vel']),
        'dof_pos':  float(scales['dof_pos']),
        'dof_vel':  float(scales['dof_vel']),
        'commands': np.array(scales['commands'], dtype=np.float32),
    }

    c.action_scale       = float(cfg['action_scale'])
    c.clip_observations  = float(cfg.get('clip_observations', 100.0))
    c.clip_actions       = float(cfg.get('clip_actions', 100.0))

    c.standup_duration   = float(cfg['standup_duration'])
    c.stabilize_duration = float(cfg['stabilize_duration'])

    c.vx_range   = tuple(cfg['vx_range'])
    c.vy_range   = tuple(cfg['vy_range'])
    c.vyaw_range = tuple(cfg['vyaw_range'])

    c.robot_ip   = cfg.get('robot_ip', '192.168.123.10')
    c.robot_port = cfg.get('robot_port', 8007)
    c.local_port = cfg.get('local_port', 8080)

    c.train_to_sdk_map = cfg['train_to_sdk_map']
    c.sdk_to_train_map = cfg['sdk_to_train_map']
    c.leg_order        = cfg['leg_order']
    c.joint_suffixes   = cfg['joint_suffixes']

    c.joint_limit_low  = np.array(cfg['joint_limit_low'],  dtype=np.float32)
    c.joint_limit_high = np.array(cfg['joint_limit_high'], dtype=np.float32)
    # SDK limits default to same as training limits (mapping handles reorder)
    c.joint_limit_low_sdk  = np.array(
        cfg.get('joint_limit_low_sdk',  cfg['joint_limit_low']),  dtype=np.float32)
    c.joint_limit_high_sdk = np.array(
        cfg.get('joint_limit_high_sdk', cfg['joint_limit_high']), dtype=np.float32)

    c.xml_path    = resolve(cfg['xml_path'])
    c.policy_path = resolve(cfg['policy_path'])

    return c


# ========== 辅助函数 ==========

def quat_from_euler_xyz(roll, pitch, yaw):
    """从欧拉角计算四元数 [x, y, z, w]"""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=np.float32)

def quat_rotate_inverse(q, v):
    """将向量从世界坐标系旋转到机体坐标系"""
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def quat_to_euler_xyz(q):
    """将四元数 [x, y, z, w] 转换为欧拉角 [roll, pitch, yaw]"""
    x, y, z, w = q
    
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw], dtype=np.float32)

def compute_projected_gravity(quat):
    """计算投影重力向量"""
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity = quat_rotate_inverse(quat, gravity_world)
    return projected_gravity

def build_obs_45(base_ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_action, config):
    """
    构建 45 维观测向量:
    1-3: base_ang_vel [wx, wy, wz] * ang_vel_scale
    4-6: projected_gravity [gx, gy, gz]
    7-9: commands [lin_vel_x, lin_vel_y, ang_vel_yaw] * commands_scale
    10-21: (dof_pos - default_dof_pos) * dof_pos_scale
    22-33: dof_vel * dof_vel_scale
    34-45: last_actions
    """
    obs = []
    
    # 1-3: Base angular velocity (scaled)
    obs.extend(list(base_ang_vel * config.obs_scales['ang_vel']))
    
    # 4-6: Projected gravity
    obs.extend(list(projected_gravity))
    
    # 7-9: Commands (scaled)
    commands_scaled = commands * config.obs_scales['commands']
    obs.extend(list(commands_scaled))
    
    # 10-21: dof_pos - default_dof_pos (scaled)
    pos_delta = (dof_pos - config.default_dof_pos) * config.obs_scales['dof_pos']
    obs.extend(list(pos_delta))
    
    # 22-33: dof_vel (scaled)
    obs.extend(list(dof_vel * config.obs_scales['dof_vel']))
    
    # 34-45: Last action
    obs.extend(list(last_action))
    
    return np.array(obs, dtype=np.float32)

def normalize_obs(obs, clip_value=100.0):
    """裁剪观测值"""
    return np.clip(obs, -clip_value, clip_value)


# ========== 键盘控制器 ==========

class KeyboardController:
    """线程安全的键盘控制器"""
    def __init__(self, vx_range=(-1.0, 2.0), vy_range=(-0.3, 0.3), vyaw_range=(-1.57, 1.57)):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range
        self.lock = threading.Lock()
        self.running = True
        self.exit_requested = False
        self.thread = None
        
        self.vx_step = 0.1
        self.vy_step = 0.05
        self.vyaw_step = 0.1
    
    def get_velocity(self):
        with self.lock:
            return self.vx, self.vy, self.vyaw
    
    def set_velocity(self, vx, vy, vyaw):
        with self.lock:
            self.vx = np.clip(vx, self.vx_range[0], self.vx_range[1])
            self.vy = np.clip(vy, self.vy_range[0], self.vy_range[1])
            self.vyaw = np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1])
    
    def keyboard_thread(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    vx, vy, vyaw = self.get_velocity()
                    
                    if key in ['i', 'I', '8']:
                        vx += self.vx_step
                    elif key in ['k', 'K', '2']:
                        vx -= self.vx_step
                    elif key in ['u', 'U', '7']:
                        vy += self.vy_step
                    elif key in ['o', 'O', '9']:
                        vy -= self.vy_step
                    elif key in ['j', 'J', '4']:
                        vyaw += self.vyaw_step
                    elif key in ['l', 'L', '6']:
                        vyaw -= self.vyaw_step
                    elif key in [' ', '5']:
                        vx, vy, vyaw = 0.0, 0.0, 0.0
                    elif key in ['q', 'Q', '\x1b']:
                        self.exit_requested = True
                        self.running = False
                        print("\n[Keyboard] Exit requested...")
                        break
                    else:
                        continue
                    
                    self.set_velocity(vx, vy, vyaw)
                    vx, vy, vyaw = self.get_velocity()
                    print(f"\r[Command] vx={vx:+.2f} m/s, vy={vy:+.2f} m/s, yaw={vyaw:+.2f} rad/s", end='', flush=True)
        
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def start(self):
        self.thread = threading.Thread(target=self.keyboard_thread, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


# ========== Sim2Sim (MuJoCo) 控制器 ==========

class Sim2SimController:
    """MuJoCo 仿真控制器"""
    
    def __init__(self, config, xml_path, policy_path, headless=False):
        self.config = config
        self.headless = headless

        # 从 config 动态生成关节和执行器名称
        self.joint_names = []
        self.actuator_names = []
        for leg in config.leg_order:
            for suffix in config.joint_suffixes:
                self.joint_names.append(f'{leg}_{suffix}_joint')
                self.actuator_names.append(f'{leg}_{suffix}')
        
        # 加载 MuJoCo 模型
        import mujoco
        import mujoco.viewer
        self.mujoco = mujoco
        self.mujoco_viewer = mujoco.viewer
        
        print(f"Loading MuJoCo model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.sim_dt
        
        # 获取关节和执行器索引
        self.joint_qpos_addrs = []
        self.joint_dof_addrs = []
        self.actuator_ids = []
        
        for joint_name, actuator_name in zip(self.joint_names, self.actuator_names):
            joint_id = self.model.joint(joint_name).id
            qpos_addr = self.model.jnt_qposadr[joint_id]
            dof_addr = self.model.jnt_dofadr[joint_id]
            actuator_id = self.model.actuator(actuator_name).id
            
            self.joint_qpos_addrs.append(qpos_addr)
            self.joint_dof_addrs.append(dof_addr)
            self.actuator_ids.append(actuator_id)
        
        # 加载策略
        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location='cpu')
        self.policy.eval()
        
        # 初始化状态
        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes = np.zeros(12, dtype=np.float32)
        
        # 策略频率控制
        self.policy_decimation = int(config.policy_dt / config.sim_dt)
        self.policy_counter = 0
        
        # 初始化机器人位置
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[qpos_addr] = config.default_dof_pos[i]
        self.data.qpos[2] = 0.27  # 初始高度
        mujoco.mj_forward(self.model, self.data)
        
        # 检测 actuator 类型：biastype==1 表示位置伺服（GO1），否则为纯力矩电机（GO2）
        first_id = self.actuator_ids[0]
        self._is_position_servo = (int(self.model.actuator_biastype[first_id]) == 1)
        
        print(f"Sim2Sim controller initialized (actuator: {'position_servo' if self._is_position_servo else 'torque'})")
    
    def get_state(self):
        """获取当前状态"""
        # Base angular velocity
        base_ang_vel = self.data.qvel[3:6].copy()
        
        # Projected gravity
        quat_wxyz = self.data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        projected_gravity = compute_projected_gravity(quat_xyzw)
        
        # Joint positions and velocities
        dof_pos = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs], dtype=np.float32)
        dof_vel = np.array([self.data.qvel[addr] for addr in self.joint_dof_addrs], dtype=np.float32)
        
        return base_ang_vel, projected_gravity, dof_pos, dof_vel
    
    def send_command(self, target_pos, kp=0.0, kd=0.0):
        """发送控制命令

        方案 A（当前启用）：位置伺服型 XML（scene_pos.xml，GO2/GO1 均适用）
          - XML 内部已设 kp=25 kv=0.5，Python 直接写入目标关节角即可
          - 适合快速切换，行为与 IsaacGym 训练完全一致

        方案 B（注释掉）：纯力矩电机 XML（scene.xml 原版 GO2）
          - Python 端计算 PD 力矩：torque = kp*(target-q) - kd*dq
          - 需要在 go2.yaml 中把 xml_path 改回 scene.xml 并取消下方注释
        """
        if self._is_position_servo:
            # ── 方案 A：位置伺服，直接发目标角 ──────────────────────────────
            for i, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = target_pos[i]
        else:
            # ── 方案 B：力矩电机，Python 端 PD 计算 ──────────────────────────
            for i, (actuator_id, qpos_addr, dof_addr) in enumerate(
                zip(self.actuator_ids, self.joint_qpos_addrs, self.joint_dof_addrs)
            ):
                q = self.data.qpos[qpos_addr]
                dq = self.data.qvel[dof_addr]
                torque = kp * (target_pos[i] - q) - kd * dq
                self.data.ctrl[actuator_id] = torque
    
    def step(self):
        """执行一步仿真"""
        self.mujoco.mj_step(self.model, self.data)
    
    def run(self, gamepad):
        """
        Main simulation loop with Absolute Time Sync and Render Decimation.
        """
        motiontime = 0 # simulation step counter 
        
        # --- Time Synchronization Setup ---
        sim_dt = self.model.opt.timestep  # Physics timestep (e.g., 0.005s)
        target_render_fps = 50            # Human eye only needs 30-60 FPS
        # Calculate how many physics steps to skip between renders
        render_skip = int(1.0 / (target_render_fps * sim_dt))
        if render_skip < 1: render_skip = 1
        
        # Launch viewer
        with self.mujoco_viewer.launch_passive(self.model, self.data) as viewer:
            # Initial Camera setup
            viewer.cam.lookat[:] = self.data.qpos[:3]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            
            # --- Establish Absolute Time Reference ---
            # Record start time immediately before entering the loop
            start_time = time.time() 
            
            while viewer.is_running():
                if gamepad.exit_requested:
                    print("\nExit request detected, ending starget_posimulation...")
                    break
                
                # Get MuJoCo internal simulation time
                sim_time = self.data.time
                
                # --- Phase 1: Stand up (Linear Interpolation) ---
                if sim_time <= self.config.standup_duration:
                    rate = min(sim_time / self.config.standup_duration, 1.0)
                    for i, qpos_addr in enumerate(self.joint_qpos_addrs):
                        current_q = self.data.qpos[qpos_addr]
                        self.qDes[i] = current_q * (1 - rate) + self.config.default_dof_pos[i] * rate
                    self.send_command(self.qDes, self.config.kp_stand, self.config.kd_stand)
                
                # --- Phase 2: Stabilize at Default Pose ---
                elif sim_time <= self.config.standup_duration + self.config.stabilize_duration:
                    self.qDes = self.config.default_dof_pos.copy()
                    self.send_command(self.qDes, self.config.kp_stand, self.config.kd_stand)
                
                # --- Phase 3: Neural Network Policy Control ---
                else:
                    # Safety check: Detect if robot is falling
                    quat_wxyz = self.data.qpos[3:7].copy()
                    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
                    rpy = quat_to_euler_xyz(quat_xyzw)
                    if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                        print(f"\nWarning at {sim_time:.2f}s: Robot tilted! roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}")
                    
                    # Policy inference (decimated frequency)
                    self.policy_counter += 1
                    if self.policy_counter >= self.policy_decimation:
                        self.policy_counter = 0
                        
                        # Get user input from keyboard
                        cmd_vx, cmd_vy, cmd_vyaw = keyboard.get_velocity()
                        commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                        
                        # Extract robot state
                        base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()
                        
                        # Prepare observation for the policy
                        obs = build_obs_45(base_ang_vel, projected_gravity, commands,
                                           dof_pos, dof_vel, self.last_action, self.config)
                        obs = normalize_obs(obs, self.config.clip_observations)
                        obs_batch = obs[np.newaxis, :].astype(np.float32)
                        
                        # Forward pass through the neural network
                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs_batch)
                            action_tensor = self.policy(obs_tensor)
                            if isinstance(action_tensor, tuple):
                                action_tensor = action_tensor[0]
                            action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                        
                        # Scale action to joint targets
                        action = np.clip(action, -self.config.clip_actions, self.config.clip_actions)
                        self.last_action = action[:12].copy()
                        self.qDes = action[:12] * self.config.action_scale + self.config.default_dof_pos
                        self.qDes = np.clip(self.qDes, self.config.joint_limit_low, self.config.joint_limit_high)
        
                    self.send_command(self.qDes, self.config.kp_walk, self.config.kd_walk)
                
                # --- Physics Step ---
                self.step()
                motiontime += 1 # 注意 这个要放在整个while循环的最后面，不能只放在policy控制的部分，因为standup和stabilize阶段也需要计数
                
                # --- Visual Update (Render Decimation) ---
                # Syncing every step is slow; sync at 50Hz for better performance
                if motiontime % render_skip == 0:
                    viewer.cam.lookat[:] = self.data.qpos[:3]
                    viewer.sync()
                
                # --- Soft Real-time Synchronization (Absolute) ---
                # Calculate the exact time we SHOULD be at
                expected_real_time = start_time + (motiontime * sim_dt)
                time_to_sleep = expected_real_time - time.time()
                
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                
                # --- Status Telemetry (Original Format) ---
                if motiontime % int(1.0 / self.config.sim_dt) == 0:
                    real_time_now = time.time() - start_time
                    actual_hz = motiontime / real_time_now if real_time_now > 0 else 0
                    
                    vx_cur, vy_cur, vyaw_cur = keyboard.get_velocity()
                    
                    print(f"[Keyboard] vx={vx_cur:+.2f} m/s | vy={vy_cur:+.2f} | yaw={vyaw_cur:+.2f} rad/s")
                    print(f"[Sim Time]: t={self.data.time:.1f}s, Base height: {self.data.qpos[2]:.3f}m")
                    print(f"[Real Time]: t={real_time_now:.1f}s, Actual Hz: {actual_hz:.2f} Hz")



# ========== Sim2Real (Unitree SDK) 控制器 ==========

class Sim2RealController:
    """Unitree Go1 真机控制器"""
    
    def __init__(self, config, policy_path):
        self.config = config
        
        # 导入 SDK
        SDK_PATH = os.path.join(os.path.dirname(__file__),
                                '../../unitree_legged_sdk/lib/python/amd64')
        sys.path.append(SDK_PATH)
        import robot_interface as sdk
        self.sdk = sdk
        
        # SDK 关节索引
        self.d = {
            'FR_0':0, 'FR_1':1, 'FR_2':2,
            'FL_0':3, 'FL_1':4, 'FL_2':5,
            'RR_0':6, 'RR_1':7, 'RR_2':8,
            'RL_0':9, 'RL_1':10,'RL_2':11
        }
        self.joint_order = ['FR_0','FR_1','FR_2',
                           'FL_0','FL_1','FL_2',
                           'RR_0','RR_1','RR_2',
                           'RL_0','RL_1','RL_2']
        
        # 初始化 UDP
        LOWLEVEL = 0xff
        self.udp = sdk.UDP(LOWLEVEL, config.local_port, 
                          config.robot_ip, config.robot_port)
        self.low_cmd = sdk.LowCmd()
        self.low_state = sdk.LowState()
        self.udp.InitCmdData(self.low_cmd)
        
        print(f"UDP initialized: {config.robot_ip}:{config.robot_port}")
        
        # 加载策略
        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location='cpu')
        self.policy.eval()
        
        # 初始化状态
        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes_train = np.zeros(12, dtype=np.float32)
        
        # 策略频率控制
        self.policy_decimation = int(config.policy_dt / config.sim_dt)
        self.policy_counter = 0
        
        print("Sim2Real controller initialized")
    
    def wait_for_connection(self):
        """等待机器人连接"""
        print("Waiting for robot connection...")
        
        # 初始化命令（阻尼模式，Kp=0）
        for i in range(12):
            self.low_cmd.motorCmd[i].q = 0.0
            self.low_cmd.motorCmd[i].dq = 0.0
            self.low_cmd.motorCmd[i].Kp = 0.0
            self.low_cmd.motorCmd[i].Kd = 3.0
            self.low_cmd.motorCmd[i].tau = 0.0
        
        # 先发送命令，激活通信
        for i in range(100):
            self.udp.Recv()
            self.udp.GetRecv(self.low_state)
            self.udp.SetSend(self.low_cmd)
            self.udp.Send()
            time.sleep(self.config.sim_dt)
        
        # 检查是否收到有效数据
        q_sum = sum(abs(self.low_state.motorState[i].q) for i in range(12))
        if q_sum < 0.01:
            print("Error: No valid joint data received!")
            print("Please check:")
            print("  1. Robot is powered on")
            print("  2. Network connection is working")
            print("  3. IP address is correct (current: {})".format(self.config.robot_ip))
            return False
        
        print("Robot connected successfully!")
        self._print_state()
        return True
    
    def _print_state(self):
        """打印机器人状态"""
        print("\nCurrent joint angles (SDK order: FR, FL, RR, RL):")
        for leg in ['FR', 'FL', 'RR', 'RL']:
            hip = self.low_state.motorState[self.d[f'{leg}_0']].q
            thigh = self.low_state.motorState[self.d[f'{leg}_1']].q
            calf = self.low_state.motorState[self.d[f'{leg}_2']].q
            print(f"  {leg}: hip={hip:+.3f}, thigh={thigh:+.3f}, calf={calf:+.3f}")
        
        rpy = self.low_state.imu.rpy
        print(f"IMU: roll={rpy[0]:+.3f}, pitch={rpy[1]:+.3f}, yaw={rpy[2]:+.3f}")
    
    def get_state(self):
        """获取当前状态"""
        # Update from robot
        self.udp.Recv()
        self.udp.GetRecv(self.low_state)
        
        # Base angular velocity (SDK format)
        base_ang_vel = np.array([
            self.low_state.imu.gyroscope[0],
            self.low_state.imu.gyroscope[1],
            self.low_state.imu.gyroscope[2]
        ], dtype=np.float32)
        
        # Projected gravity from IMU
        rpy = np.array(self.low_state.imu.rpy, dtype=np.float32)
        quat = quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])
        projected_gravity = compute_projected_gravity(quat)
        
        # Joint positions and velocities (SDK -> training order)
        q_sdk = np.array([self.low_state.motorState[i].q for i in range(12)], dtype=np.float32)
        dq_sdk = np.array([self.low_state.motorState[i].dq for i in range(12)], dtype=np.float32)
        
        dof_pos = q_sdk[self.config.sdk_to_train_map]
        dof_vel = dq_sdk[self.config.sdk_to_train_map]
        
        return base_ang_vel, projected_gravity, dof_pos, dof_vel
    
    def send_command(self, target_train, kp, kd):
        """发送控制命令 (训练顺序 -> SDK 顺序)"""
        target_sdk = target_train[self.config.train_to_sdk_map]
        
        for i, jname in enumerate(self.joint_order):
            self.low_cmd.motorCmd[self.d[jname]].q = float(target_sdk[i])
            self.low_cmd.motorCmd[self.d[jname]].dq = 0.0
            self.low_cmd.motorCmd[self.d[jname]].Kp = float(kp)
            self.low_cmd.motorCmd[self.d[jname]].Kd = float(kd)
            self.low_cmd.motorCmd[self.d[jname]].tau = 0.0
        
        self.udp.SetSend(self.low_cmd)
        self.udp.Send()
    
    def send_command_sdk(self, target_sdk, kp, kd):
        """发送控制命令 (直接使用 SDK 顺序)"""
        for i, jname in enumerate(self.joint_order):
            self.low_cmd.motorCmd[self.d[jname]].q = float(target_sdk[i])
            self.low_cmd.motorCmd[self.d[jname]].dq = 0.0
            self.low_cmd.motorCmd[self.d[jname]].Kp = float(kp)
            self.low_cmd.motorCmd[self.d[jname]].Kd = float(kd)
            self.low_cmd.motorCmd[self.d[jname]].tau = 0.0
        
        self.udp.SetSend(self.low_cmd)
        self.udp.Send()
    
    def run(self, keyboard):
        """运行真机控制循环"""
        if not self.wait_for_connection():
            print("Failed to connect to robot!")
            return
        
        print("\n" + "="*70)
        print("Starting real robot control...")
        print("⚠️  CAUTION: Robot will start moving after standup phase!")
        print("    Press Q or ESC to emergency stop")
        print("="*70 + "\n")
        
        motiontime = 0
        
        while True:
            time.sleep(self.config.sim_dt)
            motiontime += 1
            sim_time = motiontime * self.config.sim_dt
            
            # ⚠️ 关键：先接收状态（和 example_position.py 一样）
            self.udp.Recv()
            self.udp.GetRecv(self.low_state)
            
            if keyboard.exit_requested:
                print("\nEmergency stop requested!")
                # Send damping command
                for i in range(12):
                    self.low_cmd.motorCmd[i].q = 0.0
                    self.low_cmd.motorCmd[i].dq = 0.0
                    self.low_cmd.motorCmd[i].Kp = 0.0
                    self.low_cmd.motorCmd[i].Kd = 6.0
                    self.low_cmd.motorCmd[i].tau = 0.0
                self.udp.SetSend(self.low_cmd)
                self.udp.Send()
                break
            
            self._control_step(sim_time, keyboard)
            
            # 每秒打印一次状态
            if motiontime % int(1.0 / self.config.sim_dt) == 0:
                print(f"Time: {sim_time:.1f}s")
    
    def _control_step(self, sim_time, keyboard):
        """单步控制逻辑"""
        # Get current state (注意：状态已经在 run() 中通过 Recv/GetRecv 更新了)
        base_ang_vel_current, projected_gravity_current, dof_pos, dof_vel_current = self.get_state()
        
        # 每秒打印一次当前关节状态
        if int(sim_time * 1000) % 1000 < self.config.sim_dt * 1000:
            print(f"\n=== Current Joint State (Training Order: FL, FR, RL, RR) at {sim_time:.2f}s ===")
            print(f"Position: {dof_pos}")
            print(f"Velocity: {dof_vel_current}")
        
        # Phase 1: Stand up
        if sim_time <= self.config.standup_duration:
            rate = min(sim_time / self.config.standup_duration, 1.0)
            self.qDes_train = dof_pos * (1 - rate) + self.config.default_dof_pos * rate
            self.send_command(self.qDes_train, self.config.kp_stand, self.config.kd_stand)
            
            # 每秒打印一次
            if int(sim_time * 1000) % 1000 < self.config.sim_dt * 1000:
                print(f"Phase 1 (Stand up): rate={rate:.2f}, Kp={self.config.kp_stand}, Kd={self.config.kd_stand}")
        
        # Phase 2: Stabilize
        elif sim_time <= self.config.standup_duration + self.config.stabilize_duration:
            self.qDes_train = self.config.default_dof_pos.copy()
            self.send_command(self.qDes_train, self.config.kp_walk, self.config.kd_walk)
            
            # 每秒打印一次
            if int(sim_time * 1000) % 1000 < self.config.sim_dt * 1000:
                print(f"Phase 2 (Stabilize): Kp={self.config.kp_walk}, Kd={self.config.kd_walk}")
        
        # Phase 3: Policy control
        else:
            # Check tilt
            rpy = np.array(self.low_state.imu.rpy, dtype=np.float32)
            if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                print(f"\nWARNING at {sim_time:.2f}s: Robot tilted! roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}")
            
            # Policy inference
            self.policy_counter += 1
            if self.policy_counter >= self.policy_decimation:
                self.policy_counter = 0
                
                # Get commands
                cmd_vx, cmd_vy, cmd_vyaw = keyboard.get_velocity()
                
                # 🔧 调试：如果速度命令全为 0，给一个小的前进速度测试
                if cmd_vx == 0.0 and cmd_vy == 0.0 and cmd_vyaw == 0.0:
                    # 在 Phase 3 刚开始时（3-5秒）自动给一个测试速度
                    if sim_time < 15.0:
                        cmd_vx = 0.4  # 0.5 m/s 前进
                        if int(sim_time * 1000) % 1000 < self.config.sim_dt * 1000:
                            print(f"🤖 Auto-testing with vx=0.5 m/s (press I to control manually)")
                
                commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                
                # Get state
                base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()
                
                # Build observation
                obs = build_obs_45(base_ang_vel, projected_gravity, commands,
                                 dof_pos, dof_vel, self.last_action, self.config)
                obs = normalize_obs(obs, self.config.clip_observations)
                obs_batch = obs[np.newaxis, :].astype(np.float32)
                
                # Policy inference
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs_batch)
                    action_tensor = self.policy(obs_tensor)
                    if isinstance(action_tensor, tuple):
                        action_tensor = action_tensor[0]
                    action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                
                # Scale action
                action = np.clip(action, -self.config.clip_actions, self.config.clip_actions)
                self.last_action = action[:12].copy()
                
                self.qDes_train = action[:12] * self.config.action_scale + self.config.default_dof_pos
                self.qDes_train = np.clip(self.qDes_train, self.config.joint_limit_low, self.config.joint_limit_high)
                
                # 打印策略输出（每秒一次）
                if int(sim_time * 1000) % 1000 < self.config.sim_dt * 1000:
                    print(f"\n=== Phase 3 (Policy Control) at {sim_time:.2f}s ===")
                    print(f"Commands: vx={commands[0]:.2f}, vy={commands[1]:.2f}, yaw={commands[2]:.2f}")
                    print(f"Policy raw action (train order): {self.last_action}")
                    print(f"Action * scale: {self.last_action * self.config.action_scale}")
                    print(f"default_dof_pos: {self.config.default_dof_pos}")
                    print(f"Scaled qDes (train order): {self.qDes_train}")
                    print(f"Current pos (train order): {dof_pos}")
                    print(f"Target - Current (train order): {self.qDes_train - dof_pos}")
                    print(f"Max absolute diff: {np.abs(self.qDes_train - dof_pos).max():.4f}")
                    
                    # 分析观测值
                    print(f"\n📊 Observation Analysis:")
                    print(f"  Base ang vel: {base_ang_vel}")
                    print(f"  Projected gravity: {projected_gravity}")
                    print(f"  Commands (scaled): {commands * self.config.obs_scales['commands']}")
                    print(f"  dof_pos delta: {(dof_pos - self.config.default_dof_pos)[:3]} ... (first 3)")
                    print(f"  dof_vel (scaled): {(dof_vel * self.config.obs_scales['dof_vel'])[:3]} ... (first 3)")
            
            # 转换为 SDK 顺序并应用 SDK 限位
            qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
            qDes_sdk = np.clip(qDes_sdk, self.config.joint_limit_low_sdk, self.config.joint_limit_high_sdk)
            
            # 打印 SDK 命令（每秒一次）
            if int(sim_time * 1000) % 1000 < self.config.sim_dt * 1000:
                print(f"Sending qDes (SDK order FR,FL,RR,RL): {qDes_sdk}")
                print(f"Kp={self.config.kp_walk}, Kd={self.config.kd_walk}")
            
            self.send_command_sdk(qDes_sdk, self.config.kp_walk, self.config.kd_walk)


# ========== 主函数 ==========

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Unified Sim2Sim/Sim2Real Controller')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'],
                       help='Control mode: sim (MuJoCo) or real (Unitree SDK)')
    parser.add_argument('--model', type=str, default=None,
                       help='Override policy filename inside config policy_path directory (e.g. policy.pt)')
    parser.add_argument('--config', type=str, default='config/go1.yaml',
                       help='YAML config file relative to this script (e.g. config/go1.yaml, config/go2.yaml)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization (sim mode only)')
    args = parser.parse_args()

    # Resolve config path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    config = load_config(config_path)
    print(f"Loaded config: {config_path}")

    # 创建键盘控制器
    keyboard = KeyboardController(
        vx_range=config.vx_range,
        vy_range=config.vy_range,
        vyaw_range=config.vyaw_range
    )
    keyboard.start()

    print("\n" + "="*70)
    print("Keyboard Control Commands")
    print("="*70)
    print("  Forward/Backward: I/K or Numpad 8/2  (step: 0.1 m/s)")
    print("  Strafe Left/Right: U/O or Numpad 7/9  (step: 0.05 m/s)")
    print("  Turn Left/Right: J/L or Numpad 4/6  (step: 0.1 rad/s)")
    print("  Emergency Stop: Space or Numpad 5")
    print("  Exit: Q or ESC")
    print("="*70 + "\n")

    # 根据模式创建控制器
    if args.mode == 'sim':
        print("Mode: Sim2Sim (MuJoCo)")
        xml_path    = config.xml_path
        if args.model:
            policy_path = os.path.join(os.path.dirname(config.policy_path), args.model)
        else:
            policy_path = config.policy_path
        controller  = Sim2SimController(config, xml_path, policy_path, args.headless)
        controller.run(keyboard)

    else:  # real
        print("Mode: Sim2Real (Unitree SDK)")
        if args.model:
            policy_path = os.path.join(os.path.dirname(config.policy_path), args.model)
        else:
            policy_path = config.policy_path
        controller  = Sim2RealController(config, policy_path)
        controller.run(keyboard)

    # 停止键盘控制器
    keyboard.stop()
    print("\nProgram ended.")