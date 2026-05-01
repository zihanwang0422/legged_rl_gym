"""
replay_mocap.py — Replay retargeted AMP mocap data directly in IsaacGym
=======================================================================
Loads the GO2 (or GO1) environment with a single robot and drives its
state directly from mocap frames, without any policy network.

Useful for visually verifying that retargeted motion data looks correct
on the actual robot model before starting training.

Usage (from legged_rl_gym/ root):
  # Replay GO2 retargeted data (default)
  python legged_gym/scripts/replay_mocap.py --task go2_amp

  # Replay original A1 data on A1 robot
  python legged_gym/scripts/replay_mocap.py --task a1_amp

  # Loop a specific file
  python legged_gym/scripts/replay_mocap.py --task go2_amp \
      --motion_file datasets/mocap_go2/trot0.txt

  # Loop at slower speed (0.5x) to inspect details
  python legged_gym/scripts/replay_mocap.py --task go2_amp --speed 0.5
"""

import os
import re
import sys
import glob
import time
import json
import argparse

# isaacgym MUST be imported before numpy/torch
import isaacgym
from isaacgym import gymtorch

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args


# ---------------------------------------------------------------------------
# Load one motion file
# ---------------------------------------------------------------------------

def load_motion(path):
    with open(path) as f:
        text = re.sub(r"//.*", "", f.read())
    d = json.loads(text)
    frames = np.array(d["Frames"], dtype=np.float32)
    dt = float(d["FrameDuration"])
    return frames, dt


# ---------------------------------------------------------------------------
# AMPLoader index helpers (mirror rsl_rl/datasets/motion_loader.py constants)
# Raw frame layout (61-dim, PyBullet order):
#   [0:3]  root_pos  [3:7]  root_rot (xyzw)
#   [7:19] joint_pos (FR FL RR RL × hip thigh calf)
#   [19:31] toe_pos  [31:34] lin_vel  [34:37] ang_vel
#   [37:49] joint_vel  [49:61] toe_vel
# ---------------------------------------------------------------------------

ROOT_POS   = slice(0,  3)
ROOT_ROT   = slice(3,  7)
JOINT_POS  = slice(7,  19)
LIN_VEL    = slice(31, 34)
ANG_VEL    = slice(34, 37)
JOINT_VEL  = slice(37, 49)


def pybullet_to_isaac_joints(arr):
    """Reorder 12 joint values from PyBullet [FR,FL,RR,RL] to IsaacGym [FL,FR,RL,RR]."""
    fr, fl, rr, rl = arr[..., 0:3], arr[..., 3:6], arr[..., 6:9], arr[..., 9:12]
    return np.concatenate([fl, fr, rl, rr], axis=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay retargeted AMP mocap in IsaacGym (no policy needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task",        default="go2_amp",
                        help="Task name (go2_amp / go1_amp / a1_amp)")
    parser.add_argument("--motion_file", default=None,
                        help="Single .txt motion file to replay. "
                             "If not given, cycles through all files from the config.")
    parser.add_argument("--speed",       type=float, default=1.0,
                        help="Replay speed multiplier (default 1.0). "
                             "0.5 = half speed for inspection.")
    parser.add_argument("--num_repeat",  type=int, default=999,
                        help="How many times to loop through the motion (default 999).")

    # pass remaining args to legged_gym get_args so it doesn't complain
    known, _ = parser.parse_known_args()

    # ---- Build legged_gym arg namespace ----
    # We override the important ones; legged_gym's get_args reads sys.argv
    sys.argv = [sys.argv[0],
                f"--task={known.task}",
                "--num_envs=1"]
    args = get_args()

    # ---- Configure env ----
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Minimal env — 1 robot, no randomization
    env_cfg.env.num_envs                          = 1
    env_cfg.env.reference_state_initialization    = False
    env_cfg.terrain.num_rows                      = 1
    env_cfg.terrain.num_cols                      = 1
    env_cfg.terrain.curriculum                    = False
    env_cfg.noise.add_noise                       = False
    env_cfg.domain_rand.randomize_friction        = False
    env_cfg.domain_rand.push_robots               = False
    env_cfg.domain_rand.randomize_gains           = False
    env_cfg.domain_rand.randomize_base_mass       = False

    # Don't need AMP loader during env init for this script
    train_cfg.runner.amp_num_preload_transitions  = 1

    # Camera — follow robot from the side
    env_cfg.viewer.pos    = [1.0, -2.5, 1.0]
    env_cfg.viewer.lookat = [1.0,  0.0, 0.5]

    # ---- Create environment ----
    print(f"\n[replay_mocap] Creating '{args.task}' environment with 1 robot...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    device = env.device

    # ---- Collect motion files ----
    if known.motion_file:
        motion_files = [known.motion_file]
    else:
        # env_cfg.env.amp_motion_files is already a resolved list (from glob.glob in config)
        try:
            motion_files = sorted(env_cfg.env.amp_motion_files)
        except (AttributeError, TypeError):
            motion_files = sorted(glob.glob("datasets/mocap_go2/*.txt"))

    if not motion_files:
        print("ERROR: no motion files found. Use --motion_file to specify one.")
        return

    print(f"\n[replay_mocap] Motion files to replay ({len(motion_files)}):")
    for f in motion_files:
        print(f"  {f}")

    # ---- Env origins ----
    env_origins = env.env_origins[0].cpu().numpy()   # shape (3,)

    # ---- Replay loop ----
    all_envs = torch.zeros(1, dtype=torch.long, device=device)
    all_envs_int32 = all_envs.to(dtype=torch.int32)

    print(f"\n[replay_mocap] Starting replay at {known.speed}x speed. "
          f"Close the Isaac Gym window to exit.\n")

    for repeat in range(known.num_repeat):
        for motion_path in motion_files:
            frames, frame_dt = load_motion(motion_path)
            n_frames = len(frames)
            name = os.path.basename(motion_path)

            print(f"  Playing: {name}  ({n_frames} frames, "
                  f"{(n_frames-1)*frame_dt:.2f}s, "
                  f"loop {repeat+1}/{known.num_repeat})")

            for fi in range(n_frames):
                frame = frames[fi]

                # ---- Root state ----
                root_pos = frame[ROOT_POS].copy()
                root_pos[0] += env_origins[0]
                root_pos[1] += env_origins[1]

                root_rot = frame[ROOT_ROT].copy()   # xyzw
                lin_vel  = frame[LIN_VEL].copy()
                ang_vel  = frame[ANG_VEL].copy()

                # Fill root_states tensor: [pos(3), rot(4), lin_vel(3), ang_vel(3)]
                env.root_states[0, 0:3]  = torch.from_numpy(root_pos).to(device)
                env.root_states[0, 3:7]  = torch.from_numpy(root_rot).to(device)
                env.root_states[0, 7:10] = torch.from_numpy(lin_vel).to(device)
                env.root_states[0, 10:13]= torch.from_numpy(ang_vel).to(device)

                env.gym.set_actor_root_state_tensor_indexed(
                    env.sim,
                    gymtorch.unwrap_tensor(env.root_states),
                    gymtorch.unwrap_tensor(all_envs_int32),
                    1,
                )

                # ---- DOF state: reorder PyBullet → IsaacGym ----
                joint_pos_raw = frame[JOINT_POS]
                joint_vel_raw = frame[JOINT_VEL]

                joint_pos = pybullet_to_isaac_joints(joint_pos_raw[np.newaxis])[0]
                joint_vel = pybullet_to_isaac_joints(joint_vel_raw[np.newaxis])[0]

                env.dof_pos[0]  = torch.from_numpy(joint_pos).to(device)
                env.dof_vel[0]  = torch.from_numpy(joint_vel).to(device)

                env.gym.set_dof_state_tensor_indexed(
                    env.sim,
                    gymtorch.unwrap_tensor(env.dof_state),
                    gymtorch.unwrap_tensor(all_envs_int32),
                    1,
                )

                # ---- Step physics + render ----
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)
                env.gym.refresh_actor_root_state_tensor(env.sim)
                env.gym.refresh_dof_state_tensor(env.sim)

                if not env.headless:
                    env.gym.step_graphics(env.sim)
                    env.gym.draw_viewer(env.viewer, env.sim, True)
                    # Check if window was closed
                    if env.gym.query_viewer_has_closed(env.viewer):
                        print("\n[replay_mocap] Viewer closed — exiting.")
                        return

                # ---- Timing: wait to match playback speed ----
                time.sleep(frame_dt / known.speed)

    print("\n[replay_mocap] Done.")


if __name__ == "__main__":
    main()
