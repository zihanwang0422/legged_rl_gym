#!/usr/bin/env python3
"""
retarget_mocap.py — AMP Motion Retargeting for Legged Robots
=============================================================
Retargets mocap motion files from a source robot (default: A1) to a target
robot (GO1 / GO2) by:

  1. Scaling root z (height) by the total limb-length ratio
  2. Keeping all joint angles unchanged  (dimensionless, gait-invariant)
  3. Recomputing toe positions via forward kinematics with target dimensions
  4. Scaling linear velocity by the same limb-length ratio
  5. Keeping angular velocity and joint velocities unchanged

Frame format (raw .txt file, 61-dim, PyBullet recording convention):
  [0:3]   root_pos   (x, y, z)  — z is body height above ground
  [3:7]   root_rot   quaternion (xyzw)
  [7:19]  joint_pos  12 joints — PyBullet order: FR FL RR RL × [hip thigh calf]
  [19:31] toe_pos    4 feet × xyz in body frame — same PyBullet leg order
  [31:34] lin_vel    (vx, vy, vz) in body frame
  [34:37] ang_vel    (wx, wy, wz)
  [37:49] joint_vel  12 values (same order as joint_pos)
  [49:61] toe_vel    12 values (same order as toe_pos, in body frame)

Usage:
  # Retarget A1 mocap → GO2
  python datasets/retarget_mocap.py --src a1 --tgt go2

  # Retarget A1 mocap → GO1, with FK verification output
  python datasets/retarget_mocap.py --src a1 --tgt go1 --verify

  # Custom input/output directories
  python datasets/retarget_mocap.py --tgt go2 \\
      --input datasets/mocap_motions --output datasets/mocap_go2

After retargeting, point the AMP config to the new files:
  MOTION_FILES = glob.glob('datasets/mocap_go2/*')

URDF-extracted dimensions (run `python retarget_mocap.py --print-params` to review):
  Robot    thigh  calf  hip_body_y  hip_link_y
  a1       0.200  0.200   0.047      0.08505
  go1      0.213  0.213   0.04675    0.080
  go2      0.213  0.213   0.0465     0.0955
"""

import os
import re
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Robot parameters (extracted from URDF — see header for values)
# Leg order: FR=0, FL=1, RR=2, RL=3  (PyBullet convention for raw files)
# ---------------------------------------------------------------------------
ROBOT_PARAMS = {
    "a1": dict(
        thigh_len=0.200,
        calf_len=0.200,
        # hip joint position in body frame — [x, y, 0.0], one row per leg
        hip_joint_xyz=np.array([
            [ 0.183, -0.047, 0.0],   # FR
            [ 0.183,  0.047, 0.0],   # FL
            [-0.183, -0.047, 0.0],   # RR
            [-0.183,  0.047, 0.0],   # RL
        ]),
        # y-offset from hip joint to thigh joint (hip-child frame, signed)
        thigh_joint_y=np.array([-0.08505, 0.08505, -0.08505, 0.08505]),
    ),
    "go1": dict(
        thigh_len=0.213,
        calf_len=0.213,
        hip_joint_xyz=np.array([
            [ 0.1881, -0.04675, 0.0],
            [ 0.1881,  0.04675, 0.0],
            [-0.1881, -0.04675, 0.0],
            [-0.1881,  0.04675, 0.0],
        ]),
        thigh_joint_y=np.array([-0.08, 0.08, -0.08, 0.08]),
    ),
    "go2": dict(
        thigh_len=0.213,
        calf_len=0.213,
        hip_joint_xyz=np.array([
            [ 0.1934, -0.0465, 0.0],
            [ 0.1934,  0.0465, 0.0],
            [-0.1934, -0.0465, 0.0],
            [-0.1934,  0.0465, 0.0],
        ]),
        thigh_joint_y=np.array([-0.0955, 0.0955, -0.0955, 0.0955]),
    ),
}

LEG_NAMES = ["FR", "FL", "RR", "RL"]  # PyBullet raw file order


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------

def _rot_x(theta):
    """Rotation matrix about X axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def _rot_y(theta):
    """Rotation matrix about Y axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def leg_fk(theta_hip, theta_thigh, theta_calf,
           hip_xyz, thigh_y, thigh_len, calf_len):
    """
    Returns foot position in the ROOT (body) frame.

    Kinematic chain (standard Unitree convention):
      body origin
       → Trans(hip_xyz)              hip joint at hip_xyz in body
       → Rx(theta_hip)               hip abduction (rotation about X)
       → Trans(0, thigh_y, 0)        hip link to thigh joint
       → Ry(theta_thigh)             thigh pitch (rotation about Y)
       → Trans(0, 0, -thigh_len)     thigh to knee
       → Ry(theta_calf)              calf pitch (rotation about Y)
       → Trans(0, 0, -calf_len)      knee to foot
    """
    R_hip = _rot_x(theta_hip)
    p_thigh = np.array(hip_xyz, dtype=float) + R_hip @ np.array([0., thigh_y, 0.])

    R_ht = R_hip @ _rot_y(theta_thigh)
    p_calf = p_thigh + R_ht @ np.array([0., 0., -thigh_len])

    R_htc = R_ht @ _rot_y(theta_calf)
    p_foot = p_calf + R_htc @ np.array([0., 0., -calf_len])

    return p_foot


# ---------------------------------------------------------------------------
# Frame retargeting
# ---------------------------------------------------------------------------

def retarget_frame(frame, src, tgt):
    """
    Retarget a single 61-dim frame from source to target robot.

    Modifications:
      - root_z       [2]      : scaled by total-limb-length ratio
      - toe_pos_local [19:31] : recomputed via FK with target limb sizes
      - lin_vel      [31:34]  : scaled by total-limb-length ratio
      - toe_vel      [49:61]  : scaled by total-limb-length ratio (approx.)
    Everything else is kept unchanged.
    """
    frame = np.array(frame, dtype=np.float64)
    out = frame.copy()

    src_limb = src["thigh_len"] + src["calf_len"]
    tgt_limb = tgt["thigh_len"] + tgt["calf_len"]
    scale = tgt_limb / src_limb          # e.g. 0.426/0.400 = 1.065 for A1→GO2

    # 1. Root height
    out[2] *= scale

    # 2. Root rotation [3:7]  — unchanged
    # 3. Joint angles  [7:19] — unchanged (radians are scale-invariant)

    # 4. Recompute toe positions [19:31] via FK
    for leg_i in range(4):
        jp = 7 + leg_i * 3    # joint-pos base index
        tp = 19 + leg_i * 3   # toe-pos base index
        p_foot = leg_fk(
            frame[jp], frame[jp + 1], frame[jp + 2],
            tgt["hip_joint_xyz"][leg_i],
            tgt["thigh_joint_y"][leg_i],
            tgt["thigh_len"],
            tgt["calf_len"],
        )
        out[tp: tp + 3] = p_foot

    # 5. Linear velocity [31:34]
    out[31:34] *= scale

    # 6. Angular velocity [34:37] — unchanged
    # 7. Joint velocities [37:49] — unchanged (rad/s, size-invariant)

    # 8. Toe velocities [49:61] (not used by discriminator, approx. scale)
    out[49:61] *= scale

    return out.tolist()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_motion_file(path):
    """Load a .txt AMP motion file (JSON with optional // comments)."""
    with open(path, "r") as f:
        text = f.read()
    text = re.sub(r"//.*", "", text)    # strip C++ style comments
    return json.loads(text)


def retarget_file(in_path, out_path, src, tgt, verify=False):
    """Retarget a single motion file and save the result."""
    data = load_motion_file(in_path)
    src_frames = data["Frames"]

    if verify:
        # Check FK reconstruction error on the first frame using SOURCE params
        f0 = np.array(src_frames[0])
        print(f"  FK reconstruction check (source={args_global.src}, "
              f"frame 0 of {Path(in_path).name}):")
        for leg_i in range(4):
            jp = 7 + leg_i * 3
            tp = 19 + leg_i * 3
            stored = f0[tp: tp + 3]
            computed = leg_fk(
                f0[jp], f0[jp + 1], f0[jp + 2],
                src["hip_joint_xyz"][leg_i],
                src["thigh_joint_y"][leg_i],
                src["thigh_len"],
                src["calf_len"],
            )
            err = np.linalg.norm(stored - computed)
            print(f"    {LEG_NAMES[leg_i]:2s}  stored={np.round(stored, 4)}  "
                  f"computed={np.round(computed, 4)}  err={err:.4f} m")
        print()

    tgt_frames = [retarget_frame(f, src, tgt) for f in src_frames]
    data["Frames"] = tgt_frames

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    src_z_mean = np.mean([fr[2] for fr in src_frames])
    tgt_z_mean = np.mean([fr[2] for fr in tgt_frames])
    print(f"  {Path(in_path).name:20s}  "
          f"{len(tgt_frames)} frames  "
          f"root_z  {src_z_mean:.4f} → {tgt_z_mean:.4f} m  "
          f"→ {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_global = None   # needed inside retarget_file for --verify label


def main():
    global args_global

    parser = argparse.ArgumentParser(
        description="Retarget AMP mocap motions to a target robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--src", default="a1", choices=list(ROBOT_PARAMS.keys()),
        help="Source robot the mocap data was recorded on (default: a1)",
    )
    parser.add_argument(
        "--tgt", required=True, choices=list(ROBOT_PARAMS.keys()),
        help="Target robot to retarget to",
    )
    parser.add_argument(
        "--input", default="datasets/mocap_motions",
        help="Input directory containing .txt motion files "
             "(default: datasets/mocap_motions)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: datasets/mocap_<tgt>/)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Print FK reconstruction error on the first frame of each file "
             "to sanity-check source robot geometry",
    )
    parser.add_argument(
        "--print-params", action="store_true",
        help="Print robot parameter table and exit",
    )
    args = parser.parse_args()
    args_global = args

    if args.print_params:
        print(f"{'Robot':<6}  {'thigh':>6}  {'calf':>6}  "
              f"{'hip_body_y(FR)':>14}  {'hip_link_y(FR)':>14}")
        print("-" * 56)
        for name, p in ROBOT_PARAMS.items():
            print(f"{name:<6}  {p['thigh_len']:>6.3f}  {p['calf_len']:>6.3f}  "
                  f"{p['hip_joint_xyz'][0, 1]:>14.5f}  "
                  f"{p['thigh_joint_y'][0]:>14.5f}")
        return

    src = ROBOT_PARAMS[args.src]
    tgt = ROBOT_PARAMS[args.tgt]

    if args.src == args.tgt:
        print("WARNING: source and target robot are the same — "
              "output is identical to input.")

    out_dir = args.output or f"datasets/mocap_{args.tgt}"

    # Collect input files
    in_path = Path(args.input)
    if in_path.is_dir():
        files = sorted(in_path.glob("*.txt"))
    else:
        files = sorted(Path(".").glob(args.input))

    if not files:
        print(f"ERROR: no .txt files found at: {args.input}", file=sys.stderr)
        sys.exit(1)

    src_limb = src["thigh_len"] + src["calf_len"]
    tgt_limb = tgt["thigh_len"] + tgt["calf_len"]
    scale = tgt_limb / src_limb

    print(f"Retargeting {len(files)} file(s):  {args.src} → {args.tgt}")
    print(f"  limb scale = {src_limb:.3f} → {tgt_limb:.3f}  (×{scale:.4f})")
    print(f"  output dir = {out_dir}")
    print()

    for f in files:
        out_file = Path(out_dir) / f.name
        retarget_file(str(f), str(out_file), src, tgt, verify=args.verify)

    print()
    print("Done. Update your AMP config:")
    print(f"  MOTION_FILES = glob.glob('{out_dir}/*')")
    print()
    print("Recommended config changes for target robot:")
    if args.tgt == "go2":
        print("  rewards.base_height_target = 0.34  # actual GO2 stance height")
        print("  init_state.pos             = [0.0, 0.0, 0.42]")
        print("  control.stiffness          = {'joint': 25}")
        print("  control.damping            = {'joint': 0.5}")
    elif args.tgt == "go1":
        print("  rewards.base_height_target = 0.30")
        print("  init_state.pos             = [0.0, 0.0, 0.42]")


if __name__ == "__main__":
    main()
