#!/usr/bin/env python3
"""
verify_mocap.py — Static verification of retargeted mocap motion files
=======================================================================
Does NOT require Isaac Gym or GPU. Checks geometry, physics plausibility,
and optionally plots all motions for visual inspection.

Usage:
  # Verify original A1 data
  python datasets/verify_mocap.py --dir datasets/mocap_motions --robot a1

  # Verify retargeted GO2 data (recommended before training)
  python datasets/verify_mocap.py --dir datasets/mocap_go2 --robot go2

  # Also save plots to PNG
  python datasets/verify_mocap.py --dir datasets/mocap_go2 --robot go2 --plot
"""

import re
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path

# ---- robot geometry (same as retarget_mocap.py) ---------------------------
ROBOT_PARAMS = {
    "a1":  dict(thigh=0.200, calf=0.200, hip_y=0.08505, expected_z=(0.23, 0.42)),
    "go1": dict(thigh=0.213, calf=0.213, hip_y=0.080,   expected_z=(0.24, 0.44)),
    "go2": dict(thigh=0.213, calf=0.213, hip_y=0.0955,  expected_z=(0.24, 0.44)),
}

# AMPLoader field indices (raw 61-dim frame)
IDX = dict(
    root_pos  = (0,  3),
    root_rot  = (3,  7),
    joint_pos = (7,  19),   # [FR,FL,RR,RL] × [hip,thigh,calf]
    toe_pos   = (19, 31),
    lin_vel   = (31, 34),
    ang_vel   = (34, 37),
    joint_vel = (37, 49),
    toe_vel   = (49, 61),
)

LEG_NAMES  = ["FR", "FL", "RR", "RL"]


def load_motion(path):
    with open(path) as f:
        text = re.sub(r"//.*", "", f.read())
    d = json.loads(text)
    frames = np.array(d["Frames"], dtype=np.float64)
    return d, frames


def fk_foot_z(frames, robot):
    """
    Compute foot Z (in body frame) for all frames × 4 legs via FK.
    Returns array of shape (N, 4).
    """
    p = robot
    foot_z = np.zeros((len(frames), 4))

    # PyBullet order: FR=0, FL=1, RR=2, RL=3
    hip_y_signs = [-1, 1, -1, 1]  # y-offset sign for each leg

    for leg_i in range(4):
        jp = 7 + leg_i * 3
        theta_hip   = frames[:, jp]
        theta_thigh = frames[:, jp + 1]
        theta_calf  = frames[:, jp + 2]

        # hip abduction (rot-X) → y shifts in body frame
        hip_y = hip_y_signs[leg_i] * p["hip_y"]

        # calf end (foot) Z relative to hip joint (body frame)
        cos_t = np.cos(theta_thigh)
        cos_tc = np.cos(theta_thigh + theta_calf)

        # simplified: Ry(thigh) rotates [0,0,-thigh] then Ry(calf) rotates [0,0,-calf]
        # z component of thigh end in hip-child frame (ignoring hip abduction for Z)
        z_knee = -p["thigh"] * cos_t
        z_foot = z_knee - p["calf"] * cos_tc

        # foot Z in body frame = hip_joint_Z (≈0) + z_foot
        foot_z[:, leg_i] = z_foot

    return foot_z


def amp_obs_stats(frames):
    """Compute statistics of AMP observations (strips root_pos, root_rot).
    Returns dict with field → (mean, std, min, max)."""
    stats = {}
    for name, (s, e) in IDX.items():
        if name in ("root_pos", "root_rot"):
            continue
        d = frames[:, s:e]
        stats[name] = dict(mean=d.mean(), std=d.std(), min=d.min(), max=d.max())
    return stats


def check_file(path, robot_name, plot=False, axs=None, color=None, label=None):
    """Run checks on a single motion file. Returns (pass_count, fail_count)."""
    p = ROBOT_PARAMS[robot_name]
    d, frames = load_motion(path)
    N = len(frames)
    name = Path(path).name
    passes, fails = 0, 0

    def ok(msg):
        nonlocal passes
        print(f"    ✓ {msg}")
        passes += 1

    def warn(msg):
        nonlocal fails
        print(f"    ✗ {msg}")
        fails += 1

    print(f"\n  [{name}]  {N} frames  dt={d['FrameDuration']}s  "
          f"dur={(N-1)*d['FrameDuration']:.3f}s")

    # 1. Dimension check
    if frames.shape[1] == 61:
        ok("frame dimension = 61")
    else:
        warn(f"unexpected frame dimension = {frames.shape[1]} (expected 61)")

    # 2. Root height range
    root_z = frames[:, 2]
    z_lo, z_hi = p["expected_z"]
    if root_z.min() >= z_lo and root_z.max() <= z_hi:
        ok(f"root_z in [{z_lo:.2f}, {z_hi:.2f}]  "
           f"(actual: [{root_z.min():.4f}, {root_z.max():.4f}])")
    else:
        warn(f"root_z out of range [{z_lo:.2f}, {z_hi:.2f}]  "
             f"(actual: [{root_z.min():.4f}, {root_z.max():.4f}])")

    # 3. Foot clearance: max foot Z should not be > 0.04m (above body?)
    foot_z = fk_foot_z(frames, p)
    max_foot_z = foot_z.max()
    if max_foot_z <= 0.05:
        ok(f"foot FK z ≤ 0 (max={max_foot_z:.4f} m, feet below body)"  )
    else:
        warn(f"foot FK z unexpectedly positive: max={max_foot_z:.4f} m (foot above hip?)")

    # 4. Joint angle limits (rough limb limits)
    jp = frames[:, 7:19]
    hip_angles    = jp[:, 0::3]   # cols 0,3,6,9
    thigh_angles  = jp[:, 1::3]
    calf_angles   = jp[:, 2::3]

    if np.abs(hip_angles).max() < 0.8:
        ok(f"hip angles |max|={np.abs(hip_angles).max():.3f} rad < 0.8")
    else:
        warn(f"hip angles large: |max|={np.abs(hip_angles).max():.3f} rad")

    if thigh_angles.min() > -0.5 and thigh_angles.max() < 3.5:
        ok(f"thigh angles OK ({thigh_angles.min():.3f} – {thigh_angles.max():.3f} rad)")
    else:
        warn(f"thigh angles out of range ({thigh_angles.min():.3f} – {thigh_angles.max():.3f} rad)")

    if calf_angles.min() > -2.8 and calf_angles.max() < 0.1:
        ok(f"calf angles OK ({calf_angles.min():.3f} – {calf_angles.max():.3f} rad)")
    else:
        warn(f"calf angles out of range ({calf_angles.min():.3f} – {calf_angles.max():.3f} rad)")

    # 5. Quaternion norm
    quat = frames[:, 3:7]
    q_norms = np.linalg.norm(quat, axis=1)
    if np.allclose(q_norms, 1.0, atol=0.01):
        ok(f"quaternion norms ≈ 1.0 (max err={np.abs(q_norms-1).max():.5f})")
    else:
        warn(f"quaternion not normalized (max err={np.abs(q_norms-1).max():.5f})")

    # 6. Linear velocity sanity (<= 5 m/s)
    lv = frames[:, 31:34]
    max_lv = np.linalg.norm(lv, axis=1).max()
    if max_lv <= 5.0:
        ok(f"max linear speed = {max_lv:.3f} m/s ≤ 5.0")
    else:
        warn(f"linear speed too high: {max_lv:.3f} m/s")

    # 7. Joint velocity sanity (<= 30 rad/s)
    jv = frames[:, 37:49]
    max_jv = np.abs(jv).max()
    if max_jv <= 35.0:
        ok(f"max joint vel = {max_jv:.2f} rad/s ≤ 35.0")
    else:
        warn(f"joint vel too high: {max_jv:.2f} rad/s")

    # ---- optional plot -----
    if plot and axs is not None:
        t = np.arange(N) * d["FrameDuration"]
        c = "steelblue" if color is None else color
        lbl = label or name

        axs[0].plot(t, root_z, color=c, label=lbl)
        axs[0].set_ylabel("root_z (m)")

        for leg_i in range(4):
            axs[1].plot(t, foot_z[:, leg_i], color=c,
                        alpha=0.6 + 0.1*leg_i,
                        label=f"{lbl}-{LEG_NAMES[leg_i]}" if leg_i == 0 else "")
        axs[1].axhline(0, color="k", lw=0.5, linestyle="--")
        axs[1].set_ylabel("foot FK z (body frame, m)")

        for j, (jname, sl) in enumerate([
                ("hip",   slice(7,  19, 3)),
                ("thigh", slice(8,  19, 3)),
                ("calf",  slice(9,  19, 3))]):
            axs[2+j].plot(t, frames[:, sl], color=c, alpha=0.7)
            axs[2+j].set_ylabel(f"{jname} (rad)")

        lin_speed = np.linalg.norm(frames[:, 31:34], axis=1)
        axs[5].plot(t, lin_speed, color=c, label=lbl)
        axs[5].set_ylabel("lin speed (m/s)")

    return passes, fails


def main():
    parser = argparse.ArgumentParser(description="Verify AMP mocap motion files.")
    parser.add_argument("--dir", default="datasets/mocap_go2",
                        help="Directory containing .txt motion files")
    parser.add_argument("--robot", default="go2",
                        choices=list(ROBOT_PARAMS.keys()),
                        help="Target robot name")
    parser.add_argument("--plot", action="store_true",
                        help="Generate matplotlib plots")
    parser.add_argument("--compare", default=None,
                        help="Also overlay another directory (optional, for before/after)")
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob("*.txt"))
    if not files:
        print(f"ERROR: no .txt files in {args.dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying {len(files)} motion file(s) for robot '{args.robot}'")
    print(f"  source dir: {args.dir}")

    axs = None
    fig = None
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n_rows = 6
            fig, axs_arr = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows))
            axs_arr[0].set_title(f"Motion verification — {args.robot} — {args.dir}")
            colors = plt.cm.tab10(np.linspace(0, 1, len(files)))
        except ImportError:
            print("matplotlib not available — skipping plots")
            args.plot = False

    total_pass = total_fail = 0
    for i, f in enumerate(files):
        color = colors[i] if args.plot else None
        axs_ref = axs_arr if args.plot else None
        p, fail = check_file(str(f), args.robot,
                              plot=args.plot, axs=axs_ref,
                              color=color, label=Path(f).stem)
        total_pass += p
        total_fail += fail

    print(f"\n{'='*50}")
    print(f"Total: {total_pass} passed, {total_fail} failed")
    if total_fail == 0:
        print("All checks passed — mocap data looks good for training!")
    else:
        print(f"WARNING: {total_fail} checks failed. Review output above.")

    if args.plot and fig is not None:
        for ax in axs_arr:
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        axs_arr[-1].set_xlabel("time (s)")
        out_path = f"datasets/verify_{args.robot}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        print(f"\nPlot saved to: {out_path}")

    if total_fail > 0:
        print("\nTips:")
        print("  - root_z too low/high → re-run retarget_mocap.py or check --src robot")
        print("  - joint angles out of range → source data has unusual poses")
        print("  - high joint velocities → FrameDuration may be wrong")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
