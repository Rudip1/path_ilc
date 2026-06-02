"""
Contact-task experiment on the flexible KUKA: ILC on a tool dragged across a
worktable (the paper's 'contact tasks' open problem).

    python src/main_kuka_contact.py

Writes ../outputs/k5_contact.png:
  left  - ILC convergence (log) on the contact task, with the ~press force
  right - top view of the tool path: desired circle vs trial-0 vs after-ILC
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("MUJOCO_GL", "glx")
sys.path.insert(0, os.path.dirname(__file__))
from kuka import Effects  # noqa: E402
from run_kuka import run_trial, new_ilc  # noqa: E402
from run_kuka_contact import (make_contact_plant, make_contact_reference,  # noqa: E402
                              CENTER, RADIUS)

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

HOLD, SETTLE, N_TRIALS, N = 30, 500, 8, 200


def main():
    plant = make_contact_plant(effects=Effects.realistic_repeatable())
    lambdas, xyz, qref = make_contact_reference(n=N)
    ilc = new_ilc(len(lambdas))

    print("[contact] ILC on a tool dragged across a worktable ...")
    rms_hist, max_hist, press = [], [], []
    for i in range(N_TRIALS):
        r = run_trial(plant, ilc, lambdas, xyz, qref,
                      hold=HOLD, settle=SETTLE, use_ilc=(i > 0))
        rms_hist.append(r["rms_um"]); max_hist.append(r["max_um"])
        press.append(plant.read_contact_force())
        ilc.update_from_trial(r["lambdas"], r["eq"])
    rms_hist = np.array(rms_hist); max_hist = np.array(max_hist)
    print(f"      RMS: {rms_hist[0]:.0f} -> {rms_hist[-1]:.0f} um "
          f"({rms_hist[0]/rms_hist[-1]:.1f}x), press ~{np.mean(press):.0f} N")

    r0 = run_trial(plant, ilc, lambdas, xyz, qref, hold=HOLD, settle=SETTLE,
                   use_ilc=False)
    rF = run_trial(plant, ilc, lambdas, xyz, qref, hold=HOLD, settle=SETTLE,
                   use_ilc=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].semilogy(range(len(rms_hist)), rms_hist, "o-", label="RMS")
    ax[0].semilogy(range(len(max_hist)), max_hist, "s--", alpha=0.7, label="max")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("Cartesian error [um] (log)")
    ax[0].set_title(f"Contact-task ILC convergence (~{np.mean(press):.0f} N press)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    th = np.linspace(0, 2 * np.pi, 200)
    ax[1].plot(CENTER[0] + RADIUS * np.cos(th), CENTER[1] + RADIUS * np.sin(th),
               "k-", lw=2, label="desired")
    ax[1].plot(r0["xyz_true"][:, 0], r0["xyz_true"][:, 1], "b.", ms=4,
               alpha=0.5, label="trial 0 (no ILC)")
    ax[1].plot(rF["xyz_true"][:, 0], rF["xyz_true"][:, 1], "g.", ms=4,
               alpha=0.8, label="after ILC")
    ax[1].set_aspect("equal"); ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]")
    ax[1].set_title("Tool path on the worktable (top view)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "k5_contact.png"), dpi=130)
    plt.close(fig)
    print("Done. Figure saved: outputs/k5_contact.png")


if __name__ == "__main__":
    main()
