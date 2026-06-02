"""
The four PhD experiments on the FLEXIBLE-JOINT KUKA iiwa14.

    python src/main_kuka.py

Produces in ../outputs:
  k1_convergence.png  - ILC convergence on a 3D base path (joint-side encoders only)
  k2_speed.png        - reusing the learned table at different traversal speeds
  k3_transfer.png     - the open problem: a learned table on a NEW 3D path
  k4_ai_transfer.png  - the AI layer predicting a correction for an unseen path

Same algorithm as the toy demo, now on a real 7-DOF industrial arm whose hidden
error is genuine joint elasticity / transmission deflection (see kuka.py).
All errors are Cartesian RMS at the TCP, in micrometers (um).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from kuka import FlexArmPlant, Effects   # noqa: E402
from run_kuka import (make_reference, make_reference_morph, run_trial,  # noqa: E402
                      new_ilc, _CENTER)
import ai  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

N = 200
HOLD = 30
SETTLE = 500
N_TRIALS = 7
# headline experiments run under the realistic (repeatable) drivetrain effects:
# transmission error + Stribeck friction + backlash + encoder noise. Thermal
# drift is non-repetitive and is studied separately (figures_realism / dashboards).
HEADLINE = Effects.realistic_repeatable()


def train_ilc(plant, kind="A", n_trials=N_TRIALS):
    lambdas, xyz, qref = make_reference(kind, n=N)
    ilc = new_ilc(len(lambdas))
    rms_hist, max_hist, last = [], [], None
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xyz, qref,
                      hold=HOLD, settle=SETTLE, use_ilc=(i > 0))
        rms_hist.append(r["rms_um"]); max_hist.append(r["max_um"])
        ilc.update_from_trial(r["lambdas"], r["eq"])
        last = r
    return ilc, (lambdas, xyz, qref), np.array(rms_hist), np.array(max_hist), last


def train_ilc_morph(plant, m, n_trials=5):
    lambdas, xyz, qref = make_reference_morph(m, n=N)
    ilc = new_ilc(len(lambdas))
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xyz, qref,
                      hold=HOLD, settle=SETTLE, use_ilc=(i > 0))
        ilc.update_from_trial(r["lambdas"], r["eq"])
    return ilc, (lambdas, xyz, qref)


def main():
    plant = FlexArmPlant(effects=HEADLINE)

    # ---- 1) convergence on the 3D base path ----------------------------
    print("[1/4] ILC convergence on KUKA base path (joint-side encoders only) ...")
    ilcA, refA, rmsA, maxA, lastA = train_ilc(plant, "A")
    lambdas, xyzA, qrefA = refA
    print(f"      RMS: {rmsA[0]:.0f} -> {rmsA[-1]:.0f} um  ({rmsA[0]/rmsA[-1]:.1f}x)")

    r0 = run_trial(plant, ilcA, lambdas, xyzA, qrefA, hold=HOLD,
                   settle=SETTLE, use_ilc=False)
    rF = run_trial(plant, ilcA, lambdas, xyzA, qrefA, hold=HOLD,
                   settle=SETTLE, use_ilc=True)

    fig = plt.figure(figsize=(11, 4.4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.semilogy(range(len(rmsA)), rmsA, "o-", label="RMS")
    ax0.semilogy(range(len(maxA)), maxA, "s--", alpha=0.7, label="max")
    ax0.set_xlabel("trial"); ax0.set_ylabel("Cartesian error [um] (log)")
    ax0.set_title("KUKA iiwa14 ILC convergence"); ax0.legend(); ax0.grid(alpha=0.3)
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.plot(xyzA[:, 0], xyzA[:, 1], xyzA[:, 2], "k-", lw=2, label="desired")
    ax1.plot(r0["xyz_true"][:, 0], r0["xyz_true"][:, 1], r0["xyz_true"][:, 2],
             "b.", ms=3, alpha=0.5, label="trial 0 (no ILC)")
    ax1.plot(rF["xyz_true"][:, 0], rF["xyz_true"][:, 1], rF["xyz_true"][:, 2],
             "g.", ms=3, alpha=0.8, label="after ILC")
    ax1.set_title("3D path tracking"); ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "k1_convergence.png"), dpi=130)
    plt.close(fig)

    # ---- 2) speed generalization (reuse table at other speeds) ---------
    print("[2/4] Speed generalization (reuse learned table) ...")
    holds = [20, 30, 45]
    rms_by = {h: run_trial(plant, ilcA, lambdas, xyzA, qrefA, hold=h,
                           settle=SETTLE, use_ilc=True)["rms_um"] for h in holds}
    r_nob = run_trial(plant, ilcA, lambdas, xyzA, qrefA, hold=HOLD,
                      settle=SETTLE, use_ilc=False)
    for h in holds:
        tag = " (trained)" if h == HOLD else ""
        print(f"      hold {h} steps: {rms_by[h]:.0f} um{tag}")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [f"{h}\n(trained)" if h == HOLD else f"{h}" for h in holds]
    ax.bar(labels, [rms_by[h] for h in holds], color="#4477aa")
    ax.axhline(r_nob["rms_um"], color="crimson", ls="--",
               label=f"no ILC ({r_nob['rms_um']:.0f} um)")
    ax.set_xlabel("steps held per path point (slower ->)")
    ax.set_ylabel("Cartesian RMS error [um]")
    ax.set_title("Reusing the learned table at different speeds")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "k2_speed.png"), dpi=130)
    plt.close(fig)

    # ---- 3) transfer to a different 3D path (the open problem) ---------
    print("[3/4] Transfer to a new path B (open problem) ...")
    lamB, xyzB, qrefB = make_reference("B", n=N)
    ilc_naive = new_ilc(len(lamB)); ilc_naive.load_table(ilcA.correction_table())
    r_naive = run_trial(plant, ilc_naive, lamB, xyzB, qrefB, hold=HOLD,
                        settle=SETTLE, use_ilc=True)
    r_noneB = run_trial(plant, ilc_naive, lamB, xyzB, qrefB, hold=HOLD,
                        settle=SETTLE, use_ilc=False)
    ilcB, _, rmsB, _, _ = train_ilc(plant, "B")
    print(f"      no correction: {r_noneB['rms_um']:.0f} um")
    print(f"      naive A->B:    {r_naive['rms_um']:.0f} um")
    print(f"      relearn on B:  {rmsB[-1]:.0f} um  ({len(rmsB)} trials)")

    # ---- 4) AI layer: predict a correction for an unseen path ----------
    print("[4/4] AI layer: predict correction for unseen path ...")
    ai_vals = None
    if ai._HAS_TORCH:
        feats, tables = [], []
        rng = np.random.default_rng(0)
        morphs = list(np.linspace(0.0, 1.0, 9)) + list(rng.uniform(0, 1, 5))
        for m in morphs:
            ilc_m, ref_m = train_ilc_morph(plant, m, n_trials=5)
            lam_m, xyz_m, _ = ref_m
            feats.append(ai.path_features(lam_m, xyz_m))
            tables.append(ilc_m.correction_table().reshape(-1))
        net, norm = ai.train_correction_net(np.array(feats), np.array(tables),
                                            epochs=800)
        test_m = 0.72
        lam_t, xyz_t, qref_t = make_reference_morph(test_m, n=N)
        pred = ai.predict_table(net, norm, ai.path_features(lam_t, xyz_t),
                                N=len(lam_t), dof=7)
        ilc_ai = new_ilc(len(lam_t)); ilc_ai.load_table(pred)
        r_ai = run_trial(plant, ilc_ai, lam_t, xyz_t, qref_t, hold=HOLD,
                         settle=SETTLE, use_ilc=True)
        r_none_t = run_trial(plant, ilc_ai, lam_t, xyz_t, qref_t, hold=HOLD,
                             settle=SETTLE, use_ilc=False)
        ilcA0, _ = train_ilc_morph(plant, 0.0, n_trials=5)
        ilc_nv = new_ilc(len(lam_t)); ilc_nv.load_table(ilcA0.correction_table())
        r_nv = run_trial(plant, ilc_nv, lam_t, xyz_t, qref_t, hold=HOLD,
                         settle=SETTLE, use_ilc=True)
        ai_vals = [r_none_t["rms_um"], r_nv["rms_um"], r_ai["rms_um"]]
        print(f"      unseen path  no correction: {ai_vals[0]:.0f} um")
        print(f"      unseen path  naive (A table): {ai_vals[1]:.0f} um")
        print(f"      unseen path  AI (0 trials):   {ai_vals[2]:.0f} um")

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        names = ["no\ncorrection", "naive\n(base table)", "AI predicted\n(0 trials)"]
        ax.bar(names, ai_vals, color=["#cc6677", "#ddaa33", "#228833"])
        for i, v in enumerate(ai_vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
        ax.set_ylabel("Cartesian RMS error [um]")
        ax.set_title("AI correction on an UNSEEN 3D path (zero trials)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "k4_ai_transfer.png"), dpi=130)
        plt.close(fig)
    else:
        print("      torch not available; skipping AI layer.")

    # transfer figure
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    names = ["no\ncorrection", "naive\nA->B", f"relearn B\n({len(rmsB)} trials)"]
    vals = [r_noneB["rms_um"], r_naive["rms_um"], rmsB[-1]]
    ax.bar(names, vals, color=["#cc6677", "#ddaa33", "#4477aa"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    ax.set_ylabel("Cartesian RMS error [um]")
    ax.set_title("Transfer to a different 3D path (the open problem)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "k3_transfer.png"), dpi=130)
    plt.close(fig)

    print("\nDone. Figures saved in outputs/ (k1..k4).")


if __name__ == "__main__":
    main()
