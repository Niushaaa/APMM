#!/usr/bin/env python3
"""Experiment 4 paper figures from results_exp4_kalshi/per_game.json:
(1) per-game parlay P&L vs M for APMM vs ind; (2) APMM-vs-ind per-game scatter;
(3) per-game APMM/ind loss ratio vs M. (native is reported in the table, not plotted)."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results_exp4_kalshi"
COL = {"ind": "tab:red", "APMM": "tab:blue"}


def main():
    d = json.load(open(f"{OUT}/per_game.json"))
    M = np.array([r["M"] for r in d])
    ind = np.array([r["ind_parlay"] for r in d])
    ap = np.array([r["apmm_parlay"] for r in d])

    # (1) parlay P&L vs M (y clipped to the bulk; a few ind points are off-scale)
    hi = float(np.percentile(ind, 92))
    lo = min(float(ap.min()), 0.0) * 1.1 - 5
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(M, np.clip(ind, lo, hi), s=34, color=COL["ind"], marker="s", alpha=0.75, label="ind")
    ax.scatter(M, np.clip(ap, lo, hi), s=34, color=COL["APMM"], marker="o", alpha=0.75, label="APMM")
    ax.axhline(0, color="0.6", lw=1, zorder=0)
    n_off = int((ind > hi).sum())
    ax.set_ylim(lo, hi * 1.05)
    ax.set_xlabel("number of base markets $M$")
    ax.set_ylabel("operator parlay loss")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.98, 0.95, f"{n_off} ind points off-scale\n(up to {ind.max():.0f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=COL["ind"])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_parlay_loss_vs_M.{ext}", dpi=200)
    plt.close(fig)

    # (2) APMM vs ind per game
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    lim = float(max(np.abs(ind).max(), np.abs(ap).max())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=1.2, label="APMM = ind")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    below = int((ap <= ind + 1e-9).sum())
    ax.scatter(ind, ap, s=34, color=COL["APMM"], alpha=0.75, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("ind parlay loss (per game)")
    ax.set_ylabel("APMM parlay loss (per game)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.97, 0.04, f"APMM $\\leq$ ind in {below}/{len(d)} games\n"
            f"corpus: ind {ind.sum():.0f}  vs  APMM {ap.sum():.0f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_apmm_vs_ind.{ext}", dpi=200)
    plt.close(fig)

    # (3) loss gap (ind - APMM) vs M -- gap > 0  <=>  APMM loses less (sign-robust)
    gap = ind - ap
    hi = float(np.percentile(gap, 90)); lo = min(float(gap.min()), 0.0) * 1.15 - 5
    n_better = int((gap > 1e-9).sum()); n_off = int((gap > hi).sum())
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(M, np.clip(gap, lo, hi), s=34, color="tab:purple", alpha=0.75)
    ax.axhline(0, color="0.5", ls="--", lw=1.2)
    ax.set_ylim(lo, hi * 1.05)
    ax.set_xlabel("number of base markets $M$")
    ax.set_ylabel("loss gap   ind $-$ APMM   (>0 ⇒ APMM better)")
    ax.text(0.97, 0.96, f"gap>0 in {n_better}/{len(d)} games\n"
            f"corpus gap = +{gap.sum():,.0f}\n{n_off} points off-scale (up to {gap.max():,.0f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_loss_gap_vs_M.{ext}", dpi=200)
    plt.close(fig)
    # (4) corpus operator loss by parlay order |S| (grouped bars)
    levels = list(range(2, 9))

    def lvlsum(key, j):
        return sum(float(r.get(key, {}).get(str(j), r.get(key, {}).get(j, 0.0))) for r in d)
    ind_l = [lvlsum("ind_by_level", j) for j in levels]
    ap_l = [lvlsum("apmm_by_level", j) for j in levels]
    x = np.arange(len(levels)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.bar(x - w / 2, ind_l, w, color=COL["ind"], label="ind")
    ax.bar(x + w / 2, ap_l, w, color=COL["APMM"], label="APMM")
    ax.axhline(0, color="0.6", lw=1, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_xlabel("parlay order  $|S|$")
    ax.set_ylabel("operator loss (corpus total)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_loss_by_order.{ext}", dpi=200)
    plt.close(fig)

    print(f"saved fig_parlay_loss_vs_M, fig_apmm_vs_ind, fig_loss_gap_vs_M, "
          f"fig_loss_by_order  (gap>0 in {int((ind-ap>1e-9).sum())}/{len(d)} games, "
          f"corpus gap +{(ind-ap).sum():,.0f})")
    print("corpus loss by order |S|:")
    for j, il, al in zip(levels, ind_l, ap_l):
        print(f"  |S|={j}:  ind={il:11.1f}  apmm={al:9.2f}")


if __name__ == "__main__":
    main()
