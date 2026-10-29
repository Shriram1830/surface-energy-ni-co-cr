"""Figures of the report. Start it from the repository root; it reads data/*.csv, so no raw LAMMPS output is needed.

  python src/plots.py planes        Fig. 2  surface energy by orientation       -> figures/fig_planes.png
  python src/plots.py paired        Fig. 4  paired change from 300 to 700 K     -> figures/fig_temperature_paired.png
  python src/plots.py convergence   Fig. 1  convergence tests                   -> figures/fig_convergence.png
  python src/plots.py mixing        Fig. 3  rule of mixtures, (111), 500 K      -> figures/fig_mixing_111_500K.png
  python src/plots.py all

Error bars are 1 SD over the 3 random configurations; dots are the individual configurations.
Fig. 1 panels
  (a)   lattice parameter from the NPT box lengths, running mean over the full run
  (b,c) total potential energy of the bulk and slab cells, running mean over the full run
  (d)   surface energy from the running means, production run only (equilibration excluded)
  (e,f) surface energy against slab thickness and in-plane cell area
Green bands mark +-1 % of the reference surface energy (in b, c: the equivalent change in energy).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis import (BASE_TAG, COMP_LABEL, COMPS, EV_A2_TO_MJ_M2, NI_PCT, PLANES, ROOT, TEMPS,
                      load_conv, load_production, load_series, sd)
from build_cell import ORIENT

PLN = {"111": "(111)", "100": "(100)", "110": "(110)"}
PLANE_COL = {"111": "#1f6fb4", "100": "#e07b00", "110": "#2a9d5b"}
COMP_COL = {"ni22": "#b39ddb", "ni33": "#7e57c2", "ni44": "#311b92"}
GAMMA_LABEL = "Surface energy (mJ/m$^2$)"

STYLE = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "axes.spines.top": False,
         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
         "figure.dpi": 110, "savefig.bbox": "tight"}
STYLE_CONV = {   # extra settings for Fig. 1 only
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False}

DT, NEQ = 0.002, 5000            # ps per step, equilibration steps
T_EQ = NEQ * DT                  # ps; this part is never used for the production averages
AVG_FROM = 0.0                   # ps; start of the running means in a-c (set to T_EQ to hide equilibration data)
GLAB = r"Surface energy (mJ m$^{-2}$)"
GREEN, GREY = "#2a9d5b", "#555555"
TOL_SIZE = 0.03                  # half-width of the convergence band in (e): +-3 % of the production-size value
LATCOL = ["#1f6fb4", "#e07b00", "#2a9d5b"]


def figure_path(root, name):
    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    return os.path.join(root, "figures", name)


# ------------------------------------------------------------------ Fig. 2: orientation
def planes(root):
    prod = load_production(root)
    get = lambda c, p, T: prod[(c, p, T)][0]
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for a, T in zip(axs, TEMPS):
            for j, p in enumerate(PLANES):
                for i, c in enumerate(COMPS):
                    g = get(c, p, T)
                    x = i + (j - 1) * 0.27
                    a.bar(x, g.mean(), 0.25, color=PLANE_COL[p], label=PLN[p], yerr=sd(g), capsize=3,
                          error_kw={"lw": 1.2})
                    a.plot([x] * len(g), g, "k.", ms=4, alpha=0.6)
            a.set_xticks(range(3))
            a.set_xticklabels([COMP_LABEL[c] for c in COMPS], fontsize=9)
            a.set_title(f"{T} K")
            a.grid(axis="x", visible=False)
        axs[0].set_ylabel(GAMMA_LABEL)

        handles, labels = axs[0].get_legend_handles_labels()      # legend next to the title
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), title="surface", loc="center", bbox_to_anchor=(0.68, 0.955),
                   ncol=3, frameon=True, facecolor="white", edgecolor="none", fontsize=8.5, title_fontsize=8.5)
        fig.suptitle("Surface energy by orientation", fontsize=13, x=0.38, y=0.955)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(figure_path(root, "fig_planes.png"), dpi=200)
    print("wrote figures/fig_planes.png")


# ------------------------------------------------------------------ Fig. 4: paired temperature change
def paired(root):
    prod = load_production(root)
    get = lambda c, p, T: prod[(c, p, T)][0]
    with plt.rc_context(STYLE):
        fig, a = plt.subplots(figsize=(8, 4.2))
        x = 0
        ticks, tl, group_edges = [], [], []
        for p in PLANES:
            start = x
            for c in COMPS:
                d = get(c, p, 700) - get(c, p, 300)
                offs = np.linspace(-0.12, 0.12, len(d))             # individual configurations, offset to avoid overlap
                a.plot(x + offs, d, "o", ms=3.5, color=COMP_COL[c], mec="k", mew=0.5, zorder=2)
                a.errorbar(x, d.mean(), yerr=sd(d), fmt="s", ms=6, color=COMP_COL[c], mec="k", mew=0.8,
                           ecolor="k", elinewidth=1.0, capsize=3, capthick=1.0, zorder=3,
                           label=COMP_LABEL[c] if p == PLANES[0] else None)
                x += 1
            group_edges.append((start - 0.5, x - 0.5))
            ticks.append((start + x - 1) / 2)
            tl.append(PLN[p])
            x += 0.8

        for (_, hi), (lo, _) in zip(group_edges[:-1], group_edges[1:]):     # separators between plane groups
            a.axvline((hi + lo) / 2, color="0.6", lw=0.7, ls="--", zorder=0)

        a.axhline(0, color="k", lw=0.8, zorder=1)
        a.set_xlim(group_edges[0][0] - 0.3, group_edges[-1][1] + 0.3)
        a.set_ylim(-25, 25)
        a.set_xticks(ticks)
        a.set_xticklabels(tl, fontsize=11)
        a.tick_params(axis="x", length=0)
        a.set_ylabel(r"$\gamma$(700 K) $-$ $\gamma$(300 K)  (mJ/m$^2$)")
        a.set_title("Change in surface energy from 300 K to 700 K", fontsize=11)
        a.grid(False)

        leg1 = a.legend(fontsize=9, loc="upper right", frameon=False, ncol=3)      # legend 1: compositions (colours)
        a.add_artist(leg1)
        key = [Line2D([], [], marker="s", ls="-", color="k", mfc="0.7", mec="k", ms=6, mew=0.8, lw=1.0,
                      label="mean ± 1 SD (3 configurations)"),
               Line2D([], [], marker="o", ls="", mfc="0.7", mec="k", ms=3.5, mew=0.5,
                      label="individual configuration")]
        a.legend(handles=key, fontsize=8.5, loc="lower left", frameon=False)    # legend 2: meaning of the markers
        fig.tight_layout()
        fig.savefig(figure_path(root, "fig_temperature_paired.png"), dpi=300)
    print("wrote figures/fig_temperature_paired.png")


# ------------------------------------------------------------------ Fig. 1: convergence
def running(x, i0):
    """Cumulative mean of x[i0:], NaN before i0."""
    out = np.full(len(x), np.nan)
    out[i0:] = np.cumsum(x[i0:]) / np.arange(1, len(x) - i0 + 1)
    return out


def panel_label(a, s):
    a.text(0, 1.03, s, transform=a.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")


def time_axis(a, tmax):
    a.axvspan(0, T_EQ, color="0.90", lw=0, zorder=0, label="equilibration")   # named in the legend
    a.set_xlim(0, tmax)
    a.set_xlabel("Time (ps)")


def plain_y(a):
    a.ticklabel_format(axis="y", useOffset=False, style="plain")


def convergence(root):
    prod = load_production(root)
    conv, ser = load_conv(root), load_series(root)
    base = conv.get(BASE_TAG)
    per, nxyz0 = ORIENT["111"]["per"], ORIENT["111"]["default"]
    sd_conf = sd(prod[("ni33", "111", 700)][0]) if ("ni33", "111", 700) in prod else None

    def size_panel(a, xs, ys, es, ns, xlabel, ref, ref_x, label, tol_size=TOL_SIZE):
        a.axhspan(ref * (1 - tol_size), ref * (1 + tol_size), color=GREEN, alpha=0.25, lw=0,
                  label=rf"$\pm${100 * tol_size:.0f} % in $\gamma$")
        if sd_conf:
            a.axhline(ref + sd_conf, color=GREY, ls=":", lw=1.0, label=r"$\pm$1 SD, 3 configurations")
            a.axhline(ref - sd_conf, color=GREY, ls=":", lw=1.0)
        a.errorbar(xs, ys, yerr=es, fmt="o-", color="k", ms=3.5, lw=0.8, elinewidth=0.8, capsize=2, zorder=3)
        a.plot(ref_x, ref, "o", ms=8, mfc="none", mec="k", mew=0.7, zorder=4, label="production size")
        a.margins(x=0.08)
        a.set_ylim(ref * (1 - 3 * TOL_SIZE), ref * (1 + 3 * TOL_SIZE))
        a.set_xticks(xs)
        a.set_xticklabels([(f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}") + f"\n({nat})"
                           for x, nat in zip(xs, ns)], fontsize=6)
        a.set_xlabel(xlabel + "\n(atoms in parentheses)", fontsize=7)
        a.set_ylabel(GLAB)
        plain_y(a)
        a.legend(loc="upper right", frameon=False, handlelength=1.5)
        panel_label(a, label)

    def scan(tags):
        return [(nxyz, conv[tag]) for nxyz, tag in tags if tag in conv]

    with plt.rc_context({**STYLE, **STYLE_CONV}):
        fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.8))

        # ---- (a)-(d): time series of the production-size case
        if ser is not None and base is not None:
            N, A = base["N"], base["A"]
            t = ser["t_ps"]
            box = np.column_stack([ser["Lx_A"], ser["Ly_A"], ser["Lz_A"]])
            pe = {"bulk": ser["PE_bulk_eV"], "slab": ser["PE_slab_eV"]}
            i0 = int(np.searchsorted(t, AVG_FROM))

            a = ax[0, 0]                                            # (a) lattice parameter during the NPT run
            time_axis(a, t[-1])
            finals = []
            for i, (c, col) in enumerate(zip("xyz", LATCOL)):
                lat = box[:, i] / (nxyz0[i] * per[i])
                rm = running(lat, i0)
                a.plot(t, rm, color=col, lw=1.2, label=f"$L_{c}$")
                finals.append(rm[-1])
            pad = 0.001 * np.mean(finals)
            a.set_ylim(min(finals) - pad, max(finals) + 2.5 * pad)
            a.set_ylabel(r"Lattice parameter (Å)")
            a.legend(ncol=2, loc="upper right", frameon=False, handlelength=1.2, columnspacing=0.8)
            panel_label(a, "(a)")

            ip = int(np.searchsorted(t, T_EQ))                      # production starts here
            rb, rs = running(pe["bulk"], ip), running(pe["slab"], ip)
            g_final = ((rs - rb) / (2 * A) * EV_A2_TO_MJ_M2)[-1]    # production-only value: centre of all bands
            gam = (running(pe["slab"], i0) - running(pe["bulk"], i0)) / (2 * A) * EV_A2_TO_MJ_M2
            dE = 0.01 * g_final * 2 * A / EV_A2_TO_MJ_M2            # energy equivalent of 1 % in gamma (eV)

            for k, (kind, lab) in enumerate((("bulk", "(b)"), ("slab", "(c)"))):      # (b), (c)
                a = ax[0, 1 + k]
                time_axis(a, t[-1])
                e = pe[kind]
                rm = running(e, i0)
                a.axhspan(rm[-1] - dE, rm[-1] + dE, color=GREEN, alpha=0.25, lw=0, label=r"$\pm$1 % in $\gamma$")
                a.plot(t, rm, color="k", lw=1.2)
                a.set_ylim(rm[-1] - 4 * dE, rm[-1] + 4 * dE)
                a.set_ylabel(f"Potential energy, {kind} (eV)")
                plain_y(a)
                a.legend(loc="upper right", frameon=False, handlelength=1.2)
                panel_label(a, lab)
                print(f"{kind}: final mean {rm[-1]:.2f} eV, rms of instantaneous values (production) {np.std(e[ip:]):.2f} eV")

            a = ax[1, 0]                                            # (d) surface energy
            time_axis(a, t[-1])
            a.axhspan(g_final * 0.99, g_final * 1.01, color=GREEN, alpha=0.25, lw=0, label=r"$\pm$1 % in $\gamma$")
            a.plot(t, gam, "k", lw=1.2)
            a.set_ylim(g_final * 0.95, g_final * 1.05)
            a.set_ylabel(GLAB)
            plain_y(a)
            a.legend(loc="upper right", frameon=False, handlelength=1.2)
            panel_label(a, "(d)")
        else:
            print("missing data/conv_base_series.csv or the base case in data/convergence.csv -> panels a-d skipped")
            for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
                a.axis("off")

        # ---- (e), (f): size scans
        if base is not None:
            ref = base["gamma"]
            err = lambda r: 0 if np.isnan(r["err"]) else r["err"]
            th = scan([((8, 5, z), f"c700_ni33_111_8x5x{z}_s1") for z in (3, 4, 5, 6, 8)])
            th.sort(key=lambda s: s[0][2])
            size_panel(ax[1, 1], [3 * s[0][2] for s in th], [s[1]["gamma"] for s in th], [err(s[1]) for s in th],
                       [s[1]["N"] for s in th], "Slab thickness (atomic layers)", ref, 3 * nxyz0[2], "(e)", tol_size=0.03)
            ar = scan([((x, y, 5), f"c700_ni33_111_{x}x{y}x5_s1") for x, y in ((6, 4), (8, 5), (11, 7), (14, 9))])
            ar.sort(key=lambda s: s[1]["A"])
            size_panel(ax[1, 2], [s[1]["A"] / 100 for s in ar], [s[1]["gamma"] for s in ar], [err(s[1]) for s in ar],
                       [s[1]["N"] for s in ar], r"In-plane cell area (nm$^2$)", ref, base["A"] / 100, "(f)", tol_size=0.01)
        else:
            ax[1, 1].axis("off")
            ax[1, 2].axis("off")

        fig.suptitle(r"Convergence tests: Ni33 Alloy, (111) surface, 700 K (one random configuration for a-d)",
                     fontsize=9, y=0.995)
        fig.tight_layout(h_pad=1.4, w_pad=1.2, rect=(0, 0, 1, 0.96))
        fig.savefig(figure_path(root, "fig_convergence.png"), dpi=600)
    print("wrote figures/fig_convergence.png")


# ------------------------------------------------------------------ Fig. 3: rule of mixtures
def mixing(root):
    PL, T = "111", 500
    prod, conv = load_production(root), load_conv(root)
    pure = {}
    for el, name in (("ni", "Ni"), ("co", "Co"), ("cr", "Cr")):
        r = conv.get(f"p{T}_{el}_{PL}_s1")
        if r:
            pure[name] = r["gamma"]
    al = {c: prod[(c, PL, T)][0] for c in COMPS}
    mean = {c: al[c].mean() for c in COMPS}
    PURE_COL = {"Ni": "#2a9d5b", "Co": "#1f6fb4", "Cr": "#d62728"}
    MIX = dict(marker="D", mfc="white", mec="#555", mew=1.6, ms=8)     # one style for every diamond

    with plt.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})

        # (a) bars: pure metals, Co-Cr mean, then alloys
        xs, labels, tops = [], [], []
        pos = 0.0
        for name in ("Ni", "Co", "Cr"):
            if name in pure:
                a1.bar(pos, pure[name], width=0.7, color=PURE_COL[name], edgecolor="k", lw=0.8)
                tops.append((pos, pure[name]))
                labels.append(f"pure {name}")
                xs.append(pos)
                pos += 1

        has_cc = "Co" in pure and "Cr" in pure
        if has_cc:                                                  # Co-Cr mean: average of the two pure values
            cc_mean = (pure["Co"] + pure["Cr"]) / 2
            a1.bar(pos, cc_mean, width=0.7, color="#F5ED0EF7", edgecolor="k", lw=0.8)
            tops.append((pos, cc_mean))
            labels.append("Co–Cr\nmean")
            xs.append(pos)
            pos += 1

        pos += 0.4
        for c in COMPS:
            s = sd(al[c])
            a1.bar(pos, mean[c], width=0.7, color=COMP_COL[c], edgecolor="k", lw=0.8,
                   yerr=s, capsize=4, error_kw=dict(lw=1.2, ecolor="k"))
            a1.plot([pos] * len(al[c]), al[c], "o", color="k", ms=3.5, alpha=0.7, zorder=3)
            tops.append((pos, mean[c] + s))
            labels.append(COMP_LABEL[c].replace("Co", "\nCo", 1))
            xs.append(pos)
            pos += 1

        shown = list(pure.values()) + ([cc_mean] if has_cc else [])
        top = max(shown + [v for c in COMPS for v in al[c]]) * 1.10
        for (x, y), v in zip(tops, shown + [mean[c] for c in COMPS]):
            a1.text(x, y + 0.012 * top, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
        a1.set_xticks(xs)
        a1.set_xticklabels(labels, fontsize=9)
        a1.set_ylim(0, top)
        a1.set_ylabel(GAMMA_LABEL)
        a1.set_title("(a) Pure metals, Co–Cr mean, and alloys")

        # (b) gamma against Ni content with the rule of mixtures
        if len(pure) == 3:
            cc = (pure["Co"] + pure["Cr"]) / 2                      # Co = Cr in every alloy
            a2.plot([0, 100], [cc, pure["Ni"]], "-", color="#555", lw=2)
            xm = [0] + [NI_PCT[c] for c in COMPS]
            ym = [cc] + [(NI_PCT[c] / 100) * pure["Ni"] + (1 - NI_PCT[c] / 100) * cc for c in COMPS]
            a2.plot(xm, ym, ls="", label=r"rule of mixtures: $\sum x_i\gamma_i$ (Ni, Co, Cr; $x_{Co}=x_{Cr}$)", **MIX)
            for name, x in (("Co", 0), ("Cr", 0), ("Ni", 100)):
                a2.plot(x, pure[name], "o", color=PURE_COL[name], ms=9)
            a2.text(2, pure["Co"], "pure Co", va="center", fontsize=9)
            a2.text(2, pure["Cr"], "pure Cr", va="center", fontsize=9)
            a2.text(98, pure["Ni"] + 25, "pure Ni", ha="right", fontsize=9)
            print(f"{'alloy':14s}{'simulated':>10s}{'mixture':>9s}{'difference':>11s}")
            for c, m in zip(COMPS, ym[1:]):
                print(f"{COMP_LABEL[c]:14s}{mean[c]:10.1f}{m:9.1f}{mean[c] - m:+11.1f}")
        for c in COMPS:
            a2.errorbar(NI_PCT[c], mean[c], yerr=sd(al[c]), fmt="s", ms=9, color=COMP_COL[c], capsize=4,
                        label=COMP_LABEL[c] + " (simulated)")
        a2.set_xlabel("Ni content (at.%, with Co = Cr)")
        a2.set_ylabel(GAMMA_LABEL)
        a2.set_title("(b) Surface energy against Ni content")
        a2.set_ylim(1200, 2100)
        a2.legend(fontsize=8.5, loc="upper right")

        fig.suptitle(f"Effect of Ni content on surface energy: ({PL}) surface, {T} K", fontsize=13)
        fig.tight_layout()
        fig.savefig(figure_path(root, f"fig_mixing_{PL}_{T}K.png"), dpi=200)
    print(f"wrote figures/fig_mixing_{PL}_{T}K.png")


FIGURES = {"planes": planes, "paired": paired, "convergence": convergence, "mixing": mixing}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("figure", choices=list(FIGURES) + ["all"])
    ap.add_argument("--root", default=ROOT, help="repository root (default: the folder above src/)")
    args = ap.parse_args()
    for name in (FIGURES if args.figure == "all" else [args.figure]):
        FIGURES[name](args.root)


if __name__ == "__main__":
    main()
