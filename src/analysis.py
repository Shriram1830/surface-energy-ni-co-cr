"""Analysis of the LAMMPS output. Start it from the repository root.

  python src/analysis.py runs     results/ -> data/runs.csv, data/summary.csv
  python src/analysis.py conv     convergence/ -> data/convergence.csv, data/conv_base_series.csv
  python src/analysis.py cells    rebuild the 27 production cells and check composition and short-range order
                                  -> data/cell_check.csv
  python src/analysis.py tables   data/*.csv -> tables/*.csv  (needs no raw LAMMPS output)

Per run: gamma = (<PE_slab> - <PE_bulk>) / (2 A). For T > 0 the averages are recomputed from the saved
time series (<tag>_bulk_pe.dat, <tag>_slab_pe.dat) and the statistical error is the standard error of
10 block averages (propagated from bulk and slab). For T = 0 gamma comes straight from the result file
(no error: static).
Per case (same tag apart from the trailing _s<seed>): mean and SD over the configurations. The SD over
configurations is the headline uncertainty; the block error is only the MD sampling noise of one run.
The loaders at the top (load_production, load_conv, load_series) are shared with plots.py.
"""
import argparse
import csv
import glob
import os
import re
import warnings

import numpy as np

from build_cell import COMPOSITIONS, ORIENT, make_cell

EV_A2_TO_MJ_M2 = 16021.766
DT = 0.002                                   # ps per MD step
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLANES = ["111", "100", "110"]
COMPS = ["ni22", "ni33", "ni44"]
TEMPS = [300, 500, 700]
NI_PCT = {"ni22": 22.0, "ni33": 100 / 3, "ni44": 44.0}
COMP_LABEL = {"ni22": "Ni22Co39Cr39", "ni33": "Ni33Co33Cr33", "ni44": "Ni44Co28Cr28"}
PROD_RUN = re.compile(r"(ni22|ni33|ni44)_(111|100|110)_T(300|500|700)_s(\d+)")
BASE_TAG = "c700_ni33_111_8x5x5_s1"          # production-size case of the convergence study

# Reference values quoted in the report (Table 8) and experimental Ni data (Table 1)
SHUANG_111 = {"ni22": 2573, "ni33": 2508, "ni44": 2475, "ni100": 2269}   # Shuang et al., Int. J. Plast. 177 (2024)
DFT_NI_111 = 1920                                                         # Tran et al., Sci. Data 3 (2016)
EXP_NI = [("a (A)", 3.52), ("C11 (GPa)", 261), ("C12 (GPa)", 151), ("C44 (GPa)", 132), ("B (GPa)", 188)]


# ------------------------------------------------------------------ basic helpers
def block_mean_err(x, nb=10):
    n = len(x) // nb
    blocks = x[: n * nb].reshape(nb, n).mean(axis=1)
    return x[: n * nb].mean(), blocks.std(ddof=1) / np.sqrt(nb)


def analyse(prefix):
    """gamma and its block error for the run whose files start with `prefix` (path without _result.txt)."""
    T, N, Lx, Ly, Lz, Eb, Es, g_lmp = np.atleast_1d(np.loadtxt(prefix + "_result.txt"))
    A = Lx * Ly
    fb, fs = prefix + "_bulk_pe.dat", prefix + "_slab_pe.dat"
    if T > 0 and os.path.exists(fb) and os.path.exists(fs):
        pb = np.loadtxt(fb, comments="#")[:, 1]
        ps = np.loadtxt(fs, comments="#")[:, 1]
        Eb, eb = block_mean_err(pb)
        Es, es = block_mean_err(ps)
        gamma = (Es - Eb) / (2 * A) * EV_A2_TO_MJ_M2
        err = np.hypot(eb, es) / (2 * A) * EV_A2_TO_MJ_M2
    else:
        gamma, err = (Es - Eb) / (2 * A) * EV_A2_TO_MJ_M2, float("nan")
    return dict(T=int(T), N=int(N), A=A, gamma=gamma, err=err, gamma_lammps=g_lmp)


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


# ------------------------------------------------------------------ loaders (read data/*.csv)
def load_production(root=ROOT):
    """-> {(comp, plane, T): (gammas, block_errors)}, one entry per random configuration (seed order 1, 2, 3)."""
    path = os.path.join(root, "data", "runs.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run 'python src/analysis.py runs' first")
    raw = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m = PROD_RUN.fullmatch(row["run"])
            if m:
                key = (m.group(1), m.group(2), int(m.group(3)))
                raw.setdefault(key, {})[int(m.group(4))] = (float(row["gamma_mJ_m2"]), float(row["block_err_mJ_m2"]))
    return {k: (np.array([v[s][0] for s in sorted(v)]), np.array([v[s][1] for s in sorted(v)])) for k, v in raw.items()}


def load_conv(root=ROOT):
    """-> {tag: dict(T, N, A, gamma, err, L)} for the convergence, 0 K and pure-element runs ({} if the file is missing)."""
    path = os.path.join(root, "data", "convergence.csv")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                out[row["run"]] = dict(T=int(row["T"]), N=int(row["N"]), A=float(row["area_A2"]),
                                       gamma=float(row["gamma_mJ_m2"]), err=float(row["block_err_mJ_m2"]),
                                       L=(float(row["Lx_A"]), float(row["Ly_A"]), float(row["Lz_A"])))
    return out


def load_series(root=ROOT):
    """Time series of the production-size convergence run as {column: array}, or None if the file is missing."""
    path = os.path.join(root, "data", "conv_base_series.csv")
    if not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=",", names=True)
    return {k: d[k] for k in d.dtype.names}


# ------------------------------------------------------------------ runs -> data/runs.csv, summary.csv
def cmd_runs(root):
    res = os.path.join(root, "results")
    runs = {}
    for f in sorted(glob.glob(os.path.join(res, "*_result.txt"))):
        tag = os.path.basename(f)[: -len("_result.txt")]
        if PROD_RUN.fullmatch(tag):                 # production runs only
            runs[tag] = analyse(f[: -len("_result.txt")])
    if not runs:
        raise SystemExit(f"no production *_result.txt files in {res}")

    print(f"{'run':38s} {'T':>4s} {'N':>5s} {'gamma':>9s} {'block err':>10s}")
    for tag in sorted(runs, key=natural):
        r = runs[tag]
        print(f"{tag:38s} {r['T']:4d} {r['N']:5d} {r['gamma']:9.1f} {r['err']:10.2f}")

    groups = {}
    for tag, r in runs.items():
        groups.setdefault(re.sub(r"_s\d+$", "", tag), []).append(r["gamma"])
    print(f"\n{'case':34s} {'n':>2s} {'mean':>9s} {'SD over configs':>16s}   (mJ/m^2)")
    summary = []
    for key in sorted(groups, key=natural):
        g = np.array(groups[key])
        s = g.std(ddof=1) if len(g) > 1 else float("nan")
        summary.append((key, len(g), g.mean(), s))
        print(f"{key:34s} {len(g):2d} {g.mean():9.1f} {s:16.1f}")

    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    with open(os.path.join(root, "data", "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "T", "N", "gamma_mJ_m2", "block_err_mJ_m2"])
        for tag in sorted(runs, key=natural):
            r = runs[tag]
            w.writerow([tag, r["T"], r["N"], f"{r['gamma']:.3f}", f"{r['err']:.3f}"])
    with open(os.path.join(root, "data", "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n_configs", "mean_mJ_m2", "SD_over_configs_mJ_m2"])
        for key, n, m, s in summary:
            w.writerow([key, n, f"{m:.3f}", f"{s:.3f}"])
    print("\nwrote data/runs.csv and data/summary.csv")


# ------------------------------------------------------------------ convergence -> data/convergence.csv, series
def cmd_conv(root):
    d = os.path.join(root, "convergence")
    files = set(glob.glob(os.path.join(d, "*_result.txt")))
    files |= set(glob.glob(os.path.join(root, "results", "conv_*_result.txt")))     # 0 K runs from older layouts
    if not files:
        raise SystemExit(f"no *_result.txt files in {d}")
    per = ORIENT["111"]["per"]
    rows = []
    print(f"{'case':28s} {'atoms':>5s} {'gamma':>8s} {'+/-':>5s}   a_x    a_y    a_z   (from the NPT-mean box, A)")
    for f in sorted(files, key=lambda p: natural(os.path.basename(p))):
        prefix = f[: -len("_result.txt")]
        tag = os.path.basename(prefix)
        r = analyse(prefix)
        Lx, Ly, Lz = np.loadtxt(f)[2:5]
        m = re.search(r"_(\d+)x(\d+)x(\d+)_s", tag)
        n = tuple(map(int, m.groups())) if m else ORIENT["111"]["default"]
        a = [L / (k * p) for L, k, p in zip((Lx, Ly, Lz), n, per)]
        print(f"{tag:28s} {r['N']:5d} {r['gamma']:8.1f} {r['err']:5.1f}   {a[0]:.4f} {a[1]:.4f} {a[2]:.4f}")
        rows.append([tag, r["T"], r["N"], f"{r['A']:.3f}", f"{r['gamma']:.3f}", f"{r['err']:.3f}",
                     f"{Lx:.4f}", f"{Ly:.4f}", f"{Lz:.4f}"])
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    with open(os.path.join(root, "data", "convergence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "T", "N", "area_A2", "gamma_mJ_m2", "block_err_mJ_m2", "Lx_A", "Ly_A", "Lz_A"])
        w.writerows(rows)
    print("\nwrote data/convergence.csv")

    # time series of the production-size case: drift check (printed) and the input of Fig. 1 a-d (saved)
    base = os.path.join(d, BASE_TAG)
    if not os.path.exists(base + "_npt_box.dat"):
        print(f"{BASE_TAG}: time series not found, data/conv_base_series.csv not written")
        return
    box = np.loadtxt(base + "_npt_box.dat", comments="#")
    pe = {k: np.loadtxt(f"{base}_{k}_full.dat", comments="#") for k in ("bulk", "slab")}
    n = min(len(box), len(pe["bulk"]), len(pe["slab"]))
    series = np.column_stack([(box[:n, 0] - box[0, 0]) * DT, box[:n, 1:4], pe["bulk"][:n, 1], pe["slab"][:n, 1]])
    np.savetxt(os.path.join(root, "data", "conv_base_series.csv"), series, delimiter=",", fmt="%.6f", comments="",
               header="t_ps,Lx_A,Ly_A,Lz_A,PE_bulk_eV,PE_slab_eV")
    print("wrote data/conv_base_series.csv")

    def split(x, neq_rows=500):
        p = x[neq_rows:]
        return x[:neq_rows], p[: len(p) // 2], p[len(p) // 2:]

    def fmt(parts, digits):
        return " | ".join(f"{s.mean():.{digits}f}" + (f" +/- {block_mean_err(s, 5)[1]:.{digits}f}" if k else "")
                          for k, s in enumerate(parts))
    print("\nproduction-size case, 700 K: mean (+/- block error) in equilibration | 1st half | 2nd half of the averaging window")
    for i, c in enumerate("xyz"):
        lat = box[:, 1 + i] / (ORIENT["111"]["default"][i] * per[i])
        print(f"  NPT a_{c}:  " + fmt(split(lat), 4))
    N = analyse(base)["N"]
    for kind in ("bulk", "slab"):
        print(f"  NVT {kind} PE/atom (eV):  " + fmt(split(pe[kind][:, 1] / N), 5))


# ------------------------------------------------------------------ cells -> data/cell_check.csv
def cmd_cells(root):
    rows = []
    print(f"{'comp':6s} {'plane':5s} {'seed':>4s} {'atoms':>5s}   Ni/Co/Cr        max|alpha|")
    for c in COMPS:
        for p in PLANES:
            for s in (1, 2, 3):
                info = make_cell(p, COMPOSITIONS[c], s, out=os.devnull, quiet=True)
                am = "" if info["alpha_max"] is None else f"{info['alpha_max']:.3f}"
                rows.append([c, p, s, info["N"], *info["counts"].tolist(), am])
                print(f"{c:6s} {p:5s} {s:4d} {info['N']:5d}   {'/'.join(map(str, info['counts'])):14s}  {am}")
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    with open(os.path.join(root, "data", "cell_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comp", "plane", "seed", "N", "n_Ni", "n_Co", "n_Cr", "max_abs_warren_cowley_alpha"])
        w.writerows(rows)
    print("\nwrote data/cell_check.csv")


# ------------------------------------------------------------------ tables -> tables/*.csv
def cmd_tables(root):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    prod = load_production(root)
    conv = load_conv(root)
    get = lambda c, p, T: prod[(c, p, T)][0]
    out = os.path.join(root, "tables")
    os.makedirs(out, exist_ok=True)

    def write(name, header, rows):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    # Table 4: every case
    rows = [(p, COMP_LABEL[c], T, len(get(c, p, T)), f"{get(c, p, T).mean():.1f}", f"{sd(get(c, p, T)):.1f}",
             f"{np.nanmean(prod[(c, p, T)][1]):.1f}") for p in PLANES for c in COMPS for T in TEMPS]
    write("results_all.csv", ["plane", "alloy", "T_K", "n_configs", "mean_mJ_m2", "SD_configs_mJ_m2",
                              "mean_block_err_mJ_m2"], rows)

    # Section 5.3: paired temperature change. |t| > 4.3 is significant at 95 % for 3 configurations (2 dof).
    rows = []
    print("Paired temperature change (same arrangement), mJ/m^2:")
    for p in PLANES:
        for c in COMPS:
            for T in (500, 700):
                dlt = get(c, p, T) - get(c, p, 300)
                se = sd(dlt) / np.sqrt(len(dlt))
                t = dlt.mean() / se if se > 0 else float("nan")
                rows.append((p, COMP_LABEL[c], f"300->{T}", f"{dlt.mean():+.1f}", f"{sd(dlt):.1f}", f"{t:.1f}",
                             "yes" if abs(t) > 4.3 else "no"))
                print(f"  ({p}) {COMP_LABEL[c]:13s} 300->{T}: {dlt.mean():+6.1f} +/- {sd(dlt):5.1f}   t = {t:5.1f}  "
                      f"{'significant' if abs(t) > 4.3 else 'not significant'}")
    write("paired_temperature.csv", ["plane", "alloy", "change", "mean_mJ_m2", "SD_mJ_m2", "t",
                                     "differs_from_zero_95pct"], rows)

    # Table 6: composition effect (difference of means, combined standard error)
    rows = []
    print("\nComposition effect (difference of means, uncertainty = combined standard error), mJ/m^2:")
    for p in PLANES:
        for T in TEMPS:
            for c in ("ni33", "ni44"):
                a, b = get(c, p, T), get("ni22", p, T)
                diff = a.mean() - b.mean()
                se = np.sqrt(sd(a) ** 2 / len(a) + sd(b) ** 2 / len(b))
                rows.append((p, T, f"{COMP_LABEL[c]} - {COMP_LABEL['ni22']}", f"{diff:+.1f}", f"{se:.1f}", f"{diff / se:.1f}"))
                print(f"  ({p}) {T} K  {c} - ni22: {diff:+6.1f} +/- {se:4.1f}   ({abs(diff) / se:.1f} standard errors)")
    write("composition_effect.csv", ["plane", "T_K", "difference", "mean_mJ_m2", "std_error_mJ_m2", "diff_over_se"], rows)

    # Table 5: anisotropy. The 'mean' rows average the three temperatures; the last row is the bond count.
    rows = []
    print("\nAnisotropy (ratio of mean surface energies):")
    for c in COMPS:
        r1s, r2s = [], []
        for T in TEMPS:
            r1 = get(c, "100", T).mean() / get(c, "111", T).mean()
            r2 = get(c, "110", T).mean() / get(c, "111", T).mean()
            r1s.append(r1)
            r2s.append(r2)
            rows.append((COMP_LABEL[c], T, f"{r1:.3f}", f"{r2:.3f}"))
        rows.append((COMP_LABEL[c], "mean", f"{np.mean(r1s):.3f}", f"{np.mean(r2s):.3f}"))
        print(f"  {COMP_LABEL[c]:13s} (100)/(111) = {np.mean(r1s):.3f}   (110)/(111) = {np.mean(r2s):.3f}   (mean of 300, 500, 700 K)")
    rows.append(("broken nearest-neighbour bonds", "-", f"{4 / 3:.3f}", f"{5 / 3:.3f}"))
    write("anisotropy.csv", ["alloy", "T_K", "gamma100_over_gamma111", "gamma110_over_gamma111"], rows)

    # Table 2: supercells before vacuum insertion (starting lattice parameter 3.50 A)
    a0, rows = 3.50, []
    for p in PLANES:
        o = ORIENT[p]
        n = o["default"]
        L = np.array(n) * np.array(o["per"]) * a0
        layers = int(round(n[2] * o["per"][2] / o["d"]))
        rows.append((f"({p})", f"{n[0]}x{n[1]}x{n[2]}", int(round(4 * np.prod(L) / a0 ** 3)), layers,
                     f"{L[0] * L[1]:.0f}", f"{(layers - 1) * o['d'] * a0 + 1e-9:.1f}"))
    write("supercells.csv", ["surface", "repeats", "atoms", "layers", "area_A2", "thickness_A"], rows)

    # Table 7: rule of mixtures for (111) at 500 K (needs the pure-element runs)
    pure = {el: conv[f"p500_{el}_111_s1"]["gamma"] for el in ("ni", "co", "cr") if f"p500_{el}_111_s1" in conv}
    if len(pure) == 3:
        cc = (pure["co"] + pure["cr"]) / 2
        rows = []
        print(f"\nRule of mixtures, (111), 500 K (pure Ni {pure['ni']:.0f}, Co {pure['co']:.0f}, Cr {pure['cr']:.0f}; Co-Cr mean {cc:.0f}):")
        for c in COMPS:
            x = NI_PCT[c] / 100
            mix, sim = x * pure["ni"] + (1 - x) * cc, get(c, "111", 500).mean()
            rows.append((COMP_LABEL[c], f"{mix:.0f}", f"{sim:.0f}", f"{sim - mix:+.0f}", f"{100 * (sim - mix) / mix:+.1f}"))
            print(f"  {COMP_LABEL[c]:13s} mixture {mix:7.1f}   simulated {sim:7.1f}   difference {sim - mix:+6.1f}")
        write("mixing_111_500K.csv", ["alloy", "mixture_mJ_m2", "simulated_mJ_m2", "difference_mJ_m2", "difference_pct"], rows)
    else:
        print("\nmixing_111_500K.csv skipped: pure-element runs not in data/convergence.csv")

    # Table 8: (111) against the literature (alloys at 300 K, pure Ni at 0 K)
    ni0 = conv.get("conv_vac_20_ni100_s1") or conv.get("conv_nz_5_ni100_s1")
    if ni0:
        rows = []
        for c in COMPS:
            v = get(c, "111", 300).mean()
            rows.append((COMP_LABEL[c], f"{v:.1f}", SHUANG_111[c], "Shuang et al.", f"{100 * (v - SHUANG_111[c]) / SHUANG_111[c]:.1f}"))
        v = ni0["gamma"]
        rows.append(("Pure Ni (0 K)", f"{v:.1f}", SHUANG_111["ni100"], "Shuang et al.", f"{100 * (v - SHUANG_111['ni100']) / SHUANG_111['ni100']:.1f}"))
        rows.append(("Pure Ni (0 K)", f"{v:.1f}", DFT_NI_111, "DFT, Tran et al.", f"{100 * (v - DFT_NI_111) / DFT_NI_111:.1f}"))
        write("literature_111.csv", ["system", "present_mJ_m2", "reference_mJ_m2", "reference", "deviation_pct"], rows)
    else:
        print("literature_111.csv skipped: 0 K pure-Ni run not in data/convergence.csv")

    # Table 1: elastic constants of pure Ni (written by lammps/in.elastic.lmp)
    el = os.path.join(root, "data", "elastic_ni_result.txt")
    if os.path.exists(el):
        a, e0, c11, c12, c44, b, g, ey = np.loadtxt(el)
        meam = [a, c11, c12, c44, b]
        write("elastic_ni.csv", ["property", "MEAM", "experiment"],
              [(name, f"{m:.3f}" if i == 0 else f"{m:.0f}", ex) for i, ((name, ex), m) in enumerate(zip(EXP_NI, meam))])
    else:
        print("elastic_ni.csv skipped: data/elastic_ni_result.txt not found")
    print(f"\nwrote the tables to {os.path.relpath(out, root)}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("step", choices=["runs", "conv", "cells", "tables"])
    ap.add_argument("--root", default=ROOT, help="repository root (default: the folder above src/)")
    args = ap.parse_args()
    {"runs": cmd_runs, "conv": cmd_conv, "cells": cmd_cells, "tables": cmd_tables}[args.step](args.root)


if __name__ == "__main__":
    main()
