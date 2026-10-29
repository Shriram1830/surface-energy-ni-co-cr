"""Build the cells and run the LAMMPS calculations. Start it from the repository root.

  python src/run.py prod --plane 111     27 production runs for one plane (3 compositions x 3 temperatures x 3 seeds);
                                         repeat for 100 and 110 to get all 81
  python src/run.py static               0 K runs on pure Ni: vacuum-width and slab-thickness scans  -> convergence/
  python src/run.py conv base|nz|xy      Ni33Co33Cr33, (111), 700 K, seed 1: production size, thickness scan,
                                         in-plane size scan                                          -> convergence/
  python src/run.py pure                 pure Ni, Co and Cr, (111), 500 K                            -> convergence/

A case whose <tag>_result.txt already exists is skipped, so an interrupted run can simply be started again.
--dry only prints the commands. Other options: --lmp lmp  --np 4 (MPI ranks)  --vac 20  --neq 5000  --nprod 20000
"""
import argparse
import os
import subprocess
import sys
import time

from build_cell import COMPOSITIONS, ORIENT, make_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PURE = {"ni": (1, 0, 0), "co": (0, 1, 0), "cr": (0, 0, 1)}
failed = []


def run_case(args, *, lmp_input, outdir, tag, cell, plane, frac, seed, n, T, vac):
    """Build the cell if it does not exist yet, then run one LAMMPS case."""
    if os.path.exists(os.path.join(ROOT, outdir, tag + "_result.txt")):
        print(f"[skip] {tag}")
        return
    cellfile = f"cells/{cell}.data"
    if not os.path.exists(os.path.join(ROOT, cellfile)):
        make_cell(plane, frac, seed, n, 3.50, os.path.join(ROOT, cellfile))
    cmd = (["mpiexec", "-np", str(args.np)] if args.np > 1 else []) + [
        args.lmp, "-in", f"lammps/{lmp_input}", "-log", "none",
        "-var", "T", str(T), "-var", "data", cellfile, "-var", "tag", f"{outdir}/{tag}",
        "-var", "vac", str(vac), "-var", "seed", str(1000 * seed + T),
        "-var", "neq", str(args.neq), "-var", "nprod", str(args.nprod),
        "-var", "etol", "1e-12" if T == 0 else "1e-10"]
    print("[run] " + " ".join(cmd))
    if args.dry:
        return
    t0 = time.time()
    logf = os.path.join(ROOT, outdir, tag + ".log")     # LAMMPS screen output and errors
    try:
        with open(logf, "w") as fo:
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=fo, stderr=subprocess.STDOUT)
        print(f"[done] {tag} in {(time.time() - t0) / 60:.1f} min")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[FAILED] {tag}: {e}")
        if os.path.exists(logf):
            tail = open(logf, errors="replace").read().splitlines()[-25:]
            print(f"---- last lines of {outdir}/{tag}.log ----\n" + "\n".join(tail) + "\n----")
        failed.append(tag)


def prod(args):
    p = args.plane
    n = ORIENT[p]["default"]
    for s in args.seeds:                          # seed outermost, so seed 1 is complete first
        for c in args.comps:
            for T in args.temps:                  # the same arrangement is reused at every T
                run_case(args, lmp_input="in.surface.lmp", outdir="results", tag=f"{c}_{p}_T{T}_s{s}",
                         cell=f"{c}_{p}_s{s}", plane=p, frac=COMPOSITIONS[c], seed=s, n=n, T=T, vac=f"{args.vac:g}")


def static(args):
    p = args.plane
    nx0, ny0, nz0 = ORIENT[p]["default"]
    for c in args.comps:
        for s in (args.seeds if c != "ni100" else [1]):      # pure Ni has no configurational scatter
            def case(tag, n, vac):
                run_case(args, lmp_input="in.surface.lmp", outdir="convergence", tag=tag,
                         cell=f"conv_{p}_{n[0]}-{n[1]}-{n[2]}_{c}_s{s}", plane=p, frac=COMPOSITIONS[c],
                         seed=s, n=n, T=0, vac=vac)
            if "vac" in args.which:
                for v in (10, 15, 20, 30, 40):
                    case(f"conv_vac_{v}_{c}_s{s}", (nx0, ny0, nz0), f"{v:g}")
            if "nz" in args.which:
                for nz in sorted({max(2, round(nz0 * f)) for f in (0.6, 0.8, 1, 1.2, 1.6)}):
                    case(f"conv_nz_{nz}_{c}_s{s}", (nx0, ny0, nz), f"{args.vac:g}")


def conv(args):
    nx0, ny0, nz0 = ORIENT["111"]["default"]
    name = lambda n: f"c700_ni33_111_{n[0]}x{n[1]}x{n[2]}_s1"

    def case(n):
        run_case(args, lmp_input="in.conv.lmp", outdir="convergence", tag=name(n), cell=name(n), plane="111",
                 frac=COMPOSITIONS["ni33"], seed=1, n=n, T=700, vac=f"{args.vac:g}")
    if args.what == "base":
        case((nx0, ny0, nz0))
    elif args.what == "nz":
        for nz in args.nzs:
            case((nx0, ny0, nz))
    else:
        for size in args.sizes:
            nx, ny = map(int, size.split("x"))
            case((nx, ny, nz0))


def pure(args):
    n = ORIENT["111"]["default"]
    for el in args.els:
        tag = f"p500_{el}_111_s1"
        run_case(args, lmp_input="in.conv.lmp", outdir="convergence", tag=tag, cell=tag, plane="111",
                 frac=PURE[el], seed=1, n=n, T=500, vac=f"{args.vac:g}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lmp", default="lmp", help="LAMMPS executable")
    common.add_argument("--np", type=int, default=1, help="MPI ranks")
    common.add_argument("--vac", type=float, default=20.0, help="vacuum thickness (Angstrom)")
    common.add_argument("--neq", type=int, default=5000, help="equilibration steps")
    common.add_argument("--nprod", type=int, default=20000, help="averaging steps")
    common.add_argument("--dry", action="store_true", help="print the commands only")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("prod", parents=[common], help="production runs")
    a.add_argument("--plane", default="111", choices=ORIENT)
    a.add_argument("--comps", nargs="+", default=["ni22", "ni33", "ni44"], choices=["ni22", "ni33", "ni44"])
    a.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a.add_argument("--temps", type=int, nargs="+", default=[300, 500, 700])
    a.set_defaults(func=prod)

    a = sub.add_parser("static", parents=[common], help="0 K vacuum and thickness scans")
    a.add_argument("--plane", default="111", choices=ORIENT)
    a.add_argument("--comps", nargs="+", default=["ni100"], choices=list(COMPOSITIONS))
    a.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a.add_argument("--which", nargs="+", default=["vac", "nz"], choices=["vac", "nz"])
    a.set_defaults(func=static)

    a = sub.add_parser("conv", parents=[common], help="convergence tests at 700 K")
    a.add_argument("what", choices=["base", "nz", "xy"])
    a.add_argument("--nzs", type=int, nargs="+", default=[3, 4, 6, 8], help="z-repeats for 'nz'")
    a.add_argument("--sizes", nargs="+", default=["6x4", "11x7", "14x9"], help="x-by-y repeats for 'xy'")
    a.set_defaults(func=conv)

    a = sub.add_parser("pure", parents=[common], help="pure Ni, Co, Cr at 500 K")
    a.add_argument("--els", nargs="+", default=["ni", "co", "cr"], choices=list(PURE))
    a.set_defaults(func=pure)

    args = ap.parse_args()
    for d in ("cells", "results", "convergence"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    args.func(args)
    if failed:
        print("\nFAILED cases:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
