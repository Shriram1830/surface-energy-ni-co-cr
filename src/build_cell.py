"""Build a periodic, orthogonal FCC cell with a random Ni/Co/Cr arrangement for LAMMPS.

z is the surface normal. The cell is shifted in z so that the periodic z-boundary lies
BETWEEN two atomic layers; adding vacuum later (in LAMMPS) then cuts the crystal cleanly.
Atom types: 1 = Ni, 2 = Co, 3 = Cr.   --comp takes the fractions Ni Co Cr.
Atoms are placed randomly with EXACT counts (largest-remainder rounding).
A first-neighbour Warren-Cowley check is printed: for a random alloy |alpha| should be ~0.

Usage:
  python src/build_cell.py --orient 111 --comp 0.33 0.33 0.34 --seed 1 --out cell.data
  python src/build_cell.py --orient 111 --n 11 6 5 --comp 1 0 0 --out ni.data     (pure Ni)
"""
import argparse
import numpy as np

MASS = {1: 58.6934, 2: 58.9332, 3: 51.9961}  # Ni, Co, Cr

# Ni, Co, Cr fractions. ni22/ni33/ni44 are the Ni sweep with Co = Cr; ni100 is pure Ni (0 K reference runs).
COMPOSITIONS = {"ni22": (0.22, 0.39, 0.39), "ni33": (1 / 3, 1 / 3, 1 / 3),
                "ni44": (0.44, 0.28, 0.28), "ni100": (1.0, 0.0, 0.0)}

# v: crystal directions along the new x, y, z axes. per: periodic length of one repeat along
# x, y, z in units of the lattice parameter a. d: layer spacing along z in units of a.
# default: repeats (nx, ny, nz) -> about 1200 atoms, ~20 x 20 x 30 Angstrom.
ORIENT = {
    "111": dict(v=[(1, -1, 0), (1, 1, -2), (1, 1, 1)],
                per=(1 / np.sqrt(2), np.sqrt(6) / 2, np.sqrt(3)),
                d=1 / np.sqrt(3), default=(8, 5, 5)),
    "100": dict(v=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                per=(1, 1, 1), d=0.5, default=(6, 6, 8)),
    "110": dict(v=[(0, 0, 1), (1, -1, 0), (1, 1, 0)],
                per=(1, 1 / np.sqrt(2), 1 / np.sqrt(2)),
                d=1 / (2 * np.sqrt(2)), default=(6, 8, 12)),
}
BASIS = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


def exact_counts(n, frac):
    """Integer counts that sum to n and are closest to frac (largest remainder)."""
    frac = np.array(frac, dtype=float) / np.sum(frac)
    raw = frac * n
    cnt = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - cnt))[: n - cnt.sum()]:
        cnt[i] += 1
    return cnt


def build(orient, a, nx, ny, nz):
    """Positions (Angstrom) and box lengths of the oriented FCC block."""
    o = ORIENT[orient]
    R = np.array([np.array(v) / np.linalg.norm(v) for v in o["v"]])  # rows = new axes
    L = np.array([nx, ny, nz]) * np.array(o["per"]) * a
    M = int(np.ceil(np.linalg.norm(L) / a)) + 2
    g = np.arange(-M, M + 1)
    cells = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    p = (cells[:, None, :] + BASIS[None, :, :]).reshape(-1, 3) * a
    r = p @ R.T
    eps = 1e-3
    keep = np.all((r > -eps) & (r < L - eps), axis=1)
    r = r[keep]
    expected = int(round(4 * np.prod(L) / a ** 3))
    assert len(r) == expected, f"atom count {len(r)} != expected {expected}"
    r[:, 2] = (r[:, 2] + o["d"] * a / 2) % L[2]  # put the cut plane between two layers
    r[:, :2] = np.where(r[:, :2] < 0, 0.0, r[:, :2])
    return r, L


def warren_cowley(pos, L, types, a):
    """First-shell Warren-Cowley matrix alpha_ij = 1 - P(j|i)/c_j (0 for a random alloy)."""
    N, nt = len(pos), 3
    onehot = np.eye(nt)[types - 1]
    nn = np.zeros((N, nt))
    for i0 in range(0, N, 400):
        d = pos[i0:i0 + 400, None, :] - pos[None, :, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(axis=2))
        nn[i0:i0 + 400] = ((r > 0.1) & (r < 0.854 * a)) @ onehot  # 0.854 a: between shells 1 and 2
    if not np.all(nn.sum(axis=1) == 12):
        return None
    c = onehot.mean(axis=0)
    alpha = np.full((nt, nt), np.nan)
    for i in range(nt):
        if c[i] == 0:
            continue
        Pi = nn[types == i + 1].sum(axis=0) / (12.0 * (types == i + 1).sum())
        for j in range(nt):
            if c[j] > 0:
                alpha[i, j] = 1 - Pi[j] / c[j]
    return alpha


def make_cell(orient, comp, seed, n=None, a=3.50, out="cell.data", quiet=False):
    """Build one cell, write the LAMMPS data file, return a dict with the details."""
    nx, ny, nz = n or ORIENT[orient]["default"]
    pos, L = build(orient, a, nx, ny, nz)
    N = len(pos)
    cnt = exact_counts(N, comp)
    types = np.repeat([1, 2, 3], cnt)
    np.random.default_rng(seed).shuffle(types)

    with open(out, "w") as f:
        f.write(f"Ni-Co-Cr random alloy, orient {orient}, seed {seed}\n\n")
        f.write(f"{N} atoms\n3 atom types\n\n")
        f.write(f"0 {L[0]:.6f} xlo xhi\n0 {L[1]:.6f} ylo yhi\n0 {L[2]:.6f} zlo zhi\n\n")
        f.write("Masses\n\n" + "\n".join(f"{t} {m}" for t, m in MASS.items()) + "\n\n")
        f.write("Atoms # atomic\n\n")
        for i, (t, x) in enumerate(zip(types, pos), 1):
            f.write(f"{i} {t} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")

    alpha = warren_cowley(pos, L, types, a)
    amax = None if alpha is None or (cnt > 0).sum() < 2 else float(np.nanmax(np.abs(alpha)))
    if not quiet:
        print(f"wrote {out}: {N} atoms, box {L.round(2)} A, repeats {nx},{ny},{nz}, "
              f"counts Ni/Co/Cr = {cnt.tolist()}, seed {seed}, "
              f"max|Warren-Cowley alpha| = {'n/a' if amax is None else f'{amax:.3f}'}")
    return dict(N=N, L=L, counts=cnt, alpha_max=amax)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--orient", default="111", choices=ORIENT)
    ap.add_argument("--comp", type=float, nargs=3, default=[1 / 3, 1 / 3, 1 / 3], help="Ni Co Cr fractions")
    ap.add_argument("--a", type=float, default=3.50, help="initial lattice parameter (A); LAMMPS relaxes it")
    ap.add_argument("--n", type=int, nargs=3, default=None, help="repeats nx ny nz (default per orientation)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="cell.data")
    args = ap.parse_args()
    make_cell(args.orient, args.comp, args.seed, args.n, args.a, args.out)


if __name__ == "__main__":
    main()
