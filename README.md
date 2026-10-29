# Surface energies of disordered Ni-Co-Cr alloys

We calculated the (111), (100) and (110) surface energies of random fcc Ni-Co-Cr alloys with molecular dynamics (LAMMPS) and a MEAM potential. This is the Group 9 project for MM309N (Computational Methods for Materials) at IIT Indore, under Prof. K. V. Vamsi.

**This README only describes the code. The method, the convergence tests, the results and their discussion are in the report, [`report/Surface_Energy_Report.pdf`](report/Surface_Energy_Report.pdf). Please start there.**

The study covers three compositions (Ni22Co39Cr39, Ni33Co33Cr33, Ni44Co28Cr28), three temperatures (300, 500 and 700 K) and three random atomic arrangements for every case, which makes 81 production runs.

## Main results

- The order is gamma(111) < gamma(100) < gamma(110) in every case. Compared with (111), (100) is 15-17 % higher and (110) is 18-19 % higher.
- gamma falls by 3.6-5.5 % when the Ni content goes from 22 to 44 at.% (with Co = Cr). The fall is clear on (100) and (110) and less certain on (111).
- No temperature dependence shows up between 300 and 700 K (changes of at most about 8 mJ/m2). The quantity we calculate is an excess potential energy with no entropy term, so this says nothing about the surface free energy.
- The absolute values are low: 24 % below DFT for pure Ni(111), and 32-33 % below the alloy (111) values of Shuang et al. (36 % for pure Ni). The potential was not fitted to surface energies, so the trends are what to rely on, not the absolute numbers.

## How a surface energy is obtained

gamma = (E_slab - E_bulk) / 2A. E_bulk is the energy of the same atoms before the vacuum is added, so the part of the energy that depends on the arrangement cancels. For T > 0 each run does a 0 K relaxation, an NPT bulk run to get the mean box, an NVT bulk run (E_bulk), the insertion of 20 A of vacuum along z, and an NVT slab run (E_slab). Every MD stage has 10 ps of equilibration and 40 ps of averaging with a 2 fs time step. The same atomic arrangement is used at all three temperatures, and the uncertainty quoted for a case is the standard deviation over its three arrangements.

## Repository layout

```
report/      the report (PDF)
potential/   MEAM files and potential.lmp, which every LAMMPS input includes
lammps/      in.surface.lmp   one surface-energy run (0 K or finite temperature)
             in.conv.lmp      same protocol, also saves the full time series (convergence tests, pure elements)
             in.elastic.lmp   0 K lattice parameter and elastic constants
src/         build_cell.py    builds an oriented fcc cell with a random arrangement and checks the short-range order
             run.py           builds the cells and runs LAMMPS (prod, static, conv, pure)
             analysis.py      gamma for every run, convergence data, cell check, tables
             plots.py         the four figures of the report
data/        the numbers that the tables and figures are made from: runs.csv and summary.csv (the 81 production runs),
             convergence.csv and conv_base_series.csv (convergence tests and pure elements), cell_check.csv
tables/      the tables of the report as CSV files
figures/     the figures of the report
```

The raw simulation output goes to `results/`, `convergence/` and `cells/`. These folders are created when you run the simulations and are not tracked.

## Requirements

LAMMPS built with the MEAM package, and Python 3 with numpy and matplotlib (`pip install -r requirements.txt`). MPI is optional. All commands below are run from the repository root.

## Tables and figures from the stored data

`data/` holds the numbers that the tables and figures are made from, so this part needs no simulation:

```
python src/analysis.py tables
python src/plots.py all
```

One result of the report cannot be rebuilt from `data/`: the elastic constants of Ni (Table 1). The output file of that run is not stored here, and the script skips the table when the file is missing. The 0 K pure-Ni runs (vacuum width 10-40 A, 3-8 repeats in z) are in `data/convergence.csv` with their energies only, without box dimensions.

## Rerunning the simulations

```
python src/run.py prod --plane 111        # also 100 and 110, 27 runs per plane
python src/run.py static                  # 0 K runs on pure Ni (vacuum width, slab thickness)
python src/run.py conv base               # convergence cases at 700 K
python src/run.py conv nz
python src/run.py conv xy
python src/run.py pure                    # pure Ni, Co and Cr at 500 K

python src/analysis.py runs               # results/      -> data/runs.csv, data/summary.csv
python src/analysis.py conv               # convergence/  -> data/convergence.csv, data/conv_base_series.csv
python src/analysis.py cells              # data/cell_check.csv
python src/analysis.py tables
python src/plots.py all
```

`run.py` calls LAMMPS as `lmp`. Use `--lmp` for a different executable name and `--np` for MPI ranks, and `--dry` to print the commands without running them. Finished cases are skipped, so an interrupted run can be restarted.

For the elastic constants of pure Ni (Table 1 of the report):

```
mkdir -p cells
python src/build_cell.py --orient 100 --n 6 6 6 --comp 1 0 0 --out cells/ni_cubic.data
lmp -in lammps/in.elastic.lmp -var data cells/ni_cubic.data -var ncell 6 -var tag data/elastic_ni
```

## Where the numbers of the report come from

| In the report | In the repository |
|---|---|
| Table 1, elastic constants of Ni | calculated with `lammps/in.elastic.lmp` (command below); output not stored |
| Table 2, supercells | `tables/supercells.csv` |
| Table 4, surface energies | `tables/results_all.csv`, `data/summary.csv` |
| Table 5, anisotropy | `tables/anisotropy.csv` (rows marked mean) |
| Table 6, composition effect | `tables/composition_effect.csv` (300 K rows) |
| Table 7, rule of mixtures | `tables/mixing_111_500K.csv` |
| Table 8, comparison with the literature | alloy rows from `data/summary.csv` (300 K); the 0 K pure-Ni value from `data/convergence.csv` (`conv_vac_*`, `conv_nz_*`), see `tables/literature_111.csv` |
| Section 5.3, paired temperature test | `tables/paired_temperature.csv` |
| Fig. 1 to 4 | `figures/fig_convergence.png`, `fig_planes.png`, `fig_mixing_111_500K.png`, `fig_temperature_paired.png` |

## Things worth knowing when you rerun

- The random arrangements come from numpy's `default_rng` with seeds 1, 2 and 3, so they are rebuilt identically. `data/cell_check.csv` lists the exact atom counts of every cell and the largest first-shell Warren-Cowley parameter (below 0.1 everywhere), which is a quick way to check that a rebuilt cell is the same.
- The velocity seed of a run is 1000 x (arrangement seed) + T.
- Pure Ni, Co and Cr are single runs, since a perfect crystal has no arrangement scatter. Fcc Co and Cr are hypothetical phases, so those values are model quantities (see the report).
- In the pure Cr run the cell did not stay cubic: `data/convergence.csv` gives box lengths of 20.26, 23.52 and 30.83 A, against 20.09, 21.74 and 30.75 A for pure Ni. The Cr value is therefore not the energy of an undistorted fcc surface.

## Potential

The MEAM potential is from H. Sharifi and C. Wick, Comput. Mater. Sci. 248, 113595 (2025), taken from the NIST Interatomic Potentials Repository (entry `2025--Sharifi-H--Cr-Ni-Co--LAMMPS--ipr1`). The two files in `potential/` belong to their authors and are included for convenience. Please cite the paper if you use them.

## Authors and licence

Shriram Atul Naik (111), Kethavath Mounika (100) and Lakshya Rishi (110). The code is released under the MIT licence (see `LICENSE`); this does not cover the potential files.
