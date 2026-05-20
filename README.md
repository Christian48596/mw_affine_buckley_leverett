# Affine-Constrained Multiwavelet/DG Buckley--Leverett Solver

This repository contains `mw_buckley_leverett.py`, a public and fully commented Python driver for a one-dimensional Buckley--Leverett waterflooding benchmark. The code implements an affine-constrained conservative modal multiwavelet/DG solver for saturation transport in porous media, with Berea-core parameters provided as the default case.

The solver is designed for reproducible manuscript figures and tables, but it can also be used as a standalone command-line tool for testing other rock/core parameters.

## What the code solves

The code solves the one-dimensional hyperbolic Buckley--Leverett equation

```text
dS/dt + dF(S)/dx = 0,
F(S) = (v/phi) f_w(S),
```

where `S` is the water saturation, `phi` is porosity, `v` is Darcy velocity, and `f_w(S)` is a Corey fractional-flow function.

The numerical state is not only a finite-volume cell average. Each cell stores a local modal coefficient vector

```text
S[c, k]

k = 0      cell mean mode
k >= 1    intra-cell detail / multiwavelet modes
```

The residual is assembled in conservative weak form using numerical interface fluxes. The left inflow boundary condition is imposed as a linear trace constraint and enforced by an affine projection. This keeps the boundary state consistent while preserving the conservative shock-capturing structure.

## Main features

- Conservative modal multiwavelet/DG discretization.
- Affine enforcement of the left inflow saturation constraint.
- Corey fractional-flow model.
- Berea-core default benchmark.
- Rusanov, Godunov-sampled, and central numerical flux options.
- TVB, bounds, flattening, or no limiter options.
- Independent Buckley--Leverett reference profile.
- Optional `pywaterflood` reference support.
- Automatic or fixed breakthrough-probe location.
- PNG/PDF figure generation.
- CSV/JSON validation summaries.
- MPI support for independent parameter sweeps.

## Repository contents

```text
mw_buckley_leverett.py   Main executable Python solver
requirements.txt        Python package requirements
LICENSE                 MIT license
README.md               Repository documentation
```

Generated output folders are not required in the repository and can be safely ignored by Git.

## Recommended Python version

Use Python 3.10 or newer. Python 3.11 is recommended for a clean modern environment.

The code itself uses standard scientific Python packages. MPI execution requires `mpi4py` plus a working MPI implementation such as Open MPI, MPICH, Intel MPI, or the MPI stack available on your HPC cluster.

## Installation

Create a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On an HPC cluster, load the system MPI module before installing or using `mpi4py`, for example:

```bash
module load openmpi
python -m pip install -r requirements.txt
```

The exact module name depends on the cluster.

## Dependencies

Core dependencies:

- `numpy`
- `matplotlib`

Optional but useful dependencies:

- `pywaterflood`: used when available to compute the independent Buckley--Leverett reference.
- `mpi4py`: required only for MPI parameter sweeps.

If `pywaterflood` is not available, the code automatically falls back to an internal tangent-construction reference for the Corey curve. A single simulation can run without MPI.

## Default case: Berea-core benchmark

The default parser settings reproduce the Berea-core waterflood case used in the manuscript:

```text
L   = 6 in
D   = 1.5 in
phi = 0.20
Swc = 0.10
Sor = 0.20
mu_w = 1 cP
mu_o = 4 cP
nw = no = 2
q = 1 mL/min
```

Dimensional values are converted internally to SI-compatible units where needed. The injection rate is entered in `mL/min`, while the solver time unit is days. Output plots also report time in minutes for readability.

## Run the default Berea case

This command generates the main breakthrough curve and saturation-profile figures for the Berea benchmark:

```bash
python mw_buckley_leverett.py \
  --ncells 256 \
  --p 2 \
  --flux rusanov \
  --limiter tvb \
  --cfl 0.20 \
  --t-end-pvi 1.50 \
  --probe-x 0.0762 \
  --plot \
  --outdir JCP_RESULTS/final_figures_rusanov_Nc256_p2
```

For the default Berea core, `--probe-x 0.0762` places the breakthrough probe at the midpoint of the core:

```text
x = L/2 = 0.0762 m = 7.62 cm
```

Expected outputs include:

```text
Figure1_Sw_vs_t_probe.png
Figure1_Sw_vs_t_probe.pdf
Figure2_profiles.png
Figure2_profiles.pdf
Sw_probe_time_fully_mw.txt
validation_metrics_fully_mw.csv
run_summary_fully_mw.json
```

## Run a Berea MPI sweep

The following command runs independent simulations over resolution, modal order, and numerical flux:

```bash
mpirun -np 8 python mw_buckley_leverett.py \
  --mpi-sweep \
  --ncells-list 64 128 256 512 \
  --p-list 1 2 3 4 \
  --flux-list rusanov godunov \
  --limiter tvb \
  --cfl 0.20 \
  --t-end-pvi 1.50 \
  --plot-sweep \
  --outdir JCP_RESULTS/full_sweep_p2_tables
```

MPI is used only to distribute independent parameter cases. A single Buckley--Leverett solve is not domain-decomposed.

Expected aggregate outputs include:

```text
sweep_summary.csv
sweep_summary.json
manuscript_plots/Figure4_resolution_study_errors.png
manuscript_plots/Figure4_resolution_study_errors.pdf
manuscript_plots/Figure4_resolution_study_runtime.png
manuscript_plots/Figure4_resolution_study_runtime.pdf
manuscript_plots/Figure6_constraint_diagnostics.png
manuscript_plots/Figure6_constraint_diagnostics.pdf
manuscript_plots/Table_flux_comparison.csv
manuscript_plots/Table_flux_comparison.tex
manuscript_plots/Table_modal_order_sensitivity.csv
manuscript_plots/Table_modal_order_sensitivity.tex
```

## General case: using another rock or core

The Berea preset is only a default template. To simulate another rock/core, keep the same script and override the physical parameters from the command line.

The most important physical inputs are:

```text
--L             core length [m]
--D             core diameter [m]
--phi           porosity [-]
--Swc           connate/irreducible water saturation [-]
--Sor           residual oil saturation [-]
--sw-init       initial water saturation [-]
--sw-inj        injected water saturation [-]
--mu-w          water viscosity [Pa s]
--mu-o          oil viscosity [Pa s]
--nw            Corey water exponent [-]
--no            Corey oil exponent [-]
--krw0          endpoint water relative permeability [-]
--kro0          endpoint oil relative permeability [-]
--q-mL-min      injection rate [mL/min]
```

Example for a different stone/core:

```bash
python mw_buckley_leverett.py \
  --L 0.30 \
  --D 0.025 \
  --phi 0.18 \
  --Swc 0.15 \
  --Sor 0.25 \
  --sw-init 0.15 \
  --sw-inj 0.75 \
  --mu-w 1.0e-3 \
  --mu-o 8.0e-3 \
  --nw 2.5 \
  --no 2.0 \
  --krw0 1.0 \
  --kro0 1.0 \
  --q-mL-min 0.5 \
  --ncells 256 \
  --p 2 \
  --flux rusanov \
  --limiter tvb \
  --cfl 0.20 \
  --t-end-pvi 1.50 \
  --probe-mode auto-shock \
  --plot \
  --outdir RESULTS/general_stone_single_run
```

For a new rock, `--probe-mode auto-shock` is often more convenient than setting `--probe-x` manually. The code scans the independent reference profile at `--probe-auto-pvi`, detects the largest saturation gradient, and places the breakthrough probe near the displacement front.

## General-rock MPI sweep

For a different rock/core, the same parameter overrides can be combined with MPI sweeps. This is the recommended command when testing sensitivity to resolution, modal order, and flux choice:

```bash
mpirun -np 8 python mw_buckley_leverett.py \
  --mpi-sweep \
  --L 0.30 \
  --D 0.025 \
  --phi 0.18 \
  --Swc 0.15 \
  --Sor 0.25 \
  --sw-init 0.15 \
  --sw-inj 0.75 \
  --mu-w 1.0e-3 \
  --mu-o 8.0e-3 \
  --nw 2.5 \
  --no 2.0 \
  --krw0 1.0 \
  --kro0 1.0 \
  --q-mL-min 0.5 \
  --ncells-list 64 128 256 512 \
  --p-list 1 2 3 4 \
  --flux-list rusanov godunov \
  --limiter tvb \
  --cfl 0.20 \
  --t-end-pvi 1.50 \
  --probe-mode auto-shock \
  --plot-sweep \
  --outdir RESULTS/general_stone_mpi_sweep
```

This writes one subfolder per independent case, plus a global sweep summary in the main output directory.

## Choosing the breakthrough probe

Two modes are available.

### Fixed probe

Use this when the physical location is known and the run must be exactly reproducible:

```bash
--probe-x 0.0762
```

This is recommended for final manuscript figures.

### Automatic shock probe

Use this when testing a new rock/core and the front location is not known in advance:

```bash
--probe-mode auto-shock
```

Optional controls:

```bash
--probe-auto-pvi 0.20
--probe-scan-points 4000
```

The automatic probe is selected from the independent reference solution, not from the multiwavelet solution.

## Important numerical options

### Number of cells

```bash
--ncells 256
```

or for a sweep:

```bash
--ncells-list 64 128 256 512
```

### Modal order

In this code, `p` is the number of local modal basis functions per cell. The polynomial degree is `p - 1`.

```bash
--p 2
```

or for a sweep:

```bash
--p-list 1 2 3 4
```

### Flux

Recommended robust choice:

```bash
--flux rusanov
```

Comparison option:

```bash
--flux godunov
```

Sweep option:

```bash
--flux-list rusanov godunov
```

### Limiter

Recommended shock-capturing choice:

```bash
--limiter tvb
```

Other available options:

```bash
--limiter bounds
--limiter flatten
--limiter none
```

### CFL number

The default recommended value for the manuscript runs is:

```bash
--cfl 0.20
```

## Output files

Single-run outputs may include:

```text
Sw_profile_fully_mw_pvi*.txt
Sw_probe_time_fully_mw.txt
validation_metrics_fully_mw.csv
run_summary_fully_mw.json
Figure1_Sw_vs_t_probe.png/pdf
Figure2_profiles.png/pdf
```

MPI/list-sweep outputs may include:

```text
sweep_summary.csv
sweep_summary.json
manuscript_plots/*.png
manuscript_plots/*.pdf
manuscript_plots/*.csv
manuscript_plots/*.tex
```

The JSON files store physical parameters, numerical settings, probe location, wall time, and diagnostic metadata. The CSV files are suitable for direct plotting or import into a manuscript workflow.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Citation

If this code is used in a publication, cite the associated Buckley--Leverett Berea-core manuscript and this repository.
