# Fixed-Grid Affine-Constrained Multiwavelet Coefficient Solver for Buckley--Leverett Flow

This repository contains a public Python implementation of a fixed-grid conservative affine-constrained modal/multiwavelet coefficient method for one-dimensional Buckley--Leverett saturation transport.

The code evolves the saturation directly in a local orthonormal coefficient basis. The first local mode carries the conservative cell average, while the higher modes carry zero-mean intra-cell detail information. The nonlinear Buckley--Leverett flux is advanced through a conservative weak formulation with monotone numerical interface fluxes. The physical inflow condition is imposed as an affine trace constraint on the coefficient vector, and shock-induced oscillations are controlled by limiters acting on the modal detail coefficients.

The default case reproduces the Berea-core waterflood benchmark used in the manuscript. Other rocks or cores can be tested by overriding the physical and Corey fractional-flow parameters from the command line.

## Important terminology

This repository should be cited and described as a **fixed-grid affine-constrained modal/multiwavelet coefficient method**.

The method is **not presented as a standard discontinuous Galerkin package**. Its conservative weak residual is related to modal discontinuous formulations, but the implementation and manuscript organize the method around coefficient-space operations: mean/detail coefficients, affine inflow enforcement, detail-only boundary reprojection, conservative flux coupling, and detail limiting.

## Main file

```bash
mw_buckley_leverett.py
```

## What the code solves

The code solves the one-dimensional hyperbolic Buckley--Leverett saturation equation

```math
\frac{\partial S}{\partial t}
+
\frac{\partial \mathcal F(S)}{\partial x}
=0,
\qquad
\mathcal F(S)=\frac{v}{\phi} f_w(S),
```

where:

- `S` is the wetting-phase saturation,
- `v` is the Darcy velocity,
- `phi` is the porosity,
- `f_w(S)` is the Corey fractional-flow function.

The evolved unknown is a cell-local modal coefficient array:

```text
S[c, k]

k = 0      conservative cell-mean mode
k >= 1    zero-mean local detail/multiwavelet modes
```

The left boundary is treated as a physical inflow boundary. The imposed injected-water saturation is enforced as a linear trace constraint on the coefficient vector. The right boundary is treated as an outflow boundary.

## Repository contents

A typical repository layout is:

```text
.
├── mw_buckley_leverett.py
├── README.md
├── requirements.txt
└── LICENSE
```

The license is MIT.

## Python version

Recommended:

```text
Python 3.10 or newer
```

The code is pure Python and uses standard scientific Python packages. Python 3.11 is a good default choice for new environments.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

For MPI parameter sweeps, the system must also provide a working MPI installation, for example OpenMPI or MPICH. The Python package `mpi4py` is listed in `requirements.txt`, but it must be linked against a working MPI runtime.

## Dependencies

Core dependencies:

```text
numpy
matplotlib
```

Optional but recommended:

```text
pywaterflood
```

`pywaterflood` is used to generate an independent Buckley--Leverett reference solution. If it is not available, the code falls back to an internal tangent-construction reference for the Corey fractional-flow curve.

Optional for MPI sweeps:

```text
mpi4py
```

MPI is used only to distribute independent parameter-sweep cases. A single Buckley--Leverett solve is not domain-decomposed.

## Quick start: default Berea-core case

The default parser settings reproduce the Berea-core waterflood benchmark:

```text
L       = 6 in
D       = 1.5 in
phi     = 0.20
Swc     = 0.10
Sor     = 0.20
mu_w    = 1 cP
mu_o    = 4 cP
nw = no = 2
q       = 1 mL/min
```

Run the default Berea case and generate the main breakthrough/profile figures:

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

Here `--probe-x 0.0762` places the breakthrough probe at the midpoint of the Berea core:

```text
x = L/2 = 0.0762 m = 7.62 cm
```

Main outputs:

```text
Figure1_Sw_vs_t_probe.png
Figure1_Sw_vs_t_probe.pdf
Figure2_profiles.png
Figure2_profiles.pdf
validation_metrics_fully_mw.csv
run_summary_fully_mw.json
Sw_probe_time_fully_mw.txt
Sw_profile_fully_mw_pvi*.txt
```

## Full Berea sweep with MPI

To reproduce the resolution, flux, and modal-order studies:

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

Main outputs:

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

## Using another rock or core

The Berea case is only the default preset. To test a different rock, keep the same code and override the physical parameters from the command line.

The most important parameters are:

```text
--L            core length [m]
--D            core diameter [m]
--phi          porosity [-]
--Swc          connate-water saturation [-]
--Sor          residual-oil saturation [-]
--mu-w         water viscosity [Pa s]
--mu-o         oil viscosity [Pa s]
--nw           Corey exponent for water [-]
--no           Corey exponent for oil [-]
--krw0         endpoint water relative permeability [-]
--kro0         endpoint oil relative permeability [-]
--q-mL-min     injection rate [mL/min]
--sw-init      initial water saturation [-]
--sw-inj       injected water saturation [-]
```

### Example: one simulation for a different rock

```bash
python mw_buckley_leverett.py \
  --L 0.30 \
  --D 0.025 \
  --phi 0.18 \
  --Swc 0.15 \
  --Sor 0.25 \
  --mu-w 1.0e-3 \
  --mu-o 8.0e-3 \
  --nw 2.5 \
  --no 2.0 \
  --krw0 1.0 \
  --kro0 1.0 \
  --q-mL-min 0.5 \
  --sw-init 0.15 \
  --sw-inj 0.75 \
  --ncells 256 \
  --p 2 \
  --flux rusanov \
  --limiter tvb \
  --cfl 0.20 \
  --t-end-pvi 1.50 \
  --probe-mode auto-shock \
  --plot \
  --outdir RESULTS/general_rock_single
```

For a new rock, `--probe-mode auto-shock` is useful because the shock position may not be known in advance. The code scans the independent reference profile at `--probe-auto-pvi` and places the probe near the largest saturation gradient.

### Example: MPI sweep for a different rock

```bash
mpirun -np 8 python mw_buckley_leverett.py \
  --mpi-sweep \
  --L 0.30 \
  --D 0.025 \
  --phi 0.18 \
  --Swc 0.15 \
  --Sor 0.25 \
  --mu-w 1.0e-3 \
  --mu-o 8.0e-3 \
  --nw 2.5 \
  --no 2.0 \
  --krw0 1.0 \
  --kro0 1.0 \
  --q-mL-min 0.5 \
  --sw-init 0.15 \
  --sw-inj 0.75 \
  --ncells-list 64 128 256 512 \
  --p-list 1 2 3 4 \
  --flux-list rusanov godunov \
  --limiter tvb \
  --cfl 0.20 \
  --t-end-pvi 1.50 \
  --probe-mode auto-shock \
  --plot-sweep \
  --outdir RESULTS/general_rock_mpi_sweep
```

This command runs independent combinations of:

```text
number of cells: 64, 128, 256, 512
number of local modes p: 1, 2, 3, 4
flux: Rusanov and sampled Godunov
```

Each MPI rank receives independent cases. The solver itself remains a fixed-grid one-dimensional coefficient-space solver; MPI is used only for the sweep.

## Probe options

The breakthrough curve can be measured in two ways.

### Fixed probe

Use a physical location in meters:

```bash
--probe-x 0.0762
```

This is recommended for reproducing the Berea manuscript figures.

### Automatic shock probe

Use:

```bash
--probe-mode auto-shock
```

Optional control:

```bash
--probe-auto-pvi 0.20
--probe-scan-points 4000
```

This is recommended when testing a different rock or geometry.

## Numerical options

### Number of cells

```bash
--ncells 256
```

or for a sweep:

```bash
--ncells-list 64 128 256 512
```

### Number of local modes

```bash
--p 2
```

Here `p` is the number of local modes per cell. The polynomial degree is `p - 1`.

For the Berea benchmark, the manuscript uses `p = 2` as the main production setting because it gives the best accuracy--cost compromise among the tested orders for this shock-dominated problem.

### Numerical flux

Available choices:

```bash
--flux rusanov
--flux godunov
--flux godunov-sampled
--flux central
```

Recommended for robust production runs:

```bash
--flux rusanov
```

### Limiter

Available choices:

```bash
--limiter none
--limiter bounds
--limiter tvb
--limiter flatten
```

Recommended for the Berea benchmark:

```bash
--limiter tvb
```

## Output files

For a single case, the code writes:

```text
run_summary_fully_mw.json
validation_metrics_fully_mw.csv
Sw_probe_time_fully_mw.txt
Sw_profile_fully_mw_pvi*.txt
Figure1_Sw_vs_t_probe.png/pdf
Figure2_profiles.png/pdf
```

For a sweep, the code writes:

```text
sweep_summary.csv
sweep_summary.json
```

If `--plot-sweep` is used, it also writes manuscript-oriented plots and tables under:

```text
manuscript_plots/
```

## Reproducibility notes

- The default Berea physical parameters are built into the parser.
- All dimensional quantities are written to JSON summaries.
- The reference solution is independent of the coefficient-space solver.
- If `pywaterflood` is installed, it is used for the reference calculation.
- If `pywaterflood` is unavailable, the code uses an internal tangent-construction reference.
- MPI sweeps are embarrassingly parallel: each case is independent.
- The right boundary is an outflow boundary; no Dirichlet condition is imposed there.

## Citation

If you use this code, cite the associated manuscript:

```text
A Fixed-Grid Affine-Constrained Multiwavelet Coefficient Method
for Buckley--Leverett Shock Capturing
```

and cite this repository.

## License

This project is released under the MIT License.
