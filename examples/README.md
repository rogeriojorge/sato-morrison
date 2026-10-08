# Run the examples

Run from the repository root after installation. Each file is an editable top-level script with explicit inputs, printed progress, diagnostics, plots and saved data under `results/`.

Prefix each command below with `MPLBACKEND=Agg python`.

| Script | Calculation |
|---|---|
| `examples/00_uniform_reference.py` | Eq. (181) oracle and backward-Euler convergence |
| `examples/01_relaxation.py` | Positive nonlinear solve, entropy and timestep refinement |
| `examples/02_geometry.py` | Toroidal controls, eight refinements and raw nullspace |
| `examples/03_controls.py` | Lorentz, Dougherty and physical 3V Landau references |
| `examples/04_encounters.py` | Direct encounters and phase/endpoint controls |
| `examples/05_benchmarks.py` | Matched-error runtime and memory |
| `examples/06_range.py` | Finite-range spectrum and kernel resolution |
| `examples/07_equilibria.py` | Five-field stationary density quadrature |
| `examples/08_fields.py` | Three nonlinear nonuniform collision boxes |
| `examples/09_visual_summary.py` | Operator animation and README panels; uses recorded results |
| `examples/10_encounter_ensemble.py` | Bounded incoming flux and held-out scattering table |
| `examples/11_field_velocity.py` | Nonuniform initial-production velocity/tail convergence |
| `examples/12_nonuniform_evolution.py` | Positive lagged-mobility evolution with fixed initial residual scale; 45 independent refinement cases |
| `examples/13_encounter_validation.py` | Wider incoming-flux coverage and independent table validation |
| `examples/14_geometry_sensitivity.py` | Local production derivative, independent pair form and finite-difference plateau |
| `examples/15_dipole_obstruction.py` | Matched constraints, different flux populations and a positive distance floor |
| `examples/16_scattering_table.py` | Eight-node periodic table and fresh independent validation |
| `examples/17_encounter_duration.py` | Original nonexit: flight budget, timestep, endpoint and independent pair audit |
| `examples/18_near_uniform.py` | Analytic and direct near-uniform tensor-quadrature limit |
| `examples/19_solver_accuracy.py` | Separate nonlinear/linear tolerance scans at fixed grid and timestep |

Example 12 writes `results/nonuniform_fixed_reference/`. A single predefined case can be run with:

```sh
SM_EVOLUTION_FIELDS=dipole SM_EVOLUTION_CASES=mu13 MPLBACKEND=Agg python examples/12_nonuniform_evolution.py
```

It saves the last accepted state and each Newton correction's diagnostics. The old-population-scaled campaign remains archived in `results/nonuniform_entropy/`; reproduce it at its recorded commit. Equal tolerances in the two residual metrics do not imply equal accuracy.

Example 13 preserves its original audit inputs and writes new runs in `results/encounter_validation/reproduction/`. MP4 export in example 09 uses `ffmpeg`; GIF export uses the Python dependencies.

Every scientific result records its producing commit, inputs, normalized units, versions and hardware. See the [validation ledger](../results/validation.csv) and [reproduction record](../results/deep_reproduction.json).
