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
| `examples/20_local_mixed_null.py` | Mixed local dipole moment, separated-pair and ideal controls, angular obstruction and spatial derivative refinement |
| `examples/21_collocation_nullspace.py` | Unmodified rectangular pair factors, explained finite-grid nulls and exact angular overintegration |
| `examples/22_first_principles.py` | Guiding-center coordinates, full-marginal conservation and exact uniform-mode explanations |
| `examples/23_difficult_correction.py` | Two auxiliaries on one frozen dipole correction: tightened reference, independent residual, matched-error timings and memory |
| `examples/24_encounter_phase.py` | Fixed close encounter: nested incoming phases, timestep refinement and independent Cartesian trajectories |
| `examples/25_encounter_censoring.py` | Full-trajectory phase search and timestep/individual-particle checks near the reflection boundary |

Example 12 uses dense spatial blocks per velocity node and writes `results/nonuniform_spatial_blocks/`. A single predefined case can be run with:

```sh
SM_EVOLUTION_FIELDS=dipole SM_EVOLUTION_CASES=mu13 MPLBACKEND=Agg python examples/12_nonuniform_evolution.py
```

It saves the last accepted state and each Newton correction's diagnostics. The earlier fixed-reference line-preconditioned pilot is in `results/nonuniform_fixed_reference/`. The old-population-scaled campaign remains archived in `results/nonuniform_entropy/`; reproduce it at its recorded commit. Equal tolerances in the two residual metrics do not imply equal accuracy.

Example 13 preserves its original audit inputs and writes new runs in `results/encounter_validation/reproduction/`. MP4 export in example 09 uses `ffmpeg`; GIF export uses the Python dependencies.

Example 23 verifies the frozen input and exact reconstructed grid before timing. It measures one linear correction, not a complete trajectory. Run without other computational work for interpretable timing comparisons. Example 24 deliberately exits with an error if its declared phase convergence checks fail; the original 512-phase run does fail those checks, despite passing its trajectory controls. Its saved outputs distinguish these outcomes.

Example 25 writes `results/encounter_censoring/precision_checked/`. It preserves the original width-limited search from commit `687be127` and the full-trajectory search from `661e3c7` in `results/encounter_censoring/refined/`. The current search also records six failed trajectory-precision checks at its final phase bracket. Its deliberate nonzero exit reports unresolved coverage; small energy errors do not certify trajectory accuracy. No monotonicity of exit time with phase is assumed.

Every scientific result records its producing commit, inputs, normalized units, versions and hardware. See the [validation ledger](../results/validation.csv) and [reproduction record](../results/deep_reproduction.json).
