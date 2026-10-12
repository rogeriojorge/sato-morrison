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
| `examples/22_first_principles.py` | Guiding-center coordinates, full-marginal conservation, pair/entropy equations and saved relaxation curves |
| `examples/23_difficult_correction.py` | Two auxiliaries on one frozen dipole correction: tightened reference, independent residual, matched-error timings and memory |
| `examples/24_encounter_phase.py` | Fixed close encounter: nested incoming phases, timestep refinement and independent Cartesian trajectories |
| `examples/25_encounter_censoring.py` | Full-trajectory phase search and timestep/individual-particle checks near the reflection boundary |
| `examples/26_tail_guard.py` | Recorded failed Newton-ray trials and floating-point positivity guards; no new solve |
| `examples/27_encounter_movie.py` | 3D particle paths, equal-scale close projections and shared-clock movies from checked encounters |
| `examples/28_spatial_relaxation.py` | Dipole geometry, three accepted spatial-flow maps, velocity shapes and conserved μ-bin populations |
| `examples/29_operator_comparison.py` | Shared Gaussian data: constrained stationarity, Lorentz and Dougherty evolution, matched initial Landau pressure rate, full μ marginals and velocity-distribution movie |
| `examples/30_landau_consistency.py` | Full initial physical Coulomb term, continuum Gaussian reference and velocity/kernel refinements |
| `examples/31_landau_trajectory.py` | Full 3V trajectories and twelve short-time grid, timestep, tail, kernel and solver refinements |
| `examples/32_landau_audit.py` | Independent NumPy/SciPy reconstruction and replay of every saved Landau stage |
| `examples/33_encounter_precision.py` | Independent 50/70-digit Taylor paths from exact saved binary64 states |
| `examples/34_physical_validation.py` | Fixed-strength field scan, bounded flux, Born controls, heldout table, nonthermal ring and physical trajectories |
| `examples/35_encounter_audit.py` | Independent reconstruction of encounter moments, measures, gates, paths and high-precision provenance |
| `examples/36_physical_figures.py` | Equations and saved-data figures for individual moment exchange, field scaling and accuracy checks |
| `examples/37_physical_refinement.py` | Focused B = 4 quadrature, finer table with two heldout samples, and weak-force B = 32 amplitude checks |

Example 12 uses dense spatial blocks per velocity node and writes `results/nonuniform_spatial_blocks/`. A single predefined case can be run with:

```sh
SM_EVOLUTION_FIELDS=dipole SM_EVOLUTION_CASES=mu13 MPLBACKEND=Agg python examples/12_nonuniform_evolution.py
```

It saves the last accepted state and each Newton correction's diagnostics. The earlier fixed-reference line-preconditioned pilot is in `results/nonuniform_fixed_reference/`. The old-population-scaled campaign remains archived in `results/nonuniform_entropy/`; reproduce it at its recorded commit. Equal tolerances in the two residual metrics do not imply equal accuracy.

Example 13 preserves its original audit inputs and writes new runs in `results/encounter_validation/reproduction/`. MP4 export in example 09 uses `ffmpeg`; GIF export uses the Python dependencies.

Example 23 verifies the frozen input and reconstructed grid before timing one linear correction. Run without other computational work for interpretable timing comparisons.

Example 24 writes `results/encounter_phase/all_moments8192/`. It reuses the audited 2,048-phase data, then checks up to 8,192 phases within a 1,200-second budget checked between work items. All three moments must pass two successive 1% angular changes and a whole-grid 0.1% half-step comparison. Independent Cartesian checks cover three anchors and the union of the three strongest new contributors to each moment. Any unresolved criterion causes a nonzero exit. The original failed 512-phase run and the second-moment-only extension remain in their recorded directories. Continuous angular coverage and physical-rate calibration require further work.

Example 25 writes `results/encounter_censoring/precision_checked/`. It preserves the original width-limited search from commit `687be127` and the full-trajectory search from `661e3c7` in `results/encounter_censoring/refined/`. The current search also records six failed trajectory-precision checks at its final phase bracket. Its deliberate nonzero exit reports unresolved coverage; small energy errors do not certify trajectory accuracy. No monotonicity of exit time with phase is assumed.

Examples 27 and 28 render recorded data. Example 27 writes 120-frame GIF and MP4 movies; MP4 encoding uses `ffmpeg` when installed. Its 3D view compresses the long z direction, while the close projections use equal spatial scales. Example 28 verifies exact dipole field lines and reconstructs observables with the B-weighted quadrature measure. It shows only the three archived spatial states at t = 0, 0.015 and 0.02. Neither example runs a new collision solve.

Every scientific result records its producing commit, inputs, normalized units, versions and hardware. See the [validation ledger](../results/validation.csv) and [reproduction record](../results/deep_reproduction.json).

`30_landau_consistency.py` compares the full initial Coulomb collision term with an independent continuum Gaussian formula. It records velocity, extent and softening scans, saves figures and arrays, and raises an accuracy failure when the declared full-term target remains unresolved.

`31_landau_trajectory.py` saves full 3V Landau states and compares velocity-grid, timestep, quadrature-scale, kernel and solver-tolerance refinements. The completed campaign in `results/landau_trajectory_dt_refined` has twelve trajectories and 100 independently checked advances through t = 0.2; every required refinement passes the declared 1% relaxation-increment target. Earlier interrupted and timestep-underresolved campaigns remain archived. The output guard prevents overwriting saved evidence.

`32_landau_audit.py` independently rebuilds the Hermite measure, velocity derivatives and Coulomb pairs with NumPy/SciPy. It replays every saved accepted stage, checks every saved case against the baseline in two velocity boxes, and independently computes moment increments and the observed timestep order. Its default input is the archived first campaign. `SM_LANDAU_AUDIT_INPUT` selects another immutable campaign and `SM_LANDAU_AUDIT_OUTPUT` selects a fresh receipt; relative paths resolve from the repository root. Existing receipts are protected.

The current example 31 uses fixed scale θ = 0.6 for its 12³/16³/20³ grid study, timesteps 0.05/0.025/0.0125, separate scales 0.585/0.625, Coulomb softenings 0.2/0.1/0, and separate nonlinear/linear tolerance variations. It compares every shared saved time in velocity cubes of half-width 4 and 5, with quadrature orders 32/48/64. `SM_LANDAU_OUTPUT=results/landau_reproduction` selects a fresh output directory; existing outputs are protected. The initial-only scale preflights, including a failed narrower scale, are retained separately.

Example 33 converts every saved initial component to its exact binary64 value before arbitrary-precision integration. It reproduces the reflected phase-2,054 path and finds that both previously ambiguous endpoint states transmit. The 50/70-digit refinement resolves these selected states, while the complete angular integral remains open. Install `.[dev,validation]` for the plotting and mpmath dependencies. `SM_PRECISION_OUTPUT=results/precision_reproduction` selects a fresh output.

Example 34 keeps κ = 0.03 and λ = 3 while scanning B = 2, 4, 8, 16, 32. It tests conditional incoming bands with the tagged-particle plane-flux measure; no physical density is supplied. The archived campaign exits nonzero for its B = 4 coordinate comparison and heldout table's 95th-percentile error. Its other checks, including all 24 refined ring comparisons, pass. `SM_PHYSICAL_OUTPUT=results/physical_reproduction` selects a fresh output. For examples 34 and 37, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; four CPU workers start before the parent initializes a JAX device.

Example 37 reads the unchanged archived campaign, independently declares its focused refinements, and saves them in `results/physical_refinement`. `SM_PHYSICAL_REFINEMENT_OUTPUT=results/physical_refinement_reproduction` selects a fresh output. The new B = 32 weak-force check compares against a 2% analytical-amplitude target; it is separate from the original campaign's acceptance criteria.

Example 35 audits historical scientific source hashes against their matching Git blobs, so later output-path changes do not relabel the producing revision. `SM_ENCOUNTER_AUDIT_OUTPUT=results/encounter_audit_reproduction` selects a fresh audit directory. Example 36 protects its output with `PHYSICAL_FIGURE_OUTPUT`; its figures use the original campaign's immutable input hashes and visibly retain that campaign's failed checks.
