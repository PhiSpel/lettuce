# Part B — Reusable inventory for an NRBC autoresearch task

Lookup table of everything in `lettuce` (HEAD `121359d`) that a "implement a non-reflecting
boundary condition + benchmark cases to validate it" task can be built on, plus what is missing.

Legend for the **NRBC fit** column:
**COPY** = direct template to clone · **USE** = use unchanged · **ADAPT** = small edit needed ·
**REF** = read-only reference · **AVOID** = misleading / not a good model.

---

## 1. Boundary conditions — `lettuce/ext/_boundary/`

All five implement the `Boundary` ABC ([_flow.py:31-52](lettuce/_flow.py#L31-L52)):
`__call__(flow) -> torch.Tensor` (returns a **full-domain** `f`, node selection happens in
`Simulation._collide`), `make_no_collision_mask(shape, context)`,
`make_no_streaming_mask(shape, context)`, `native_available()`, `native_generator(index)`.

| BC | File (LOC) | Algorithm | `__call__` returns | `no_collision_mask` | `no_streaming_mask` | Native? | NRBC fit |
|---|---|---|---|---|---|---|---|
| `AntiBounceBackOutlet` | [anti_bounce_back_outlet.py](lettuce/ext/_boundary/anti_bounce_back_outlet.py) (109) | Anti-bounce-back on the outgoing populations at one domain face; wall velocity extrapolated as `u_w = u[face] + 0.5*(u[face] - u[neighbour])`; Krüger et al. p. 195 | mutates `flow.f` in place at the face, returns `flow.f` ([L71-91](lettuce/ext/_boundary/anti_bounce_back_outlet.py#L71-L91)) | whole face slice ([L100-103](lettuce/ext/_boundary/anti_bounce_back_outlet.py#L100-L103)) | only the *opposite* velocities at the face ([L93-98](lettuce/ext/_boundary/anti_bounce_back_outlet.py#L93-L98)) | **No** ([L105](lettuce/ext/_boundary/anti_bounce_back_outlet.py#L105)) | **COPY — nearest relative.** Same geometry handling (direction vector → `self.index`/`self.neighbor` slices, `self.velocities` selection, per-dimension `einsum` string) that any NRBC needs |
| `EquilibriumOutletP` | [equilibrium_outlet_p.py](lettuce/ext/_boundary/equilibrium_outlet_p.py) (91) | Zero-th-order extrapolation outlet: impose `rho = rho_outlet`, take `u` from the *neighbour* node, overwrite the face with `f_eq(rho_w, u_w)` | writes `flow.f[face] = feq`, returns it ([L63-73](lettuce/ext/_boundary/equilibrium_outlet_p.py#L63-L73)) | whole face slice | **all velocities except** the outgoing ones ([L75-80](lettuce/ext/_boundary/equilibrium_outlet_p.py#L75-L80)) | **No** | **COPY — second nearest relative.** This is the "reflective baseline" an NRBC must beat. Note it subclasses `AntiBounceBackOutlet` and then re-does 40 lines of `__init__` verbatim ([L18-61](lettuce/ext/_boundary/equilibrium_outlet_p.py#L18-L61)) — do not replicate that |
| `EquilibriumBoundaryPU` | [equilibrium_boundary_pu.py](lettuce/ext/_boundary/equilibrium_boundary_pu.py) (98) | Dirichlet inlet: set masked nodes to `f_eq(rho(p), u)` from prescribed pu values. Has a 45-line `checked_tensor` broadcasting helper ([L23-69](lettuce/ext/_boundary/equilibrium_boundary_pu.py#L23-L69)) accepting scalar / vector / field / full tensor | `feq` broadcast to full shape | user-supplied mask | `pass` (None) | **Yes** (`NativeEquilibriumBoundaryPu`) | **USE** as the inlet of a duct case; **REF** for the pu→lu conversion idiom and for `checked_tensor` if the NRBC takes a prescribed far-field state |
| `BounceBackBoundary` | [bounce_back_boundary.py](lettuce/ext/_boundary/bounce_back_boundary.py) (32) | Full-way bounce-back: `flow.f[flow.stencil.opposite]` | full reversed `f` | user-supplied mask | None | **Yes** (`NativeBounceBackBoundary`) | **USE** for duct side walls; **REF** as the minimal complete `Boundary` (32 lines) |
| `PartiallySaturatedBC` | [partially_saturated_boundary.py](lettuce/ext/_boundary/partially_saturated_boundary.py) (51) | Noble & Torczynski (1998) blend of BGK and bounce-back weighted by `B(ε, τ)`; Krüger p. 448 | `torch.where(mask, blended, flow.f)` | user-supplied mask | `pass` | No | **REF** — best in-tree example of a BC that *blends* two operators via a scalar weight; the shape an "absorbing layer / sponge" variant would take |

### Geometry idiom every face-boundary reuses

```python
# lettuce/ext/_boundary/anti_bounce_back_outlet.py:38-69  (condensed)
self.velocities = np.concatenate(np.argwhere(np.matmul(stencil.e, direction) > 1 - 1e-6), axis=0)
for i in direction:                     # direction e.g. [1,0] / [0,-1,0]
    if i == 0:  self.index.append(slice(None)); self.neighbor.append(slice(None))
    if i == 1:  self.index.append(-1);          self.neighbor.append(-2)
    if i == -1: self.index.append(0);           self.neighbor.append(1)
self.dims = {3: 'dc, cxy -> dxy', 2: 'dc, cx -> dx', 1: 'dc, c -> dc'}[len(direction)]
```

This gives, for free: outgoing-velocity selection, face slice, one-cell-inward neighbour slice
(what a characteristic/extrapolation NRBC needs for its `∂/∂n` stencil), and a dimension-agnostic
`einsum` string. Both existing outlets validate `direction` with the same two asserts
([L27-34](lettuce/ext/_boundary/anti_bounce_back_outlet.py#L27-L34)).

---

## 2. Flows — `lettuce/ext/_flows/`

Base class `ExtFlow` ([_ext_flow.py](lettuce/ext/_flows/_ext_flow.py), 42 LOC) requires only
`make_resolution`, `make_units`, `initial_pu`; picks a default stencil per dimension
(`[D1Q3(), D2Q9(), D3Q19()]`, [L25-26](lettuce/ext/_flows/_ext_flow.py#L25-L26)) and
`QuadraticEquilibrium`.

| Flow | File (LOC) | Reference solution? | Boundaries | Periodic? | Inlet/outlet? | NRBC fit |
|---|---|---|---|---|---|---|
| `TaylorGreenVortex` | [taylorgreen.py](lettuce/ext/_flows/taylorgreen.py) (122) | **Yes** — `analytic_solution(t)` exact in 2D ([L66-94](lettuce/ext/_flows/taylorgreen.py#L66-L94)); 3D only at `t=0` | none | fully | no | **USE** as the "unchanged-physics" control: an NRBC must not perturb a periodic case. Also the substrate of the existing convergence harness |
| `PoiseuilleFlow2D` | [poiseuille.py](lettuce/ext/_flows/poiseuille.py) (86) | **Yes** — parabolic profile ([L30-40](lettuce/ext/_flows/poiseuille.py#L30-L40)) | bounce-back top/bottom ([L77-82](lettuce/ext/_flows/poiseuille.py#L77-L82)); body force via `acceleration` property | x periodic | no | **ADAPT — best "duct" skeleton.** Replace x-periodicity with inlet + NRBC outlet and you have a duct with a known steady solution to subtract off |
| `CouetteFlow2D` | [couette.py](lettuce/ext/_flows/couette.py) (75) | **Yes** — linear profile ([L43-47](lettuce/ext/_flows/couette.py#L43-L47)); note it returns only `u`, not `(p,u)`, so it is **not** `ErrorReporter`-compatible as-is | `EquilibriumBoundaryPU` top + `BounceBackBoundary` bottom ([L63-75](lettuce/ext/_flows/couette.py#L63-L75)) | x periodic | no | **REF** — shows the moving-wall + wall pairing |
| `Obstacle` | [obstacle.py](lettuce/ext/_flows/obstacle.py) (151) | no | **inlet `EquilibriumBoundaryPU` + outlet `AntiBounceBackOutlet` + `BounceBackBoundary(mask)`** ([L107-122](lettuce/ext/_flows/obstacle.py#L107-L122)) | no | **YES — the only in-tree inflow/outflow case** | **COPY — the single most important flow for this task.** Already has the exact inlet/outlet topology; `EquilibriumOutletP` is even sitting there commented out at [L119-120](lettuce/ext/_flows/obstacle.py#L119-L120). Settable `mask` property ([L83-92](lettuce/ext/_flows/obstacle.py#L83-L92)) makes "vortex shed past a cylinder and convected out" a two-line change |
| `LambOseenVortex2D` | [lamboseenvortex.py](lettuce/ext/_flows/lamboseenvortex.py) (134) | initial condition only (Wissocq et al. 2017), `Rc=20`, `beta=0.5` hard-coded ([L94-116](lettuce/ext/_flows/lamboseenvortex.py#L94-L116)); **includes a consistent density/pressure perturbation** | none (periodic) | fully | no | **ADAPT — the "vortex convected through an outlet" case.** It already advects at `ux0` and carries a `p` perturbation; swap periodic-x for inlet + NRBC and the exit error is directly measurable |
| `DoublyPeriodicShear2D` | [doublyshear.py](lettuce/ext/_flows/doublyshear.py) (80) | `analytic_solution` raises `NotImplementedError` | none | fully | no | **REF** — a clean "smooth analytic IC via `torch.tanh`/`torch.sin`" template, i.e. exactly how to write a Gaussian pressure pulse IC |
| `DecayingTurbulence` | [decayingturbulence.py](lettuce/ext/_flows/decayingturbulence.py) (189) | spectrum only (`energy_spectrum` property); `analytic_solution` returns `None` | none | fully | no | **REF** — `randseed` ([L31](lettuce/ext/_flows/decayingturbulence.py#L31)), divergence removal, FFT-based IC construction; source of broadband content if a broadband-reflection test is wanted |
| `Cavity2D` (lid-driven) | [liddrivencavity.py](lettuce/ext/_flows/liddrivencavity.py) (68) | no | walls + moving lid ([L51-68](lettuce/ext/_flows/liddrivencavity.py#L51-L68)) | no | no | **AVOID** for NRBC — fully closed domain |

### What a new benchmark flow must supply

| Case | Needs | Already available |
|---|---|---|
| **Pressure pulse in a duct** (1D/2D) | `initial_pu()` returning a Gaussian `p` bump, `u=0`; side walls or periodic-y; NRBC on one/both x faces | `DoublyPeriodicShear2D.initial_pu` as IC template ([L54-65](lettuce/ext/_flows/doublyshear.py#L54-L65)); `PoiseuilleFlow2D` grid/units; D1Q3 stencil exists but **no 1D flow class does** — `ExtFlow` and `conftest.TestFlow` both support `d=1` |
| **Vortex convected through an outlet** | advecting vortex IC + inlet + NRBC outlet | `LambOseenVortex2D` IC (already has mean flow `ux0` and density perturbation) + `Obstacle`'s boundary list |
| **Reflection-free check against a larger reference domain** | run the same IC on a 2–4× longer domain, truncate, diff | nothing — must be written (see Gaps G3) |

---

## 3. Reporters & observables

`Reporter` ABC: `interval` + `__call__(simulation)` ([_simulation.py:31-39](lettuce/_simulation.py#L31-L39)).
`Observable` ABC: `__init__(flow)` + `__call__(f=None)` ([observable_reporter.py:17-24](lettuce/ext/_reporter/observable_reporter.py#L17-L24)).

| Component | File:line | Output | Scalar time series a grader can eat? | NRBC fit |
|---|---|---|---|---|
| `ObservableReporter` | [observable_reporter.py:161-200](lettuce/ext/_reporter/observable_reporter.py#L161-L200) | with `out=None`: list of `[step, time_pu, value…]`; else prints | **Yes — the primary grading channel.** `np.asarray(r.out)` | **USE** |
| `IncompressibleKineticEnergy` | [observable_reporter.py:34-42](lettuce/ext/_reporter/observable_reporter.py#L34-L42) | scalar, pu, volume-weighted | Yes | **USE** — energy left in the domain after the pulse should → 0 with a good NRBC |
| `Enstrophy` | [observable_reporter.py:45-68](lettuce/ext/_reporter/observable_reporter.py#L45-L68) | scalar; uses `torch_gradient` order 6 | Yes | **ADAPT** — docstring warns *"only works for periodic domains"* ([L50](lettuce/ext/_reporter/observable_reporter.py#L50)); the O(h⁶) rolled stencil wraps around the outlet face |
| `MaximumVelocity` | [observable_reporter.py:27-31](lettuce/ext/_reporter/observable_reporter.py#L27-L31) | scalar, pu | Yes | **USE** — cheap stability watchdog |
| `Mass` | [observable_reporter.py:140-158](lettuce/ext/_reporter/observable_reporter.py#L140-L158) | scalar; note it hard-codes an interior slice `f[..., 1:-1, 1:-1]` | Yes | **ADAPT** — mass conservation is exactly what a pressure outlet breaks; the hard-coded trim must be checked against the NRBC face |
| `EnergySpectrum` | [observable_reporter.py:71-137](lettuce/ext/_reporter/observable_reporter.py#L71-L137) | vector over wavenumber bins; FFT of `u` over the **whole** domain | Vector (handled by `ObservableReporter`, see [test_generic_reporters.py:21-23](tests/reporter/test_generic_reporters.py#L21-L23)) | **REF only** — velocity-only, assumes periodicity, no pressure/acoustic spectrum |
| `ErrorReporter` | [error_reporter.py](lettuce/ext/_reporter/error_reporter.py) (42) | `[err_u, err_p]` per interval, grid-normalised L2 vs. any `analytic_solution(t)` | **Yes** | **USE** — plug in a reference-domain solution or an analytic pulse |
| `NaNReporter` | [failure_reporter.py:131-153](lettuce/ext/_reporter/failure_reporter.py#L131-L153) | log file + optional VTK + `reporter.failed_iteration` | Yes (`failed_iteration`) | **USE** — needs `BreakableSimulation` |
| `HighMaReporter` | [failure_reporter.py:156-183](lettuce/ext/_reporter/failure_reporter.py#L156-L183) | same, threshold on local Ma (default 0.3) | Yes | **USE** — an NRBC that amplifies is caught here |
| `FailureReporterBase` | [failure_reporter.py:12-128](lettuce/ext/_reporter/failure_reporter.py#L12-L128) | abstract: `is_failed`, `get_results`, `locations_string`; aborts via `flow.i += 1e10` | — | **COPY** if a custom "reflection exceeded X" abort is wanted |
| `VTKReporter` / `write_vtk` | [vtk_reporter.py](lettuce/ext/_reporter/vtk_reporter.py) (64) | `.vtr` files of `p, ux, uy[, uz]`; `output_mask()` dumps the boundary mask | No (files) | **USE** for qualitative evidence in a report; `output_mask` is a good "did my BC land on the right nodes" check |
| `ProgressReporter` | [progress_reporter.py](lettuce/ext/_reporter/progress_reporter.py) (137) | wall-clock log + ETA; prints that checkpointing is **not** implemented ([L100-109](lettuce/ext/_reporter/progress_reporter.py#L100-L109)) | No | **USE** for long-run budgeting |
| `HDF5Reporter` / `LettuceDataset` | [util/datautils.py](lettuce/util/datautils.py) (156) | `.h5` of `f` per interval + pickled flow/collision metadata + version; torch `Dataset` | No (but replayable) | **USE** to archive full fields for post-hoc reflection analysis |
| `write_image` | [write_image.py](lettuce/ext/_reporter/write_image.py) (13) | PNG of a 2D array | No | **USE** for figures |
| `DragCoefficient` / `LiftCoefficient` | [examples/.../observables_force_coefficients.py](examples/advanced_projects/efficient_bounce_back_obstacle/reporter/observables_force_coefficients.py) (83) | scalar via momentum-exchange | Yes | **REF** — *not* importable from `lettuce`; shows how to write a custom `Observable` that reads state off a boundary object |
| `ProfileReporter` | [examples/.../reporter_ProfileReporter.py](examples/advanced_projects/efficient_bounce_back_obstacle/reporter/reporter_ProfileReporter.py) (54) | velocity profile at an x-station, with linear interpolation for off-grid positions | Vector per step | **COPY — closest thing to a probe reporter.** A point-probe pressure reporter (needed for reflection-coefficient work) is ~15 lines starting from this |

---

## 4. Tests — `tests/` (2 479 LOC, 59 files)

| Test / fixture | File:line | Pattern | NRBC fit |
|---|---|---|---|
| `fix_configuration` | [conftest.py:66-84](tests/conftest.py#L66-L84), [L181-186](tests/conftest.py#L181-L186) | device × dtype × native, invalid combos filtered, CUDA auto-skip | **USE** as the sweep axis |
| `fix_stencil` / `fix_stencil1d/2d/3d` | [conftest.py:18-47](tests/conftest.py#L18-L47), [L145-162](tests/conftest.py#L145-L162) | all five stencils as fixtures with readable ids | **USE** — an NRBC must work for D1Q3…D3Q27 |
| `TestFlow` | [conftest.py:195-232](tests/conftest.py#L195-L232) | minimal `ExtFlow` with uniform `u=1.01`, `p=0.01` and a **settable** `boundaries` list | **USE** — instantiate a BC with no real flow, exactly as the outlet tests do |
| `DummyFlow` | [conftest.py:243-266](tests/conftest.py#L243-L266) | `initialize()` is a no-op → uninitialised `f` | **USE** for mask/shape-only tests |
| `test_anti_bounce_back_outlet` | [test_antibounceback_outlet_bc.py:6-103](tests/boundary/test_antibounceback_outlet_bc.py#L6-L103) | **re-derives the published formula by hand** per stencil, compares to the implementation with `pytest.approx` | **COPY — the gold-standard grader shape**: "your BC must equal the equation you cite" |
| `test_equilibrium_outlet_p_algorithm` | [test_equilibrium_bc_outlet_p.py](tests/boundary/test_equilibrium_bc_outlet_p.py) (31) | builds the expected `f_eq` from `rho_outlet`, `u`; asserts at `rel=1e-2` with *"TODO rel is too big!"* ([L30-31](tests/boundary/test_equilibrium_bc_outlet_p.py#L30-L31)) | **ADAPT** — and note the weak tolerance is an open defect |
| `test_equilibrium_pressure_outlet` | [test_equilibrium_pressure_outlet.py](tests/boundary/test_equilibrium_pressure_outlet.py) (69) | **skipped** at [L5](tests/boundary/test_equilibrium_pressure_outlet.py#L5) (*"rtol is too big or simulation time too short"*); subclasses `Obstacle` with three `EquilibriumOutletP` outlets and asserts `p ≈ 0` on three faces | **COPY + FIX — this is arguably the task itself.** The assertions `torch.allclose(zeros, p[:, -1, :], atol=1e-4)` are literally a reflection test that the current BC cannot pass |
| `test_masks` | [test_bc_masks.py](tests/boundary/test_bc_masks.py) (19) | asserts `simulation.no_streaming_mask.any()` and `no_collision_mask.any()` | **USE** — cheap "is my BC wired in at all" check |
| `test_equilibrium_boundary_pu_algorithm` | [test_equilibrium_bc_pu.py](tests/boundary/test_equilibrium_bc_pu.py) (164) | builds a per-stencil × per-moment-dims fixture via `itertools.product` ([L3-21](tests/boundary/test_equilibrium_bc_pu.py#L3-L21)); subclasses the BC to override a mask; runs 1 step with `NoCollision()` | **COPY** — best example of isolating a BC from collision (`NoCollision`) and of a custom fixture generator |
| `test_flow` | [test_flow.py](tests/flow/test_flow.py) (17) | `@pytest.mark.parametrize("flowname", flow_by_name.keys())` — every registered flow runs 1 step | **ADAPT** — register the new benchmark flow to get free coverage |
| `test_obstacle` | [test_obstacle.py](tests/flow/test_obstacle.py) (32) | 2D/3D obstacle, 2 steps, skips native when a BC lacks it | **COPY** — the native-skip guard is needed verbatim for a non-native NRBC |
| `test_generic_reporters` | [test_generic_reporters.py](tests/reporter/test_generic_reporters.py) (25) | observable × case matrix; `np.asarray(reporter.out)` then `pytest.approx` | **COPY — the grader harness shape** |
| `test_initialize_fneq` | [test_initialize_fneq.py:23-32](tests/flow/test_initialize_fneq.py#L23-L32), [L69-92](tests/flow/test_initialize_fneq.py#L69-L92) | two identically-seeded flows, A/B on `ErrorReporter` output, `assert error_A < error_B` | **COPY — the "new BC beats old BC" comparison**, seeding included |
| `test_divergence` | [test_divergence.py](tests/flow/test_divergence.py) (34) | derived-quantity invariant with `pytest.approx(abs=2e-3)` | **REF** |
| `test_checkpoint` | [test_checkpoint.py:5-20](tests/test_checkpoint.py#L5-L20) | `dump` → run → `load` → `assert torch.eq(f0, flow.f).all()` | **USE** to pin a shared initial state |
| `test_nan_reporter` | [test_nan_reporter.py:5-16](tests/reporter/test_nan_reporter.py#L5-L16) | drives a flow unstable on purpose (`Re=1e12, Ma=1`), asserts `reporter.failed_iteration in [70, 80]` and that a log file exists | **COPY** for the stability half of the grader |
| `test_command_line_interface` | [test_cli.py](tests/test_cli.py) (37) | `click.testing.CliRunner`, asserts exit code | **REF** if the grader is exposed as a CLI subcommand |
| native tests | [tests/native/](tests/native) (6 files) | compare native kernel vs. python for streaming, BGK, bounce-back, masks | **REF** — the pattern for "if you also write a CUDA kernel, prove it matches" |

---

## 5. Stencils — `lettuce/ext/_stencil/`

`Stencil` is plain data: `e`, `w`, `opposite` as python lists, with `d`/`q` derived
([_stencil.py:14-28](lettuce/_stencil.py#L14-L28)). `TorchStencil` mirrors them onto the device.
`cs = 1/√3` is a class attribute on both.

| Stencil | File | `q` | `d` | `w` groups | Tested by |
|---|---|---|---|---|---|
| `D1Q3` | [d1q3.py](lettuce/ext/_stencil/d1q3.py) (10) | 3 | 1 | 2/3, 1/6×2 | [tests/stencil/](tests/stencil) (first-zero, opposite, symmetry, weights) |
| `D2Q9` | [d2q9.py](lettuce/ext/_stencil/d2q9.py) (10) | 9 | 2 | 4/9, 1/9×4, 1/36×4 | same |
| `D3Q15` | [d3q15.py](lettuce/ext/_stencil/d3q15.py) (12) | 15 | 3 | 2/9, 1/9×6, 1/72×8 | same |
| `D3Q19` | [d3q19.py](lettuce/ext/_stencil/d3q19.py) (13) | 19 | 3 | 1/3, 1/18×6, 1/36×12 | same |
| `D3Q27` | [d3q27.py](lettuce/ext/_stencil/d3q27.py) (12) | 27 | 3 | 8/27, 2/27×6, 1/54×12, 1/216×8 | same |

Defaults: `ExtFlow` picks `[D1Q3, D2Q9, D3Q19]` by dimension
([_ext_flow.py:25-26](lettuce/ext/_flows/_ext_flow.py#L25-L26)). `flow_by_name` exposes
`taylor3d_d3q19` and `taylor3d_d3q27` as separate entries
([_flow_by_name.py:10-18](lettuce/ext/_flows/_flow_by_name.py#L10-L18)).

### What dimensionality costs (analytic — torch was not available to benchmark)

Work and memory both scale as `q × ∏N`. One buffer of `f` in bytes = `4·q·∏N` (fp32) or `8·q·∏N` (fp64);
CUDA-native additionally allocates `f_next` ([_flow.py:145-155](lettuce/_flow.py#L145-L155)).

| Config | cells | populations | `f` @ fp32 | `f` @ fp64 | rel. to D2Q9 @ 256² |
|---|---|---|---|---|---|
| D1Q3, 1 024 | 1.0 k | 3.1 k | 12 kB | 25 kB | 0.005× |
| D2Q9, 128² | 16 k | 147 k | 0.6 MB | 1.2 MB | 0.25× |
| D2Q9, 256² | 66 k | 590 k | 2.4 MB | 4.7 MB | **1×** |
| D2Q9, 512² | 262 k | 2.36 M | 9.4 MB | 19 MB | 4× |
| D3Q15, 64³ | 262 k | 3.93 M | 16 MB | 31 MB | 6.7× |
| D3Q19, 64³ | 262 k | 4.98 M | 20 MB | 40 MB | 8.4× |
| D3Q27, 64³ | 262 k | 7.08 M | 28 MB | 57 MB | 12× |
| D3Q19, 128³ | 2.10 M | 39.8 M | 159 MB | 319 MB | **67×** |
| D3Q27, 128³ | 2.10 M | 56.6 M | 226 MB | 453 MB | 96× |

Two extra costs specific to this implementation:

* **Streaming is a python loop of `q-1` `torch.roll` calls per step**, each followed by a
  `torch.where` when a no-streaming mask exists
  ([_simulation.py:245-256](lettuce/_simulation.py#L245-L256)). Per-step kernel-launch overhead
  therefore scales with `q`: D3Q27 issues 3× the launches of D2Q9 at equal grid size.
* **Each boundary costs one full-domain `torch.where` per step**
  ([_simulation.py:288-304](lettuce/_simulation.py#L288-L304)) — independent of how few nodes the
  boundary actually touches. A benchmark with 3 outlets pays 3 full-field passes.

**Time-step budget for an acoustic case.** Sound travels at `cs = 0.577` lattice units/step, so a
pulse crosses `N` cells in `≈ 1.73 N` steps: 128 cells ≈ 222 steps, 256 ≈ 443, 512 ≈ 886. A
reflection test typically needs 2–3 transits. **Recommendation: 2D (D2Q9) at 128–512 in the long
direction is the right size for a CPU-only grading harness; 3D at 128³ is ~67× more expensive and
should be at most an optional extension.**

---

## 6. Utilities, CLI, checkpointing

| Utility | Location | Gives you | NRBC fit |
|---|---|---|---|
| `Context` | [_context.py](lettuce/_context.py) (122) | device/dtype/native selection with sanity asserts ([L13-56](lettuce/_context.py#L13-L56)); `empty/zero/one/full_tensor`, `convert_to_tensor`, `convert_to_ndarray`, `synchronize` | **USE** — every mask/tensor a BC allocates goes through it |
| `UnitConversion` | [_unit.py](lettuce/_unit.py) (145) | Re/Ma scaling; `relaxation_parameter_lu`, `viscosity_lu/pu`, `convert_pressure_pu_to_density_lu`, `convert_time_to_pu`, `convert_incompressible_energy_to_pu` | **USE** — an NRBC needs `cs`, `tau` and pressure↔density |
| `torch_gradient` | [utility.py:37-99](lettuce/util/utility.py#L37-L99) | O(h²/h⁴/h⁶) central differences in 2D/3D via `roll` | **REF/ADAPT** — **periodic only** (it wraps, see [L44](lettuce/util/utility.py#L44)); a one-sided difference at the NRBC face must be written |
| `torch_jacobi` | [utility.py:119-156](lettuce/util/utility.py#L119-L156) | Jacobi Poisson solver | **REF** |
| `pressure_poisson` / `initialize_pressure_poisson` | [_flow.py:271-338](lettuce/_flow.py#L271-L338) | consistent pressure field from a velocity field (`initialize_pressure` flag); `TODO: still not working in 3D` at [L309](lettuce/_flow.py#L309) | **USE** for a divergence-consistent IC (avoids spurious acoustic launch at t=0) |
| `initialize_f_neq` | [_flow.py:341-367](lettuce/_flow.py#L341-L367) | Chapman-Enskog `f⁽¹⁾` from finite-difference strain (`initialize_fneq` flag) | **USE** — halves the startup transient; proven beneficial by [test_initialize_fneq.py:90](tests/flow/test_initialize_fneq.py#L90) |
| `append_axes` | [utility.py:159-161](lettuce/util/utility.py#L159-L161) | trailing-axis broadcast helper | **USE** |
| `grid_fine_to_coarse` | [utility.py:102-116](lettuce/util/utility.py#L102-L116) | restrict `f` to every 2nd node preserving `f_eq + τ-scaled f_neq` | **USE** — the natural way to compare a small domain against a 2× reference domain |
| moments / transforms | [util/moments.py](lettuce/util/moments.py) (579) | `moment_tensor`, `Transform` ABC, `D1Q3Transform`, `D2Q9Dellar`, `D2Q9Lallemand`, `D3Q27Hermite`, `D3Q19DHumieres` | **REF** — required if the NRBC is formulated in moment space (characteristic BCs usually are) |
| `get_subclasses` | [utility.py:15-18](lettuce/util/utility.py#L15-L18) | reflection over a module | **USE** to auto-discover candidate BCs in a grader |
| CLI `benchmark` | [cli.py:57-131](lettuce/cli.py#L57-L131) | MLUPS, optional cProfile dump, optional VTK, `--streaming-strategy` | **USE** for the cost side of a score |
| CLI `convergence` | [cli.py:134-186](lettuce/cli.py#L134-L186) | TGV2D order-of-accuracy check, `sys.exit(1)` on failure | **COPY — the grader skeleton** |
| `Flow.dump` / `Flow.load` | [_flow.py:258-268](lettuce/_flow.py#L258-L268) | pickle of `f` only (not `i`, not `units`) | **USE** with the caveat in G9 |
| `StreamingStrategy` | [_default_code_gen.py:13-25](lettuce/cuda_native/_default_code_gen.py#L13-L25) | `NO_/PRE_/POST_/DOUBLE_STREAMING`, consumed at [_simulation.py:149-166](lettuce/_simulation.py#L149-L166) | **USE** — lets an NRBC be evaluated under both substep orderings |
| `BreakableSimulation` | [_simulation.py:325-344](lettuce/_simulation.py#L325-L344) | reporter-abortable time loop | **USE** with any failure reporter |
| Collision operators | [ext/_collision/](lettuce/ext/_collision) — BGK 35, TRT 33, MRT 35, Regularized 50, Smagorinsky 44, KBC 180, **`NoCollision` 17** | 7 operators behind one ABC | **USE** — `NoCollision` isolates a BC from collision (as in [test_equilibrium_bc_pu.py](tests/boundary/test_equilibrium_bc_pu.py)); TRT/Regularized change the reflection behaviour of an outlet and are a free sweep axis |
| Equilibria | [ext/_equilibrium/](lettuce/ext/_equilibrium) — `QuadraticEquilibrium` 30, `IncompressibleQuadraticEquilibrium` 26, `QuadraticEquilibriumLessMemory` 35 | 3 variants | **USE** — `EquilibriumOutletP` calls `flow.equilibrium(...)` so the NRBC inherits whichever is configured |
| Forces | [ext/_force/](lettuce/ext/_force) — `Guo` 43, `ShanChen` 29 | body forcing with velocity correction | **REF** — relevant only if the duct is force-driven like `PoiseuilleFlow2D` |

---

## 7. Gaps — what an NRBC benchmark needs that does **not** exist

| # | Gap | Evidence | Consequence for task design |
|---|---|---|---|
| **G1** | **No acoustics or compressibility diagnostic of any kind.** A repo-wide grep for `acoustic\|reflect\|nrbc\|absorbing\|sponge\|impedance\|far.?field\|thompson` over all `.py/.rst/.md` returns exactly one hit — a passing mention of "speed of sound" in a tutorial comment ([01b:22](examples/01b_first_example_obstacle.py#L22)) | — | The task must define its own metric. Nothing to reuse, nothing to accidentally copy from |
| **G2** | **No reflection-coefficient measure.** The observable set is `MaximumVelocity`, `IncompressibleKineticEnergy`, `Enstrophy`, `EnergySpectrum`, `Mass` ([observable_reporter.py:12-14](lettuce/ext/_reporter/observable_reporter.py#L12-L14)) — all velocity/energy, none pressure-based | — | `R = p_reflected/p_incident` must be built. Cheapest route: a point-probe `Observable` returning `flow.p_pu[0, x_probe, y_probe]`, ~10 lines on the `Observable` ABC, then post-process the time trace |
| **G3** | **No far-field / large-reference-domain machinery.** No flow takes a "reference length" or "truncate this domain" argument; no utility diffs two runs of different extent | — | The reference-solution route (run a 4× longer domain, compare the common sub-region) must be written from scratch. `grid_fine_to_coarse` handles refinement, not extension |
| **G4** | **No point- or line-probe reporter in the installed package.** The only probe-like code is `ProfileReporter` in an examples subproject, not importable from `lettuce` | [reporter_ProfileReporter.py](examples/advanced_projects/efficient_bounce_back_obstacle/reporter/reporter_ProfileReporter.py) | Must be written (or copied from the example). This is the single most load-bearing missing piece for acoustic grading |
| **G5** | **No 1D flow class.** `D1Q3` exists and is fixtured, `ExtFlow` accepts `len(resolution)==1` ([_ext_flow.py:22-26](lettuce/ext/_flows/_ext_flow.py#L22-L26)), `conftest.TestFlow` runs 1D — but no shipped flow is 1D, and `flow_by_name` has no 1D entry | [_flow_by_name.py:10-18](lettuce/ext/_flows/_flow_by_name.py#L10-L18) | The cheapest possible NRBC test case (1D pressure pulse, ~1 000 cells, microseconds per step) has to be authored. Low effort, high value |
| **G6** | **No time-domain signal-processing helpers.** FFT appears only spatially (`EnergySpectrum`, `DecayingTurbulence`); nothing windows or transforms a *time* series | — | Reflection spectra / phase analysis require numpy work in the grader, outside `lettuce` |
| **G7** | **Neither outlet BC is available on the CUDA-native path.** `AntiBounceBackOutlet.native_available() → False` ([L105](lettuce/ext/_boundary/anti_bounce_back_outlet.py#L105)); `EquilibriumOutletP` likewise ([L87](lettuce/ext/_boundary/equilibrium_outlet_p.py#L87)); there is no outlet template under [cuda_native/ext/_boundary/](lettuce/cuda_native/ext/_boundary) (only bounce-back, equilibrium-pu, no-boundary) | Tests skip native whenever an outlet is present ([test_obstacle.py:41-45](tests/flow/test_obstacle.py#L41-L45)) | State explicitly that the NRBC is python/torch-only, or the agent will chase a CUDA codegen rabbit hole. Conversely, "also write the native kernel" is a good hard-mode extension |
| **G8** | **No per-boundary convergence/order harness.** `lettuce convergence` measures the *global* scheme on a periodic TGV; nothing measures the order of a boundary condition | [cli.py:134-186](lettuce/cli.py#L134-L186) | The grader must extend the CLI pattern rather than reuse it verbatim |
| **G9** | **Checkpoints are populations-only.** `dump`/`load` pickle `f` and nothing else — not `flow.i`, not `units`, not resolution; the aspiration is commented out in the test | [_flow.py:258-268](lettuce/_flow.py#L258-L268), [test_checkpoint.py:22-53](tests/test_checkpoint.py#L22-L53); `ProgressReporter` prints *"current version of lettuce does not support checkpointing"* ([L100-109](lettuce/ext/_reporter/progress_reporter.py#L100-L109)) | A grader can pin the initial state but not resume mid-run; runs must be short enough to complete in one process |
| **G10** | **Existing outlet tests are skipped or loose.** `test_equilibrium_pressure_outlet` is skipped ([L5](tests/boundary/test_equilibrium_pressure_outlet.py#L5)); `test_equilibrium_bc_outlet_p` passes only at `rel=1e-2` with *"TODO rel is too big!"* ([L30-31](tests/boundary/test_equilibrium_bc_outlet_p.py#L30-L31)) | — | There is **no trustworthy baseline number** for outlet reflection in-tree. The task should require the agent to establish the baseline for `EquilibriumOutletP` / `AntiBounceBackOutlet` first, then beat it |
| **G11** | **`Enstrophy` and `torch_gradient` assume periodicity** — both say so verbatim: *"The function only works for periodic domains"* at [observable_reporter.py:50](lettuce/ext/_reporter/observable_reporter.py#L50) and [utility.py:44](lettuce/util/utility.py#L44) | — | Any gradient-based metric evaluated near the NRBC face is wrong by construction. Either restrict the metric to the interior or write one-sided differences |
| **G12** | **`Mass` hard-codes a 2D-style interior slice** `f[..., 1:-1, 1:-1]` ([observable_reporter.py:155](lettuce/ext/_reporter/observable_reporter.py#L155)) | — | Mass-conservation grading needs a corrected observable |
| **G13** | **`CouetteFlow2D.analytic_solution()` returns only `u`** and takes no `t` ([couette.py:43-47](lettuce/ext/_flows/couette.py#L43-L47)) — incompatible with `ErrorReporter`, which calls `analytical_solution(t=t)` and unpacks `(p, u)` ([error_reporter.py:30](lettuce/ext/_reporter/error_reporter.py#L30)) | — | Do not point the task at Couette as an `ErrorReporter` case without flagging this |
| **G14** | **Two live boundary APIs.** `Flow.boundaries` is deprecated with a warning ([_flow.py:103-120](lettuce/_flow.py#L103-L120)) but `Obstacle`, `Poiseuille`, `Couette`, `Cavity2D` and most tests still use `post_boundaries` / `boundaries` inconsistently | — | The task prompt should state which property a new flow must implement, or grading will be noisy |
| **G15** | **No no-slip/no-penetration-free "open" side walls.** Everything that is not periodic is bounce-back. A duct with an NRBC outlet and periodic transverse direction is fine; a truly open 2D far field (all four faces non-reflecting) has no precedent in-tree | — | Prefer a duct/channel geometry for v1 of the benchmark; a free-field radiating monopole is a stretch goal |

---

## 8. Recommended minimal asset set for a v1 NRBC task

| Role | Asset | Action |
|---|---|---|
| BC template | [anti_bounce_back_outlet.py](lettuce/ext/_boundary/anti_bounce_back_outlet.py) | copy the direction/slice/velocity-selection block |
| Reflective baseline to beat | [equilibrium_outlet_p.py](lettuce/ext/_boundary/equilibrium_outlet_p.py) + [anti_bounce_back_outlet.py](lettuce/ext/_boundary/anti_bounce_back_outlet.py) | measure, report |
| Geometry / topology | [obstacle.py:107-122](lettuce/ext/_flows/obstacle.py#L107-L122) | inlet + outlet + walls, already assembled |
| IC style | [doublyshear.py:54-65](lettuce/ext/_flows/doublyshear.py#L54-L65) (smooth analytic) and [lamboseenvortex.py:79-130](lettuce/ext/_flows/lamboseenvortex.py#L79-L130) (advected vortex with pressure) | write a Gaussian pressure pulse; write an advected vortex |
| Clean-start helpers | `initialize_pressure` + `initialize_fneq` flags ([_flow.py:127-143](lettuce/_flow.py#L127-L143)) | enable both |
| Metric plumbing | `Observable` ABC + `ObservableReporter(out=None)` | write a point-probe pressure observable (**new**, ~10 lines) |
| Stability guard | `NaNReporter`, `HighMaReporter`, `BreakableSimulation` | use unchanged |
| Grader skeleton | [cli.py:134-186](lettuce/cli.py#L134-L186) + [test_generic_reporters.py](tests/reporter/test_generic_reporters.py) + [test_initialize_fneq.py:69-92](tests/flow/test_initialize_fneq.py#L69-L92) | refine → measure → assert; A/B against baseline |
| Sweep axes | `fix_stencil`, `fix_configuration`, `StreamingStrategy`, collision operator | free breadth |
| Size | D2Q9, 128–512 long direction, fp64, ~2–3 acoustic transits (≈ 1.7 N steps each) | CPU-gradeable |
