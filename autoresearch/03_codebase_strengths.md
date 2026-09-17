# Part A — Why `lettuce` is a good autoresearch substrate

Survey of the `lettuce` repository (v0.2.3, branch `worktree-autoresearch-nrbc`, HEAD `121359d`)
from one question: *can an LLM agent be handed a self-contained research task in here, and can the
result be graded automatically?*

Verdict in one line: **yes** — the package is small (≈5.3 kLOC of real source), the extension
points are single-file and single-method, the physics/numerics/execution split is clean, and there
is already a working convergence-order harness plus a scalar-time-series reporter that a grader
can consume without modification.

---

## A1. Size and readability — the whole thing fits in a context window

| Scope | LOC |
|---|---|
| `lettuce/` total, excluding the versioneer-generated `_version.py` | **5 957** |
| `lettuce/*.py` core only (context, flow, simulation, stencil, unit, base, cli) | **1 294** |
| `tests/` total | **2 479** |
| Largest hand-written module ([`util/moments.py`](lettuce/util/moments.py)) | 579 |
| Largest core module ([`_flow.py`](lettuce/_flow.py)) | 367 |
| Largest boundary condition ([`anti_bounce_back_outlet.py`](lettuce/ext/_boundary/anti_bounce_back_outlet.py)) | **109** |
| Smallest boundary condition ([`bounce_back_boundary.py`](lettuce/ext/_boundary/bounce_back_boundary.py)) | **32** |
| Abstract base ([`base.py`](lettuce/base.py)) | 52 — and it is ~90 % docstring |

A complete, working boundary condition is 32 lines. An agent asked to write a new one has a
*whole* reference implementation in front of it, not an excerpt:

```python
# lettuce/ext/_boundary/bounce_back_boundary.py:10-26
class BounceBackBoundary(Boundary):
    _mask: torch.Tensor
    def __init__(self, mask: torch.Tensor):
        self._mask = mask
    def __call__(self, flow: 'Flow'):
        return flow.f[flow.stencil.opposite]
    def make_no_streaming_mask(self, shape, context) -> Optional[torch.Tensor]:
        return None
    def make_no_collision_mask(self, shape, context) -> Optional[torch.Tensor]:
        return self._mask
```

**Dependency depth is one hop.** Every extension module imports only from the package root
(`from ... import Boundary, Context`) plus `torch`/`numpy` —
[bounce_back_boundary.py:1-5](lettuce/ext/_boundary/bounce_back_boundary.py#L1-L5),
[partially_saturated_boundary.py:1-3](lettuce/ext/_boundary/partially_saturated_boundary.py#L1-L3).
There is no service locator, no config system, no dependency-injection container, no plugin
manifest. A new file plus two lines in an `__init__.py` is the entire integration surface.

---

## A2. Extension points are explicit, abstract and tiny

The four extension protocols are declared as ABCs in two files:

| Protocol | Declared at | Required methods |
|---|---|---|
| `Boundary` | [_flow.py:31-52](lettuce/_flow.py#L31-L52) | `__call__(flow)`, `make_no_collision_mask`, `make_no_streaming_mask`, `native_available`, `native_generator` |
| `Equilibrium` | [_flow.py:17-28](lettuce/_flow.py#L17-L28) | `__call__(flow, rho, u)`, `native_available`, `native_generator` |
| `Flow` | [_flow.py:55-125](lettuce/_flow.py#L55-L125) | `initial_pu()`; optionally the `pre_boundaries` / `post_boundaries` properties |
| `Collision`, `Reporter` | [_simulation.py:17-39](lettuce/_simulation.py#L17-L39) | `__call__` (+ native hooks for collision) |

`Boundary` has exactly **five** abstract methods and two of them (`native_available`,
`native_generator`) are discharged with `return False` / `pass` — which is precisely what both
existing outlet conditions do
([anti_bounce_back_outlet.py:105-109](lettuce/ext/_boundary/anti_bounce_back_outlet.py#L105-L109)).
So the *real* contract for a new non-reflecting boundary is three methods.

**Registration is declarative and grep-able.**
[`_boundary/__init__.py`](lettuce/ext/_boundary/__init__.py) is 13 lines: five imports, one
`__all__`. [`ext/__init__.py`](lettuce/ext/__init__.py#L1-L7) is seven star-imports and the package
root re-exports everything ([`lettuce/__init__.py`](lettuce/__init__.py)), so `lt.MyNewBoundary`
works the moment those two lines are added. No build step, no registry file to regenerate.

Two genuine registries exist and both are useful for grading:

* [`flow_by_name`](lettuce/ext/_flows/_flow_by_name.py#L10-L18) — a
  `Dict[str, (FlowClass, Stencil)]` that drives the CLI (`lettuce benchmark -f taylor2d`) *and* the
  parametrised smoke test `@pytest.mark.parametrize("flowname", flow_by_name.keys())` at
  [test_flow.py:4](tests/flow/test_flow.py#L4). Adding one entry buys a CLI-runnable case and a
  CI-covered case for free.
* [`get_subclasses`](lettuce/util/utility.py#L15-L18), used at
  [conftest.py:112](tests/conftest.py#L112) as
  `COLLISIONS = list(get_subclasses(Collision, lettuce.ext._collision))` — reflection-based test
  discovery, so new components are swept into existing test matrices automatically.

---

## A3. Physics / numerics / execution are genuinely separated

This is the strongest structural argument: an agent can change one layer without touching the others.

| Layer | File | LOC | Imports | Owns |
|---|---|---|---|---|
| **Hardware / precision** | [`_context.py`](lettuce/_context.py) | 122 | torch, numpy | device, dtype, `use_native`; tensor factories `zero_tensor` / `one_tensor` / `full_tensor` / `convert_to_tensor` / `convert_to_ndarray` |
| **Physics scaling** | [`_unit.py`](lettuce/_unit.py) | 145 | **numpy only — no torch** | Re, Ma, pu↔lu for velocity, pressure, density, time, length, acceleration, energy; `relaxation_parameter_lu` |
| **Discrete velocity set** | [`_stencil.py`](lettuce/_stencil.py) | 50 | torch, numpy | `Stencil` (plain python lists) + `TorchStencil` (device tensors); `d`, `q` as `cached_property` |
| **State + observables** | [`_flow.py`](lettuce/_flow.py) | 367 | torch, numpy | `f`, `rho()`, `u()`, `j()`, `p_pu`, `u_pu`, entropy, shear tensor, initialisation |
| **Time loop** | [`_simulation.py`](lettuce/_simulation.py) | 343 | torch, numpy | ordering of boundary / collision / streaming, masks, reporters, native dispatch |
| **Codegen** | [`cuda_native/`](lettuce/cuda_native) | ~740 | — | optional CUDA kernel generation, fully opt-in via `Context(use_native=...)` |

Notable: **`UnitConversion` has no torch dependency at all** — pure algebra over floats, tensors or
arrays ([_unit.py:1-24](lettuce/_unit.py#L1-L24)). An agent reasoning about the physical scaling of
an acoustic problem ("what is `cs` here?", "what Mach number keeps the pulse linear?") reads 145
lines and is done.

The execution layer's contract with boundaries is explicit and heavily commented. Boundaries are
selected by an **integer** mask, not a boolean one, so an arbitrary number of boundaries can be
layered in a defined order:

```python
# lettuce/_simulation.py:288-304
for i, boundary in enumerate(self.pre_boundaries):
    torch.where(torch.eq(self.no_collision_mask, i), boundary(self.flow), self.flow.f, out=self.flow.f)
torch.where(torch.eq(self.no_collision_mask, self.collision_index),
            self.collision(self.flow), self.flow.f, out=self.flow.f)
for i, boundary in enumerate(self.post_boundaries, start=self.collision_index+1):
    torch.where(torch.eq(self.no_collision_mask, i), boundary(self.flow), self.flow.f, out=self.flow.f)
```

The `collision_index` bookkeeping is documented in place with a worked example
([_simulation.py:60-69](lettuce/_simulation.py#L60-L69)) and the mask semantics again at
[_simulation.py:83-96](lettuce/_simulation.py#L83-L96). For an NRBC — which must run in a specific
place relative to collision and streaming — this explicit pre/post split is exactly the hook needed,
and `StreamingStrategy`
([_default_code_gen.py:13-25](lettuce/cuda_native/_default_code_gen.py#L13-L25):
`NO_STREAMING` / `PRE_STREAMING` / `POST_STREAMING` / `DOUBLE_STREAMING`, consumed at
[_simulation.py:149-166](lettuce/_simulation.py#L149-L166)) lets the agent vary the substep order
without editing the loop.

---

## A4. Validation machinery already exists

**Analytical / reference solutions shipped in-tree:**

| Flow | Method | Location | Valid for |
|---|---|---|---|
| Taylor-Green vortex | `analytic_solution(t)` → `(p, u)` | [taylorgreen.py:66-94](lettuce/ext/_flows/taylorgreen.py#L66-L94) | exact in 2D at all `t`; 3D only at `t=0` (warns at [L67-68](lettuce/ext/_flows/taylorgreen.py#L67-L68)) |
| Poiseuille 2D | `analytic_solution(t=0)` → `(p, u)` | [poiseuille.py:30-40](lettuce/ext/_flows/poiseuille.py#L30-L40) | steady parabolic profile |
| Couette 2D | `analytic_solution()` → `u` | [couette.py:43-47](lettuce/ext/_flows/couette.py#L43-L47) | linear profile |
| Lamb-Oseen vortex | `initial_lamboseenvortex()` | [lamboseenvortex.py:79-130](lettuce/ext/_flows/lamboseenvortex.py#L79-L130) | initial condition after Wissocq et al. 2017 |

**A ready-made error metric.** [`ErrorReporter`](lettuce/ext/_reporter/error_reporter.py) is 42
lines: it takes any `analytical_solution(t)` callable and emits a grid-normalised L2 error pair per
interval —

```python
# lettuce/ext/_reporter/error_reporter.py:34-37
err_u = torch.norm(u - uref) / resolution ** (simulation.flow.stencil.d / 2)
err_p = torch.norm(p - pref) / resolution ** (simulation.flow.stencil.d / 2)
```

with `out=None` it accumulates into `reporter.out` as a plain python list of `[err_u, err_p]`.

**A convergence-order harness that already exits non-zero on failure.** The CLI `convergence`
command runs TGV2D at resolutions 2^4…2^8 in diffusive scaling, computes observed orders, and fails
the process ([cli.py:134-186](lettuce/cli.py#L134-L186)):

```python
# lettuce/cli.py:176-184
tol = 1e-1
if not (2 - tol) < factor_u / 2 < (2 + tol):
    print(f"FAILED: Velocity convergence order {factor_u / 2} is not in [1.9, 2.1")
    sys.exit(1)
if not (1 - tol) < factor_p / 2 < (1 + tol): ... sys.exit(1)
```

This is *the* template for an NRBC grader: swap the flow and the metric, keep the "refine, measure,
assert order" skeleton. It is already wired into CI, so the pattern is known-good on CPU-only runners.

**Physics-invariant tests** (gradeable without a reference solution): mass and momentum conservation
([tests/collision/](tests/collision)), collision fixed-point, moment orthogonality
([tests/moments/](tests/moments)), divergence-freeness of an initial condition
([test_divergence.py:32-34](tests/flow/test_divergence.py#L32-L34) asserts `divergence ≈ 0` to
`abs=2e-3`), and "better initialisation ⇒ smaller error"
([test_initialize_fneq.py:90](tests/flow/test_initialize_fneq.py#L90):
`assert error_u_neq < error_u_eq`).

---

## A5. Results are measurable and replicable

**Scalar time series out of the box.** `ObservableReporter(observable, interval, out=None)` appends
`[step, time_pu, value]` rows to a python list
([observable_reporter.py:185-200](lettuce/ext/_reporter/observable_reporter.py#L185-L200)).
`np.asarray(reporter.out)` is directly gradeable — exactly what the existing test does:

```python
# tests/reporter/test_generic_reporters.py:15-25
reporter = ObservableReporter(Observable(flow), interval=1, out=None)
simulation = Simulation(flow, collision, [reporter]); simulation(2)
values = np.asarray(reporter.out)
assert values[1, 2] == pytest.approx(values[0, 2], rel=0.05)
```

**Determinism.** All flows except `DecayingTurbulence` are deterministic given
`(resolution, Re, Ma, stencil, dtype)`; `DecayingTurbulence` takes an explicit `randseed`
([decayingturbulence.py:31](lettuce/ext/_flows/decayingturbulence.py#L31), applied at
[L90](lettuce/ext/_flows/decayingturbulence.py#L90)), and
[test_initialize_fneq.py:23-32](tests/flow/test_initialize_fneq.py#L23-L32) demonstrates seeding two
flows identically for an A/B comparison — the exact pattern a "does my new BC beat the old one"
grader needs. `Context(dtype=torch.float64)` gives reproducible CPU runs.

**Checkpointing.** `Flow.dump` / `Flow.load` pickle the populations
([_flow.py:258-268](lettuce/_flow.py#L258-L268)) and
[test_checkpoint.py:5-20](tests/test_checkpoint.py#L5-L20) asserts a **bit-exact** round trip
(`assert torch.eq(f0, flow.f).all()`) on a 512×512 Poiseuille flow. A grader can pin one initial
state across candidate implementations. `HDF5Reporter` / `LettuceDataset`
([util/datautils.py](lettuce/util/datautils.py)) additionally serialise `f` per interval together
with pickled flow + collision metadata and a version stamp, exposed as a
`torch.utils.data.Dataset`.

**Failure detection is first-class** — important when a grader must distinguish "reflective" from
"blew up". `NaNReporter` and `HighMaReporter`
([failure_reporter.py:131-183](lettuce/ext/_reporter/failure_reporter.py#L131-L183)) share a base
class, log the top-`k` offending nodes with coordinates, optionally dump a VTK frame, and abort the
run via `BreakableSimulation` ([_simulation.py:325-344](lettuce/_simulation.py#L325-L344)).
`reporter.failed_iteration` is a clean scalar, asserted on directly at
[test_nan_reporter.py:5-16](tests/reporter/test_nan_reporter.py#L5-L16).

**CI.** [.github/workflows/CI.yml](.github/workflows/CI.yml) runs on every PR over a 6-way matrix
(ubuntu × cpu/cu124/cu126/cu128/cu130, macos × cpu), executing `pytest tests` *and* the two
integration commands `lettuce --no-cuda convergence --use-no-cuda_native` and
`lettuce --no-cuda benchmark --use-no-cuda_native`. So the convergence grader is proven to run in a
clean CPU-only container — which matters, because a benchmark harness cannot assume a GPU.

**Test fixtures are a ready-made sweep matrix.** [tests/conftest.py](tests/conftest.py) provides
`fix_stencil` (D1Q3, D2Q9, D3Q15, D3Q19, D3Q27 — [L18-47](tests/conftest.py#L18-L47)), `fix_dtype`
(f32/f64), `fix_device`, and the combined `fix_configuration`
([L66-84](tests/conftest.py#L66-L84)) which yields device × dtype × native with the impossible
CPU+native combination filtered out and auto-skips CUDA when unavailable
([L181-186](tests/conftest.py#L181-L186)). Plus two throwaway flow harnesses — `TestFlow`
([L195-232](tests/conftest.py#L195-L232), with a *settable* `boundaries` property) and `DummyFlow`
([L243-266](tests/conftest.py#L243-L266)) — so an agent testing a new boundary in isolation does not
have to invent a flow.

---

## A6. Pedagogical value — the examples read like a tutorial

`examples/` is ordered and prose-annotated:

* [`00_simplest_TGV.py`](examples/00_simplest_TGV.py) — **21 lines**, the entire API surface:
  `Context → Flow → Simulation → simulation(num_steps)` returning MLUPS.
* [`01b_first_example_obstacle.py`](examples/01b_first_example_obstacle.py) — 107 lines, about half
  of which are triple-quoted teaching blocks explaining Reynolds vs. Mach, why the mask is boolean,
  why `imshow` needs a transpose
  ([L52-57](examples/01b_first_example_obstacle.py#L52-L57)), and BGK vs. KBC. It also states the
  exact constraint an NRBC task inherits: *"Native CUDA is currently not supported for the
  anti-bounce-back outlet."* ([L11](examples/01b_first_example_obstacle.py#L11)).
* [`02_converging_obstacle_flow.py`](examples/02_converging_obstacle_flow.py) — the
  run-until-converged loop with a relative-energy-change stopping criterion and an explicit
  `CRASHED!` branch ([L58-72](examples/02_converging_obstacle_flow.py#L58-L72)).
* [`03_outputs_TGV.py`](examples/03_outputs_TGV.py) — reporters, `np.save`, plotting a slice.
* Notebooks for Couette, Poiseuille, DecayingTurbulence, Obstacle, LidDrivenCavity, MixingLayer,
  PorousMedium ([simple_flows](examples/simple_flows), [advanced_flows](examples/advanced_flows)).
* [`examples/advanced_projects/efficient_bounce_back_obstacle/`](examples/advanced_projects/efficient_bounce_back_obstacle)
  — a full research project: three custom bounce-back boundaries (185–254 LOC each), a custom flow
  (590 LOC), custom reporters, and **24 CSV files of published reference profiles** from four
  papers ([profile_reference_data](examples/advanced_projects/efficient_bounce_back_obstacle/profile_reference_data)).
  This is an existence proof that "add a boundary condition + a validation case + reference data" is
  a *native* shape of work in this repo, not something bolted on.

**Docstrings cite the literature with page numbers**, giving an agent a checkable anchor:

> *"Based on equations from page 195 of 'The lattice Boltzmann method' (2016 by Krüger et al.)"*
> — [anti_bounce_back_outlet.py:14-20](lettuce/ext/_boundary/anti_bounce_back_outlet.py#L14-L20)
>
> *"first presented by Noble and Torczynski (1998), see Krüger et al., pp. 448."*
> — [partially_saturated_boundary.py:7-13](lettuce/ext/_boundary/partially_saturated_boundary.py#L7-L13)

The corresponding test then *re-derives the same formula by hand* and compares
([test_antibounceback_outlet_bc.py:6-103](tests/boundary/test_antibounceback_outlet_bc.py#L6-L103)) —
a ready-made template for "prove your NRBC implements the equation you claim it does".

---

## A7. Honest caveats that are also *features* for a benchmark

None of these disqualify the codebase; several are what make the task non-trivial and therefore
discriminating between models.

1. **The two outlet BCs are only weakly verified.** `test_equilibrium_pressure_outlet` is
   `pytest.skip`ped outright with *"TODO rtol is too big or simulation time too short!"*
   ([test_equilibrium_pressure_outlet.py:5](tests/boundary/test_equilibrium_pressure_outlet.py#L5)),
   and `test_equilibrium_outlet_p_algorithm` passes only at `rel=1e-2` with the comment
   *"TODO rel is too big!"*
   ([test_equilibrium_bc_outlet_p.py:30-31](tests/boundary/test_equilibrium_bc_outlet_p.py#L30-L31)).
   There is a genuine, maintainer-acknowledged gap for an NRBC to fill.
2. **Boundaries return the full `f` tensor** and the simulation selects nodes with `torch.where`
   ([_simulation.py:288-304](lettuce/_simulation.py#L288-L304)). Memory-hungry, but it means a
   candidate cannot "cheat" by touching only its own nodes — the semantics are uniform and easy to grade.
3. **Two coexisting boundary APIs.** `Flow.boundaries` is deprecated in favour of
   `pre_boundaries` / `post_boundaries` and emits a warning
   ([_flow.py:103-120](lettuce/_flow.py#L103-L120)), yet several in-tree flows and tests still use
   the old one. An agent must notice this — a good discriminator.
4. **`EquilibriumOutletP` copies ~40 lines of `AntiBounceBackOutlet.__init__` verbatim** *after*
   calling `super().__init__`
   ([equilibrium_outlet_p.py:18-61](lettuce/ext/_boundary/equilibrium_outlet_p.py#L18-L61)). A strong
   candidate will factor this out; a weak one will copy it a third time.
5. **`Flow.dump` / `load` persist only `f`** — not `i`, not `units`, not resolution. The commented-out
   block at [test_checkpoint.py:22-53](tests/test_checkpoint.py#L22-L53) documents the wish for more.
6. **No global seeding** anywhere (`torch.manual_seed` is never called); determinism relies on the
   flows themselves being deterministic. Fine for every flow but `DecayingTurbulence`, which must be
   given `randseed` explicitly.
7. **The runtime figures in Part B §7 are analytic** (grid-point / population counts), not measured:
   torch was not importable in this survey environment.

---

## A8. Summary scorecard

| Property an autoresearch substrate needs | Status |
|---|---|
| Small enough to read entirely | Yes — 5.3 kLOC source, 109 LOC worst-case BC |
| Single documented extension point for the task | Yes — `Boundary` ABC, 3 real methods |
| A near-relative to copy from | Yes — two outlet BCs, one literally named "outlet" |
| Physics separated from execution | Yes — `UnitConversion` is torch-free |
| Analytical reference solutions | Yes for TGV / Poiseuille / Couette — but **not** acoustic (see Part B gaps) |
| Scalar time series for grading | Yes — `ObservableReporter(out=None)` |
| Order-of-accuracy harness | Yes — `lettuce convergence`, already in CI |
| Deterministic / checkpointable | Yes — fp64, explicit seeds, bit-exact `dump`/`load` |
| Failure detection | Yes — `NaNReporter`, `HighMaReporter`, `BreakableSimulation` |
| CPU-only CI proven | Yes — 6-way matrix including macOS CPU |
| Parametrised fixtures reusable as a sweep | Yes — `fix_stencil`, `fix_configuration`, `TestFlow` |
| Tutorial-grade examples | Yes — `00`…`03`, 7 notebooks, one full research project |
