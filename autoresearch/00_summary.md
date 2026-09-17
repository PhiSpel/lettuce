# Publishing an autoresearch question from `lettuce`

**Verdict: the idea holds — with three changes to how it is posed, one of which
is load-bearing and was only discoverable by measurement.**

This directory assesses whether "implement a non-reflecting boundary condition
(NRBC) for `lettuce`" can be published as a reusable autoresearch question, and
if so how it must be specified so that another researcher can run it against
their own model or prompts and get a number that means something.

---

## The three requirements

| requirement | verdict | evidence |
|---|---|---|
| **feasible** | **yes** | A ~60-line prototype reaches `R_L2 = 0.0047` against a best-existing-BC baseline of `1.148`. Written and run, not estimated. |
| **measurable** | **yes, but only with the right metric** | A full-trace `R_L2` against an oversized reference separates working from broken by 150–250× across six configs. The obvious alternative metric is silently gameable — see below. |
| **replicable** | **yes, unusually so** | The entire grader runs in **~10 s on one CPU core**, and reproduces **to every printed digit** across Python 3.11/torch 2.0.1 and Python 3.12/torch 2.8.0 ([`05`](05_measured_baselines.md) §0a). No GPU, no overnight run. |

## The finding that changes the design

The natural way to measure reflection is the ratio of the reflected pulse's
height to the incident pulse's height at a probe. Measured:

| boundary condition | peak ratio | full-trace `R_L2` |
|---|---|---|
| **no boundary condition at all (periodic)** | **0.0000** | 1.1319 |
| `AntiBounceBackOutlet` | 0.5557 | 1.1500 |
| `EquilibriumOutletP` | 0.5542 | 1.1477 |
| convective prototype | 0.0016 | **0.0047** |

**A domain with no boundary condition scores a perfect zero on the peak ratio**,
beating a genuinely working NRBC. Nothing was absorbed — the wave wrapped around
the periodic domain and was scheduled to arrive outside the scoring window. An
agent optimising against that metric is rewarded for deleting the boundary
condition.

This is the single most important design constraint on the question, it is not
visible from reading the code, and it would have quietly invalidated the
benchmark. **Grade on `R_L2` against a regenerated oversized reference.**

A second, similar trap: placing the probe at the domain midpoint makes
wrap-around and reflection arrive at the *same instant*, which flattens every
score to within 0.5 % of every other. The probe must be off-centre. Both traps
are documented with numbers in [`05_measured_baselines.md`](05_measured_baselines.md) §2.

## The headline result, which is also the sales pitch

> **Both outlet boundary conditions currently in `lettuce` are, on an acoustic
> reflection metric, indistinguishable from having no boundary condition at all.**

`R_L2`: periodic `1.1319`, `AntiBounceBackOutlet` `1.1500`, `EquilibriumOutletP`
`1.1477`. This holds in every one of six configurations tested
([`06`](06_robustness_sweep.md)) — the two real BCs are consistently a hair
*worse* than the empty domain.

The repository already half-knows this. `tests/boundary/test_equilibrium_pressure_outlet.py:5`
is a module-level `pytest.skip("TODO rtol is too big or simulation time too
short!")`, and the assertions it disables demand that pressure at the outlet
faces stay below `1e-4` — which is an outlet-quality acceptance test that the
current BC cannot pass. A sibling test passes only at `rel=1e-2` with the
comment `# TODO rel is too big!`.

So the question is not a synthetic exercise. It is an open defect with a
disabled test sitting next to it, and a solution would be a real contribution to
the project. That is worth stating when publishing: it is what separates this
from a toy benchmark.

## A blocker that must be fixed first

The friction audit turned up a genuine bug in `lettuce`, which I reproduced
independently ([`07`](07_confirmed_bug_pre_boundaries.md)):

> **Registering a boundary condition in `pre_boundaries` silently disables
> streaming.** With exactly one pre-boundary, `no_streaming_mask` is initialised
> to `1` everywhere, so `_stream()` keeps the old populations at every node. The
> simulation collides but never streams. With two or more, a boundary's genuine
> no-streaming request is silently discarded instead.

Measured: an identical acoustic pulse advances from node 100 to node 215 when
the outlet is registered in `post_boundaries`, and does not move *at all* —
amplitude bit-identical to the initial condition after 200 steps — when it is
registered in `pre_boundaries`.

It is latent because all nine shipped flows use `post_boundaries` only and no
test touches the other path, so the suite is green. But `pre_boundaries` is the
natural place to put a boundary applied before collision, and an agent that puts
one there gets a stable, finite, plausible-looking, completely wrong answer. That
is the worst failure mode a benchmark can have: it scores the model on a
framework bug it cannot reasonably find. **Fix it, or the prompts must mandate
`post_boundaries`** — they now do so explicitly.

### The other friction worth knowing about

From [`02`](02_agent_friction_audit.md), which built its own environment and
verified most findings by execution:

| finding | why it matters for a benchmark |
|---|---|
| `lt.Context(use_native=False)` silently selects `cuda:0` | The natural way to ask for "CPU, no codegen" fails much later with a traceback that never mentions `Context`. Prompts must pass the device explicitly. |
| Bare `uv sync` installs an unimportable torch | `--extra cpu` is required and appears **only in CI**, never in the README. |
| `Boundary`'s five abstract methods have **zero docstrings** | Undocumented contract: `__call__` must return a full-domain `f`; the two mask methods take differently-shaped `shape` arguments under the same parameter name. |
| A second, competing boundary framework in `examples/advanced_projects/` | Newest code, best docstrings, a third boundary category — and it **does not import**. Prime misdirection. |
| `lettuce/base.py` is dead v0.1 API | Never imported, but has the best docstrings in the repo for exactly the two methods an agent must stub out. A decoy in a file named `base.py`. |
| 5 of 7 runnable examples fail or hang on CPU | Several carry copy-pasted advice that is actively wrong. |
| Boundary tests are disabled or loosened to `rel=1e-2` | The subsystem being extended has its strictest tests off, so **the agent cannot calibrate against the suite.** The prompt must hand it a tolerance. |
| 66 % of classes and 81 % of public functions undocumented | Every class in the extension path is undocumented. |

## Three changes to the original idea

1. **Split "build the benchmark" from "build the boundary condition."** Asked
   together, the agent writes both the ruler and the thing measured, and two
   submissions produce two incomparable numbers. Two prompts:
   [`harness_build.md`](prompts/harness_build.md) and
   [`nrbc_t1_acoustic.md`](prompts/nrbc_t1_acoustic.md).

2. **Do not grade the harness on ranking the existing BCs.** An earlier proposal
   was to grade a submitted harness by whether it ranks four known BCs
   correctly. Measured, `AntiBounceBackOutlet` and `EquilibriumOutletP` differ by
   at most **0.38 %**, and the sign of the difference flips between
   configurations. There is no true ordering to recover; requiring one grades
   noise. Grade instead on *separating* the absorbing case from the reflecting
   cluster, and specifically on whether the periodic case is correctly placed
   among the reflectors.

3. **Drop the "machine-operations / HPC" framing, or attach it separately.**
   This task never leaves single-device PyTorch. The claim that CFD is
   interesting because it couples physics, numerics *and* machine operations is
   true of the field but not of this question. `lettuce`'s CUDA-native code
   generation is a genuine third axis, but it belongs in its own task — and note
   that its `Generator` runs `setup.py install` at runtime, which is
   non-hermetic and should be excluded from any graded benchmark.

## On the two original claims

**"`lettuce` is suitable due to its simplicity" — half right, and the other
half is the more interesting selling point.** The package is 5 957 lines of
Python excluding the generated `_version.py` (6 602 with it), plus 2 479 of
tests; the
`Boundary` interface is five methods, two of which are legitimately discharged
with `return False` / `pass`; the smallest complete boundary condition is 32
lines; a new one is registered by adding two lines to an `__init__.py`. That is
genuinely small.

But it is not *simple* in the sense of frictionless, and a benchmark built on the
premise that it is will mismeasure. Streaming is `torch.roll`, so **the domain is
periodic by default** and a boundary with a wrong `no_streaming_mask` silently
does nothing while appearing to run. `no_collision_mask` is not a boolean mask
but a `uint8` ownership index, so an owned node gets no collision, and in 2-D a
corner belongs to whichever face was listed last. `post_boundaries` is a property
that `Simulation.__init__` reads twice, so a stateful boundary constructed there
loses its history — my prototype hit this and had to cache the list on the flow.

The honest framing is: **small enough to read in an afternoon, with real
load-bearing conventions you have to discover.** That is a better autoresearch
substrate than either a trivial codebase or an opaque one. The conventions that
are *infrastructure* rather than *research* are written into the prompts so they
do not become the thing being measured.

**"CFD is interesting because it couples physics, numerics and machine
operations" — true for the field, over-claimed for this question.** What this
question genuinely tests, and a web-app task does not: the agent must decide what
to measure before it can tell whether it has succeeded, and the obvious
measurement is wrong in a way that rewards doing nothing. There is no test suite
that goes green. That is the interesting part, and it is worth foregrounding over
the HPC angle.

## What is in this directory

| file | what it is | measured? |
|---|---|---|
| [`00_summary.md`](00_summary.md) | this file | — |
| [`01_idea_assessment.md`](01_idea_assessment.md) | feasibility analysis, 10 alternative questions scored | **no** — analytic |
| [`02_agent_friction_audit.md`](02_agent_friction_audit.md) | what blocks or misleads an agent | partial |
| [`03_codebase_strengths.md`](03_codebase_strengths.md) | what makes this a good substrate | **no** — analytic |
| [`04_reusable_inventory.md`](04_reusable_inventory.md) | catalogue of reusable assets, and gaps | **no** — analytic |
| [`05_measured_baselines.md`](05_measured_baselines.md) | **the experiments** — baselines, traps, prototype, test-suite state | **yes** |
| [`06_robustness_sweep.md`](06_robustness_sweep.md) | metric stability across six configs | **yes** |
| [`07_confirmed_bug_pre_boundaries.md`](07_confirmed_bug_pre_boundaries.md) | **a real bug in `lettuce`, reproduced** | **yes** |
| [`prompts/`](prompts/) | four task prompts with grader notes | — |

`01`, `03` and `04` were produced by agents that **could not import torch** and
are explicitly analytic. Where they disagree with `05` or `06`, the measurements
win — and in one case they do: the harness-ranking design in `01` is contradicted
by the measured 0.38 % gap.

## Recommended sequence

1. **Fix the `pre_boundaries` bug** ([`07`](07_confirmed_bug_pre_boundaries.md)),
   or make `post_boundaries` mandatory in every prompt.
2. **Document the working install command.** `uv sync --extra cpu` produces a
   correct CPU environment (Python 3.12, torch 2.8.0+cpu) — but `--extra cpu`
   appears only in CI, and a bare `uv sync` yields a torch that cannot be
   imported. Three other install commands in the README do not give a CPU
   development path. None of this is about the model being benchmarked, and all
   of it will be charged to it. See [`02`](02_agent_friction_audit.md) §1.1 and
   [`05`](05_measured_baselines.md) §0.
3. **Also fix `Context`'s device default** — `lt.Context(use_native=False)`
   silently selects `cuda:0` and fails much later with a traceback that never
   mentions `Context`. The prompts work around this by passing the device
   explicitly.
4. **Publish [`harness_build.md`](prompts/harness_build.md) first.** It is the
   cheapest to grade and it validates the grading infrastructure before any
   physics is at stake.
5. **Then [`nrbc_t1_acoustic.md`](prompts/nrbc_t1_acoustic.md)**, the flagship —
   fully measured, thresholds calibrated, ~10 s to grade.
6. **Calibrate [`nrbc_t2_2d.md`](prompts/nrbc_t2_2d.md) before publishing it.**
   Its thresholds are extrapolated from 1-D and its reference-domain cost is not
   yet measured; at the observed ~0.05 MLUPS a `1024²` reference may be too slow
   to be practical.

## What has not been validated

- **Everything is 1-D.** Corners, oblique incidence and vortex transmission are
  untested; the 2-D tier is a design, not a result.
- **One prototype from one method family.** Convective extrapolation works;
  characteristic schemes and sponge layers are untried here.
- ~~Wrong Python.~~ **Resolved** — re-run on Python 3.12.14 / torch 2.8.0+cpu
  with identical results ([`05`](05_measured_baselines.md) §0a).
- **Contamination is unaddressed.** `lettuce` is public on GitHub and the NRBC
  literature is public in closed form. The mitigation is to grade only on the
  held-out metric and never on which method was chosen — but a model that has
  memorised a specific LBM characteristic-BC paper will have an advantage, and
  that should be disclosed alongside any published results.
