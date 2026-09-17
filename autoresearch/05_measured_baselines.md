# Measured baselines — the NRBC autoresearch question, validated by experiment

**Status: these numbers were actually measured**, not estimated. Everything in
this file was produced by running `lettuce` at HEAD `121359d` on CPU. The other
documents in this directory (`01`–`04`) were written by agents that could not
import torch and are explicitly analytic; where this file disagrees with them,
**this file wins**.

---

## 0. The environment that actually works

The default interpreter on this machine cannot run `lettuce`:

| probe | result |
|---|---|
| `python3` (`/home/pspelt3s/miniconda3/bin/python3`) | Python 3.10.11, **no torch** |
| `pyproject.toml` `requires-python` | `==3.12.*` — satisfied by neither interpreter present |
| `uv` | **not installed** |
| conda env `lettuce` | Python 3.11.3, torch 2.0.1, **CUDA not available** |

The conda env `lettuce` works, but only after a workaround — bare invocation
dies before reaching Python:

```
symbol lookup error: .../libmkl_intel_thread.so.2: undefined symbol: __kmpc_global_thread_num
```

Every command below is therefore prefixed `MKL_THREADING_LAYER=GNU`.

```bash
MKL_THREADING_LAYER=GNU /home/pspelt3s/miniconda3/envs/lettuce/bin/python <script>
```

**Consequence for publishing:** the reference environment must be pinned and
shipped, and it must be CPU-only. A researcher who clones this repo and runs
`python script.py` on a machine like this one gets `ModuleNotFoundError` or an
MKL symbol error — neither of which says anything about the model being
benchmarked. See §5.

Smoke test, CPU, fp64: TGV 32² × 200 steps = **0.05 MLUPS, 4.2 s wall**. This
is slow (single-core, fp64, torch 2.0.1), and it is the number the runtime
budget in §4 is based on.

### 0b. The test suite, actually run

```
MKL_THREADING_LAYER=GNU python -m pytest tests -q
400 passed, 884 skipped, 24 warnings in 68.16s
```

**No pre-existing failures.** This matters more than it sounds: an agent that
opens a repository whose suite is already red cannot use "tests pass" as a
signal, and a grader cannot distinguish damage from inherited breakage. Here the
baseline is clean, and the whole suite runs in ~1 minute on CPU. Good.

**But 884 of 1 284 tests — 69 % — are skipped on a CPU-only machine.** Almost
all of them are CUDA or `cuda_native` variants of parametrised cases. The
consequence for benchmark design: *"the test suite still passes" is very weak
evidence that a submitted boundary condition is correct.* Roughly two thirds of
the assertions never execute. The grader must carry its own acceptance test — it
cannot delegate to `pytest`.

Two warnings worth knowing about, both of which an agent will see and may chase:

- `lettuce/util/moments.py:189` and `:246` emit
  `ExperimentalWarning: I am not 100% sure if this equilibrium is correct.`
- `lettuce/_simulation.py:70,73` emit the deprecated-`boundaries` warning during
  `test_partiallysaturated_bc.py` — i.e. **an in-tree test still trips the
  deprecation path**, which is exactly the `post_boundaries`-read-twice code at
  lines 70 and 73 that breaks stateful boundary conditions (§6).

---

## 1. The test case

1-D acoustic pulse, `D1Q3`, BGK, fp64, CPU.

| parameter | value |
|---|---|
| domain `N` | 2000 |
| pulse centre `x0` | 400 |
| pulse width `sigma` | 10 |
| pulse amplitude `A` (density) | 1e-3 |
| probe `x_p` | **700 — deliberately off-centre, see §2** |
| `tau` | 0.6 |
| mean flow `U0` | 0 (and 0.05, 0.10 in the sweep) |
| steps `T` | 5600 |
| reference domain `N_ref` | 9000, periodic, same everything else |

A Gaussian density perturbation splits into a left- and a right-going pulse of
half amplitude each. Outlets are placed on **both** ends. Sound travels
`c_s = 1/sqrt(3) ≈ 0.577` lu/step, so with these numbers:

- incident pulse passes the probe at ≈ **520** steps
- it reaches the outlet at ≈ 2771 steps
- a reflection returns to the probe at ≈ **5023** steps
- in a *periodic* domain the wrapped pulse instead arrives at ≈ **3984** steps

The reference domain is large enough that nothing ever returns: measured tail
after the incident pulse is `4.8e-10` against a peak of `4.31e-4`, i.e. six
orders down. The reference is clean.

---

## 2. Two trap-doors found by measurement

Both of these would have silently wrecked the benchmark, and neither is
visible from reading the code.

### 2a. Do not put the probe at the domain midpoint

First attempt used `x_p = 1000 = N/2`. Then the periodic wrap-around path
`(N-x0) + x_p` and the reflection path `(N-x0) + (N-x_p)` are **the same
length**, so wrap-around and reflection arrive at the same instant and are
indistinguishable. Every case scored the same and the metric looked dead:

```
periodic (no BC)      R_peak 0.6505
AntiBounceBackOutlet  R_peak 0.6495
EquilibriumOutletP    R_peak 0.6475
```

With `x_p = 700` the control separates cleanly (periodic → 0.0000). **The task
spec must fix the probe off-centre**, or every submission is graded on noise.

### 2b. A windowed peak ratio is gameable by doing nothing

This is the important one.

| BC | `R_peak` (windowed) | `R_L2` (full trace vs reference) |
|---|---|---|
| **periodic — i.e. no boundary at all** | **0.0000** | 1.1319 |
| AntiBounceBackOutlet | 0.5557 | 1.1500 |
| EquilibriumOutletP | 0.5542 | 1.1477 |
| Orlanski convective outlet (prototype) | **0.0016** | **0.0047** |

A do-nothing BC scores a **perfect 0.0000** on the windowed peak ratio, beating
the real NRBC. The wave has not been absorbed — it has merely wrapped around
and is scheduled to arrive at 3984 steps, outside the scoring window.

An agent optimising against `R_peak` is rewarded for deleting the boundary.

`R_L2` against an oversized reference domain is not fooled: it sees the whole
trace, so the wrap-around arrival lands in the norm and periodic scores 1.13.
**Grade on `R_L2`. If `R_peak` is reported at all, report it as a diagnostic,
never as the objective.**

---

## 3. The headline result

> **Both outlet boundary conditions currently in `lettuce` are, on an acoustic
> reflection metric, indistinguishable from having no boundary condition at
> all.**

`R_L2`: periodic **1.1319**, `AntiBounceBackOutlet` **1.1500**,
`EquilibriumOutletP` **1.1477**. The two real BCs are marginally *worse* than
the empty domain, and differ from each other by 0.2 %.

Two consequences for the autoresearch design:

1. **The research gap is real and large.** A ~60-line prototype
   ([Orlanski convective outlet](#appendix-prototype)) reaches `R_L2 = 0.0047`
   — a **244× improvement** over the best existing BC. The task is neither
   trivially pre-solved nor out of reach. Headroom of that size is exactly what
   a benchmark wants.

2. **The proposed "rank four BCs correctly" grader does not work.**
   `01_idea_assessment.md` §N-A proposes grading a submitted *harness* by
   whether it ranks four known BCs in the right order. Measured, two of those
   four (`AntiBounceBackOutlet` and `EquilibriumOutletP`) are separated by
   0.2 %, which is inside run-to-run and config-to-config variation. There is no
   true ordering to recover. **Either drop that task, or restate it with
   baselines that are actually separated** — periodic / existing-outlet /
   Orlanski / a deliberately over-damped sponge span 0.0047 → 1.15 and *are*
   separable.

---

## 4. Feasibility and runtime, measured

The whole grader — oversized reference plus four candidate BCs — runs in
**about 10 seconds on one CPU core**:

| case | wall |
|---|---|
| reference, `N=9000`, 5600 steps | 1.8 s |
| periodic | 1.0 s |
| AntiBounceBackOutlet | 3.0 s |
| EquilibriumOutletP | 2.7 s |
| Orlanski prototype | 1.7 s |

No GPU. No overnight run. This is as replicable as a CFD benchmark realistically
gets, and it makes the question cheap enough that a researcher can iterate on
prompts without a compute budget.

The prototype is ~60 lines including the two mask methods. **A competent agent
can do this in one session** — which settles the feasibility requirement in the
affirmative, with evidence rather than argument.

---

## 5. What must be pinned before publishing

| item | why |
|---|---|
| `MKL_THREADING_LAYER=GNU` or a non-MKL torch build | otherwise the interpreter dies before Python starts |
| CPU + `float64` + `use_native=False` | `Context()` with no args auto-selects CUDA and `use_native=True`; neither outlet has a native generator |
| probe position off-centre | §2a — otherwise the metric is degenerate |
| `R_L2` as objective, oversized reference regenerated by the grader | §2b — otherwise doing nothing wins |
| a working Python 3.12 env, since `requires-python = "==3.12.*"` | no interpreter on this machine satisfies it |

---

## 6. Honest limits of this validation

- **1-D only.** 2-D (`D2Q9`) corner handling, oblique incidence and vortex
  transmission are *not* validated here. The first agent's concern that corners
  belong to whichever face is listed last remains untested.
- **One prototype, one family.** Orlanski convective extrapolation works. That
  characteristic-based or sponge-layer approaches also work here is untested.
- **No mean flow in the headline numbers.** `U0 = 0`. The sweep in §7 addresses
  this; acoustics with mean flow is the case where a naive convective BC using
  `c = c_s` instead of `u ± c_s` should degrade.
- **torch 2.0.1 / Python 3.11**, not the declared 3.12 target.
- The prototype BC exposed a real trap: `post_boundaries` is a *property* and
  `Simulation.__init__` reads it twice, so a naive stateful BC is constructed
  twice and loses its state. The prototype had to cache the boundary list on the
  flow. Any NRBC that stores history hits this. Confirms the finding in
  `02_agent_friction_audit.md`.

---

## 7. Robustness sweep

See `06_robustness_sweep.md` — varies `U0 ∈ {0, 0.05, 0.10}`, `sigma`, `tau`
and amplitude, which is the "held-out grading config" design.

---

## Appendix: prototype

`OrlanskiOutlet`, discretising `∂f/∂t + c ∂f/∂n = 0` at the face as
`f_b^{new} = (f_b^{old} + C·f_n) / (1 + C)` with `C = c_s`, applied to the
populations pointing *into* the domain. Source:
`/tmp/.../scratchpad/refl3.py`. It is a throwaway prototype written to prove
headroom exists — **it is deliberately not committed to the repo**, because
publishing a working NRBC alongside the question would destroy the question.
