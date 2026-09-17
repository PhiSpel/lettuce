# Task NRBC-T1 — a non-reflecting outlet for lattice Boltzmann (1-D acoustics)

> Hand this to the agent verbatim. Do not add context. Do not name a method.

---

## Prompt

You are working in `lettuce`, a PyTorch lattice Boltzmann CFD package. Your task
is to implement a **non-reflecting outlet boundary condition** and to
demonstrate that it works.

### The problem

When an acoustic wave reaches the edge of a finite simulation domain, a poor
outlet boundary condition reflects it back into the domain, where it
contaminates the solution. A good outlet lets the wave leave as if the domain
continued forever.

`lettuce` currently ships two outlet boundary conditions,
`AntiBounceBackOutlet` and `EquilibriumOutletP`. **Both of them reflect almost
all of an incident acoustic pulse.** Measured on the benchmark below they score
`R_L2 = 1.150` and `R_L2 = 1.148`, which is no better than removing the
boundary condition entirely (`R_L2 = 1.132`). Your job is to do substantially
better.

### What to deliver

1. A new boundary condition class in `lettuce/ext/_boundary/`, exported from
   that package's `__init__.py`, implementing the `Boundary` interface defined
   in `lettuce/_flow.py`.
2. A flow class for the benchmark case described below, so the result is
   reproducible.
3. A short `RESULTS.md` reporting your measured `R_L2` for your boundary
   condition and for the two existing outlets, and explaining what your
   boundary condition does and why it works.

Your class's constructor must take `(direction, flow)` as its first two
positional arguments, with `direction` a list like `[1]` or `[-1]` (or
`[1, 0]`, `[0, -1]` in 2-D), matching the convention of the existing outlets.
Any further arguments must have defaults. The grader constructs your class with
exactly those two arguments.

### The benchmark

1-D, `D1Q3`, BGK collision, `float64`, CPU.

| parameter | value |
|---|---|
| domain length `N` | 2000 |
| initial condition | density `1 + A·exp(-(x-x0)²/(2σ²))`, velocity `U0` |
| `x0` | 400 |
| `σ` | 10 |
| `A` | 1e-3 |
| `U0` | 0 |
| `τ` | 0.6 |
| probe position `x_p` | 700 |
| steps `T` | 5600 |

Place your outlet on **both** ends of the domain. Record the density
perturbation `p'(t) = ρ(x_p, t) − 1` at every step.

### The metric

Run the identical case in a domain of `N_ref = 9000` with **no** boundary
conditions (periodic). In that domain nothing ever returns to the probe within
`T` steps, so its probe trace `p'_ref(t)` is the exact "infinite domain" answer.

```
R_L2 = ||p'_test − p'_ref||₂ / ||p'_ref||₂
```

Lower is better; `0` is perfect. **Target: `R_L2 < 0.05`**, i.e. at least a
20× improvement on the existing outlets.

Do not score yourself on the ratio of reflected to incident pulse height. That
measure is degenerate: a domain with no boundary condition at all scores a
perfect zero on it, because the wave wraps around rather than being absorbed.

### Rules

- Do not modify `AntiBounceBackOutlet` or `EquilibriumOutletP`.
- Do not modify the metric, the benchmark parameters, or the reference domain.
- Your boundary condition must be a genuine boundary treatment. It must not
  detect the test case, special-case the probe, or damp the entire interior.
- It will be re-graded on configurations you have not seen, including different
  `σ`, `A`, `τ` and **non-zero mean flow `U0`**. Do not tune to the numbers
  above.

### Things you need to know about this codebase

These are infrastructure facts, not part of the problem. They are stated so you
do not have to discover them.

- **Pin the context**: `lt.Context(torch.device('cpu'), dtype=torch.float64,
  use_native=False)`. A bare `lt.Context()` selects CUDA and native codegen,
  and no outlet boundary condition here has a native implementation.
- **You do not need to write any CUDA or C++.** Return `False` from
  `native_available()` and `pass` from `native_generator()`, as both existing
  outlets do.
- **Streaming is `torch.roll`, so the domain is periodic by default**
  (`lettuce/_simulation.py`). A boundary condition only takes effect where
  `make_no_streaming_mask` blocks the roll. If you get that mask wrong your
  boundary silently does nothing and the domain stays periodic — check for this
  explicitly, it is the most common way to get a wrong result that looks right.
- **`no_collision_mask` is not a boolean mask.** It is a `uint8` field of
  *ownership indices*: each node is owned by exactly one boundary or by the
  collision step. A node your boundary owns does **not** get the normal
  collision applied. If your scheme needs collision at the boundary node you
  must do it yourself (see `PartiallySaturatedBC` for precedent).
- **`Flow.post_boundaries` is a property, and `Simulation.__init__` reads it
  twice.** If it constructs new objects on each access, a boundary condition
  that stores history between steps will silently lose it. Cache the list on
  the flow.
- A "perfectly matched layer" cannot be expressed through this `Boundary`
  interface, because a `Boundary` cannot modify the streaming step over a
  region. Do not spend time attempting one.
- The environment may need `MKL_THREADING_LAYER=GNU` set.

---

## Grader notes (not shown to the agent)

**Pass condition.** Instantiate the submitted class via its `(direction, flow)`
constructor on held-out configs. Compute `R_L2` on each. Pass if the **median**
`R_L2 < 0.05` and **no** config exceeds `0.15` or produces a non-finite value.

**Held-out configs.** Vary at minimum `U0 ∈ {0, 0.05, 0.10}`, `σ ∈ {4, 10}`,
`τ ∈ {0.55, 0.6}`, `A ∈ {1e-3, 1e-2}`. All four axes were measured; see
[`../06_robustness_sweep.md`](../06_robustness_sweep.md). Two findings matter
for grading:

- **`τ` is the most discriminating axis tested** — it degraded the prototype by
  34 % (`0.0047 → 0.0063`). Include `τ = 0.55`.
- **Mean flow is a weaker discriminator than expected.** `U0 = 0 → 0.10` moved
  the prototype only `0.0047 → 0.0048`, because `U0` is at most ~17 % of `c_s`
  at these magnitudes, so assuming a wave speed of `c_s` instead of `u ± c_s`
  costs little. Keep `U0` in the sweep as a correctness check, but do not rely
  on it to separate a physically-reasoned scheme from a copied formula.

The baselines move across configs too (`0.96 – 1.23`), which is why the
threshold is set on the submission's own score rather than as a ratio to a
per-config baseline.

**Anti-gaming checks, in order of importance.**

1. Re-run with the probe at a different position. A submission that special-cases
   the probe dies here.
2. Re-run the *interior* against the reference over the whole field, not just at
   the probe. A submission that damps the bulk shows up as a mismatch before the
   wave ever reaches the outlet.
3. Confirm the boundary is actually active: assert the returned
   `make_no_streaming_mask` is not all-zero.
4. Diff against the two existing outlets to catch a copy.

**Reference values** (measured, CPU, fp64, torch 2.0.1 — see
[`../05_measured_baselines.md`](../05_measured_baselines.md)):

| BC | `R_L2` |
|---|---|
| no boundary (periodic) | 1.1319 |
| `AntiBounceBackOutlet` | 1.1500 |
| `EquilibriumOutletP` | 1.1477 |
| convective-outlet prototype (~60 lines) | 0.0047 |

The prototype establishes that `R_L2 < 0.05` is comfortably reachable. The
target is set at `0.05` rather than at the prototype's `0.0047` to leave room
for schemes that trade a little accuracy for robustness under mean flow.

**Runtime.** Reference plus four candidates ≈ **10 s on one CPU core**. A full
held-out sweep of 24 configs is a few minutes. No GPU needed.
