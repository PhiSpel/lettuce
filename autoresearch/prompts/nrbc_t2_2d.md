# Task NRBC-T2 — two-dimensional non-reflecting outlet

> The discriminating tier. T1 can be passed by a correct 1-D formula. T2
> additionally requires handling oblique incidence, corners, and the
> transmission of a *vortical* (non-acoustic) structure — three failure modes
> that a 1-D acoustic scheme does not exercise.

**Caveat: unlike T1, the numbers in this task have NOT been measured.** The
validation in [`../05_measured_baselines.md`](../05_measured_baselines.md) is
1-D only. The thresholds below are extrapolations and **must be calibrated
against a prototype before this tier is published.** See "Calibration required".

---

## Prompt

You are working in `lettuce`, a PyTorch lattice Boltzmann CFD package. Implement
a **non-reflecting outlet boundary condition** for two-dimensional flow and
demonstrate that it works on two physically distinct tests.

`lettuce`'s existing outlets, `AntiBounceBackOutlet` and `EquilibriumOutletP`,
reflect essentially all of an incident acoustic wave. Do substantially better,
in 2-D, on all four faces at once.

### Deliverables

1. A boundary condition class in `lettuce/ext/_boundary/`, exported from that
   package's `__init__.py`, implementing the `Boundary` interface in
   `lettuce/_flow.py`. Constructor `(direction, flow, ...)` with `direction` a
   list such as `[1, 0]` or `[0, -1]`; further arguments must have defaults.
2. Flow classes for both benchmark cases below.
3. A `RESULTS.md` with your measured scores for your BC and for both existing
   outlets, and an explanation of the scheme.

### Test A — acoustic pulse, all four faces

`D2Q9`, BGK, `float64`, CPU, domain `256 × 256`, `τ = 0.6`. Initial condition: a
Gaussian density pulse `1 + 1e-3·exp(-((x-x_c)² + (y-y_c)²)/(2σ²))` with
`σ = 8`, centred **off-centre** at `(x_c, y_c) = (96, 112)` so that the wave
reaches the four faces at four different times and strikes the corners
obliquely. Mean flow `U0 = 0`. Your outlet on all four faces. Run 900 steps.

Probe the density perturbation at `(70, 150)` — deliberately not on a symmetry
axis. Score `R_L2` against the same case run in a `1024 × 1024` periodic domain
with the pulse at the same absolute position, over the same number of steps, in
which no reflection can return within the run.

```
R_L2 = ||p'_test − p'_ref||₂ / ||p'_ref||₂
```

**Target: `R_L2 < 0.10`.**

### Test B — vortex leaving the domain

A Lamb–Oseen vortex advected by a mean flow through the outlet. `lettuce` ships
`LambOseenVortex2D`; use it or a case derived from it, with a mean velocity in
`+x` of `U0 = 0.05` in lattice units, an outlet on the `+x` face and your
boundary on the others.

This test measures something different from Test A. The acoustic test asks
whether pressure waves leave cleanly; this asks whether a *vortical* structure —
which is advected, not propagated at the speed of sound — leaves without
generating spurious pressure. A scheme tuned only to acoustics typically emits a
pressure pulse as the vortex crosses the face.

Score the peak spurious density perturbation on the outlet face during the
vortex's passage, relative to the vortex's own core perturbation.

**Both tests must pass.** A submission that passes A and fails B has built an
acoustic absorber, not a non-reflecting outlet.

### Rules

- Do not modify the existing outlets, the metrics, or the benchmark parameters.
- Corners belong to more than one face. Handle them; do not leave them to
  whichever boundary happens to be constructed last.
- Your boundary must not detect the test cases, special-case the probes, or damp
  the interior.
- You will be re-graded on configurations you have not seen, including a
  different domain aspect ratio, a different pulse position, and a different
  mean-flow magnitude.

### Things you need to know about this codebase

- Pin the context: `lt.Context(torch.device('cpu'), dtype=torch.float64,
  use_native=False)`. A bare `lt.Context()` selects CUDA and native codegen.
- No CUDA or C++ required: `native_available() → False`,
  `native_generator() → pass`.
- Streaming is `torch.roll`; the domain is periodic unless
  `make_no_streaming_mask` blocks it. A wrong mask means your boundary silently
  does nothing.
- **`no_collision_mask` is a `uint8` ownership index, not a boolean.** Each node
  is owned by exactly one boundary, and later boundaries in the list overwrite
  earlier ones. **In 2-D this means a corner node is owned by whichever face is
  listed last** — this is the single most likely source of a subtly wrong
  result in this task.
- `Flow.post_boundaries` is a property read twice by `Simulation.__init__`; if
  it rebuilds its objects, a stateful boundary loses its history. Cache it.
- A perfectly matched layer cannot be expressed through this `Boundary`
  interface — it cannot modify streaming over a region. Do not attempt one.
- `Enstrophy` and `torch_gradient` assume periodicity (they use `roll`), so any
  gradient-based diagnostic near an outlet face is wrong by construction. Do
  not use them to assess your boundary.
- The environment may need `MKL_THREADING_LAYER=GNU` set.

---

## Grader notes (not shown to the agent)

**Calibration required before publishing.** Run, in 2-D, at minimum: no
boundary (periodic), `AntiBounceBackOutlet`, `EquilibriumOutletP`, and a
prototype convective outlet. Set the Test A threshold at roughly 10× better than
the best existing BC and confirm the prototype clears it. If periodic and the
existing outlets are not well separated in 2-D, revisit the probe position
first — in 1-D a probe on a symmetry axis made wrap-around and reflection
indistinguishable and flattened every score (see `05_measured_baselines.md` §2a).
The off-centre pulse and off-axis probe above are chosen to avoid that, but this
must be confirmed, not assumed.

**Cost.** 256² × 900 steps with `D2Q9` is ~5.9e7 node-updates; the `1024²`
reference is ~16× that. At the ~0.05 MLUPS measured on a single CPU core with
fp64 this reference is uncomfortably slow and should be either generated once
and cached with a checksum, run at `float32`, or replaced by a non-reflecting
*analytic* reference. **Measure this before publishing** — if the reference
alone costs an hour per grade, the replicability claim fails and Test A should
drop to 128² with a 512² reference.

**Anti-gaming.** As T1, plus: verify corner nodes are handled by checking the
boundary list ordering does not change the score by more than a few percent.
