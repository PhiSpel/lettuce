# Task NRBC-T0 — capability floor

> A cut-down NRBC-T1 for weaker or smaller models. Its purpose is to
> distinguish "cannot navigate the codebase" from "cannot solve the physics".
> Passing T0 says nothing about research ability; **failing** T0 says the model
> never got far enough for T1 to measure anything.

---

## Prompt

You are working in `lettuce`, a PyTorch lattice Boltzmann CFD package.

Implement a new boundary condition class in `lettuce/ext/_boundary/` that
implements the `Boundary` interface from `lettuce/_flow.py`, and export it from
`lettuce/ext/_boundary/__init__.py`.

The boundary sits on one face of the domain, identified by a `direction` list
such as `[1]` (positive x) or `[-1]` (negative x). At that face, the
populations that point **into** the domain must be set by copying the
corresponding populations from the neighbouring node one cell inside the
domain, rather than by the default periodic wrap-around.

Constructor signature: `__init__(self, direction, flow)`. Any further arguments
must have defaults.

Verify it by running this case and confirming the simulation completes with no
`NaN` or `Inf`:

- `D1Q3`, BGK, `float64`, CPU, domain length 2000, `τ = 0.6`
- initial density `1 + 1e-3·exp(-(x-400)²/200)`, zero velocity
- your boundary on both ends, 5600 steps

You need to know:

- Pin the context: `lt.Context(torch.device('cpu'), dtype=torch.float64,
  use_native=False)`.
- You do not need to write CUDA or C++: return `False` from
  `native_available()` and `pass` from `native_generator()`.
- Register your boundary condition in your flow's `post_boundaries`, not
  `pre_boundaries`. The `pre_boundaries` path is currently broken in the
  framework and silently disables streaming.
- Streaming is `torch.roll`, so the domain is periodic unless your
  `make_no_streaming_mask` blocks the roll for the populations you are setting.
  If that mask is wrong, your boundary silently does nothing.
- `make_no_collision_mask` marks the nodes your boundary owns; owned nodes do
  not get the normal collision step.
- The environment may need `MKL_THREADING_LAYER=GNU` set.

---

## Grader notes (not shown to the agent)

Pass requires **all** of:

1. The class imports from `lettuce` and constructs with `(direction, flow)`.
2. The 5600-step run completes, all values finite.
3. `make_no_streaming_mask` returns a tensor that is **not** all-zero — this is
   the check that catches a boundary which is silently inert. Expect this to be
   the most common failure.
4. `R_L2 < 0.5` on the T1 benchmark. Zeroth-order extrapolation is a weak
   non-reflecting condition but it is a real one, and it should land far below
   the existing outlets' `1.15`. A submission scoring `≈1.13` has produced an
   inert boundary and should fail even if it satisfies (3) syntactically.

This task **does** name the method, deliberately. It is a floor test, not a
research question — do not report T0 results as evidence about research
capability.
