# Confirmed bug: `pre_boundaries` silently breaks streaming

**Status: reproduced and verified by running the code.** This is a defect in
`lettuce` at `121359d`, independent of the autoresearch question. It was found
during the friction audit ([`02`](02_agent_friction_audit.md) §3.1) and is
confirmed here with an independent reproduction. It is worth filing as an issue
regardless of what happens to the benchmark.

## The defect

[`lettuce/_simulation.py:104-107`](../lettuce/_simulation.py#L104-L107):

```python
self.no_collision_mask = self.context.full_tensor(
    flow.resolution, self.collision_index, dtype=torch.uint8)
self.no_streaming_mask = self.context.full_tensor(
    [flow.stencil.q, *flow.resolution], self.collision_index, dtype=torch.uint8)
```

Filling `no_collision_mask` with `collision_index` is correct and deliberate —
it is an integer field of boundary *ownership indices*, documented in place.

Filling **`no_streaming_mask`** with `collision_index` is not. That mask is
boolean in intent — the comment at
[`_simulation.py:93`](../lettuce/_simulation.py#L93) calls it "a boolean mask" —
it is accumulated with `|=`, and it is consumed at
[`_simulation.py:253-254`](../lettuce/_simulation.py#L253-L254) as:

```python
self.flow.f[i] = torch.where(torch.eq(
    self.no_streaming_mask[i], 1), self.flow.f[i], new_fi)
```

Since `collision_index == len(pre_boundaries)`, the initial fill value is 0 only
when there are no pre-boundaries.

## Reproduction

Measured, CPU, fp64:

| `n_pre` | `n_post` | `collision_index` | mask values | whole domain frozen? | BC's no-streaming request honoured? |
|---:|---:|---:|---|---|---|
| 0 | 1 | 0 | `[0, 1]` | no | yes |
| 1 | 0 | 1 | `[1]` | **YES** | (moot) |
| 1 | 1 | 1 | `[1]` | **YES** | (moot) |
| 2 | 0 | 2 | `[2, 3]` | no | **NO — silently ignored** |
| 2 | 1 | 2 | `[2, 3]` | no | **NO — silently ignored** |
| 3 | 0 | 3 | `[3]` | no | **NO — silently ignored** |
| 3 | 1 | 3 | `[3]` | no | **NO — silently ignored** |

Two distinct failures, both silent:

- **Exactly one pre-boundary** → the mask is `1` everywhere, so `_stream()`
  keeps the old populations at every node. The simulation collides but never
  streams.
- **Two or more pre-boundaries** → `2 | True == 3`, and `torch.eq(3, 1)` is
  `False`, so a boundary's genuine no-streaming request is discarded.

Physical confirmation of the first case — identical 1-D acoustic pulse, the only
difference being which list the outlet is registered in:

| registered in | peak position, t=0 | peak position, t=200 | peak amplitude after 200 steps |
|---|---|---|---|
| `post_boundaries` | 100 | **215** | 4.692727e-04 |
| `pre_boundaries` | 100 | **100** | **1.000000e-03 — bit-identical to the initial condition** |

The field does not evolve at all. Because the flow is initialised in
equilibrium, and collision has equilibrium as its fixed point, freezing the
streaming step freezes everything — with no error and no warning.

## Why it is latent

Every flow shipped in `lettuce/ext/_flows/` uses only `post_boundaries`: nine
`post_boundaries` definitions, zero `pre_boundaries`. No test exercises the
`pre_boundaries` path. The full suite is green (400 passed, 884 skipped) and
would stay green with this bug in place.

## Why it matters for the autoresearch question

`pre_boundaries` is the natural-sounding place to put a boundary condition
applied before collision, and the `Simulation` docstring explicitly describes
the pre/post split as a supported choice. An agent that registers its new
boundary condition there, implements `make_no_streaming_mask` correctly, and
validates it will get a result that is *stable, finite, plausible-looking and
completely wrong* — and no amount of competence will find the cause, because
the failure is in framework code the task never pointed at.

That is the worst possible failure mode for a benchmark: it scores the model on
a bug it cannot reasonably be expected to find, and it does so non-deterministically
depending on an arbitrary API choice.

## Disposition

**Fix before publishing.** The fill value for `no_streaming_mask` should be `0`,
or the tensor should simply be `dtype=torch.bool`:

```python
self.no_streaming_mask = self.context.zero_tensor(
    [flow.stencil.q, *flow.resolution], dtype=torch.bool)
```

This has not been applied here — this branch deliberately contains no changes to
`lettuce/` itself. If the maintainers prefer to leave it, then **every task
prompt must state "register your boundary condition in `post_boundaries`"**, and
the grader must not rely on `no_streaming_mask` behaving correctly. The prompts
in [`prompts/`](prompts/) currently assume `post_boundaries` throughout, but they
say so only implicitly; make it explicit if the bug stays.

## Reproduction scripts

`verify_prestream_bug.py` and `verify_mask_values.py` in the session scratchpad
(not committed). Both run in a few seconds on CPU.
