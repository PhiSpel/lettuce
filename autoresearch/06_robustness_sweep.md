# Robustness sweep — is the metric stable across held-out configs?

Measured, CPU, fp64, torch 2.0.1, `lettuce` @ `121359d`. Companion to
[`05_measured_baselines.md`](05_measured_baselines.md).

The question: if the grader re-scores a submission on configurations the agent
never saw, does `R_L2` stay meaningful, or does it wander enough that a pass/fail
threshold becomes arbitrary?

## Results

`R_L2`, lower is better. Base config is `N=2000`, `x0=400`, `σ=10`, `A=1e-3`,
`x_p=700`, `τ=0.6`, `T=5600`, reference `N_ref=9000`.

| config | periodic | AntiBounceBack | EqOutletP | Orlanski (proto) |
|---|---|---|---|---|
| base, `U0=0` | 1.1319 | 1.1500 | 1.1477 | **0.0047** |
| mean flow `U0=0.05` | 1.1332 | 1.1527 | 1.1495 | **0.0048** |
| mean flow `U0=0.10` | 1.1345 | 1.1556 | 1.1512 | **0.0048** |
| narrow pulse `σ=4` | 0.9553 | 0.9862 | 0.9827 | **0.0050** |
| `τ=0.55` | 1.2163 | 1.2263 | 1.2226 | **0.0063** |
| larger amplitude `A=1e-2` | 1.1325 | 1.1389 | 1.1424 | **0.0047** |

## What this establishes

**The metric is stable.** Across every config the baselines sit in
`0.96 – 1.23` and the working boundary condition sits in `0.0047 – 0.0063`. The
separation is between **150× and 250×**, never less. A threshold at `R_L2 < 0.05`
is nowhere near either cluster, so pass/fail is not sensitive to which configs
the grader happens to pick. This is the property the benchmark needs.

**The existing outlets are uniformly no better than nothing.** In every single
config, `AntiBounceBackOutlet` and `EquilibriumOutletP` score *slightly worse*
than the empty periodic domain. This is not a quirk of one parameter choice.

**A working scheme is robust for free.** The prototype was tuned on nothing —
it uses `c = c_s` with no fitting — and it degrades by at most 34 % (`0.0047 →
0.0063`) across the whole sweep. So the target threshold does not need to be
generous to be fair.

## Corrections to earlier claims

**Mean flow is a weak discriminator, not a strong one.** The T1 grader notes
originally asserted that non-zero `U0` was "the discriminating axis", on the
reasoning that a scheme assuming wave speed `c_s` rather than `u ± c_s` should
degrade there. Measured, it barely does: `0.0047 → 0.0048` from `U0 = 0` to
`U0 = 0.10`. At these magnitudes `U0` is only ~17 % of `c_s`, so the error in
the assumed wave speed is small. To make mean flow genuinely discriminating the
sweep would need `U0 ≳ 0.2` in lattice units, which starts to strain the low-Mach
assumption. **`τ` is the better discriminator of the axes tested** — it produced
the only visible degradation (+34 %). This is corrected in
[`prompts/nrbc_t1_acoustic.md`](prompts/nrbc_t1_acoustic.md).

## Limits

- Still 1-D. Oblique incidence and corners are untested; see
  [`prompts/nrbc_t2_2d.md`](prompts/nrbc_t2_2d.md).
- One prototype from one method family. A sponge layer or a characteristic
  scheme might have a different robustness profile.
- The sweep re-generates the `N=9000` reference for every cell rather than
  caching it, so the 48 runs took ~4 minutes rather than the ~1 minute they
  need. A published grader should cache the reference per config.

## Reproducing

Scripts are in the session scratchpad, not committed (`refl_probe.py`,
`refl2.py`, `refl3.py`, `refl4.py`). Before publishing, the sweep should be
rewritten as a single committed grader module with the reference cached and a
fixed seed. Run as:

```bash
MKL_THREADING_LAYER=GNU /home/pspelt3s/miniconda3/envs/lettuce/bin/python refl4.py
```
