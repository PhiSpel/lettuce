# Task prompts

One file per autoresearch task. Each prompt is written to be handed to an agent
**verbatim**, with no additional context, and is designed so the grader can run
without a human in the loop.

| file | task | tier | graded on |
|---|---|---|---|
| [`nrbc_t0_smoke.md`](nrbc_t0_smoke.md) | 1-D convective outlet | T0 — smoke | runs, stable, `R_L2 < 0.5` |
| [`nrbc_t1_acoustic.md`](nrbc_t1_acoustic.md) | 1-D NRBC, held-out configs | T1 — core | `R_L2` on unseen configs |
| [`nrbc_t2_2d.md`](nrbc_t2_2d.md) | 2-D NRBC, four faces + corners | T2 — hard | `R_L2` acoustic **and** vortex exit |
| [`harness_build.md`](harness_build.md) | build the measurement harness | orthogonal | separation of known baselines |

## Design rules these prompts follow

Derived from [`05_measured_baselines.md`](../05_measured_baselines.md), which
contains the measurements that justify each one.

1. **Never state the method.** The prompts say *what must be achieved*, never
   "implement Orlanski" or "use characteristic boundary conditions". Naming the
   method turns a research task into a transcription task, and all of these
   methods are in the public literature.
2. **Grade on `R_L2` against a regenerated reference, never on a windowed peak
   ratio.** Measured: a do-nothing boundary scores a perfect `0.0000` on peak
   ratio and `1.13` on `R_L2`. Peak ratio rewards deleting the boundary.
3. **Grade on held-out configs.** The prompt states the metric and one example
   config; the grader instantiates the submitted class on configs the agent
   never saw (different `sigma`, `U0`, `tau`, amplitude, and in T2 a different
   geometry). This is what stops hard-coding.
4. **State the environment traps, do not make the agent discover them.**
   Device/dtype/native pinning, the `post_boundaries`-read-twice trap and the
   integer `no_collision_mask` are *infrastructure*, not the research question.
   Time spent on them is noise in the measurement.
5. **State what is out of scope.** PML cannot be expressed as a `Boundary` in
   this architecture; CUDA-native codegen is not required. Without these lines
   agents burn entire sessions on them.
6. **Give the baseline to beat as a number.** `R_L2 = 1.148` for the best
   existing BC. An agent that cannot measure its own progress cannot iterate.

## Which model to start with

The stated intent is an open-weight model first. The tiering exists for that
reason: T0 is a capability floor (can the model navigate an unfamiliar ABC at
all), T1 is the real question, T2 separates strong models. Running only T1 on a
weak model produces an uninformative zero.
