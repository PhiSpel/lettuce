# Task HARNESS — build the measurement, not the boundary condition

> Orthogonal to the NRBC tasks. Here the agent is given a *working* and a
> *broken* boundary condition and asked to build the instrument that tells them
> apart. This isolates experimental design from physics implementation.
>
> **Why this exists as a separate task:** in the combined form ("build the BC
> *and* the benchmark"), the agent writes both the ruler and the thing being
> measured, and two submissions produce two incomparable numbers. Splitting it
> makes each half gradeable.

---

## Prompt

You are working in `lettuce`, a PyTorch lattice Boltzmann CFD package.

`lettuce` ships two outlet boundary conditions, `AntiBounceBackOutlet` and
`EquilibriumOutletP`. It is claimed that both of them reflect acoustic waves
badly — that neither is meaningfully better than simply leaving the domain
periodic. Nobody has ever measured this: there is no reflection diagnostic, no
pressure-based observable, and no acoustic test case anywhere in the repository.

**Build the measurement.** Deliver a harness that quantifies how much of an
acoustic wave a given outlet boundary condition reflects, and use it to settle
the claim.

### Deliverables

1. A test case: a flow class with a localised pressure disturbance and an outlet
   whose boundary condition is a parameter, so any `Boundary` can be dropped in.
2. A diagnostic that reduces a run to a **single scalar** measuring reflection.
   Justify your choice of scalar.
3. A reference against which "no reflection" is defined. State what makes it
   trustworthy.
4. A report: your scalar for `AntiBounceBackOutlet`, for `EquilibriumOutletP`,
   and for the domain with no outlet at all, plus your verdict on the claim.
5. The harness must be runnable as one command and must print the table itself.

### Constraints

- CPU, `float64`. The whole harness must run in **under five minutes on one
  core** — this is a hard requirement, not a preference.
- Do not modify the existing boundary conditions.
- Your scalar must be able to rank boundary conditions from "absorbs
  everything" to "reflects everything". State the value it takes at each
  extreme, and *demonstrate* both extremes rather than asserting them.

### What makes this harder than it looks

Three specific ways a reflection measurement can be silently wrong. You are not
told how to avoid them; recognising them is part of the task.

- A domain with no boundary condition at all is **periodic**, not absorbing.
  A wave that leaves one side re-enters the other. Any diagnostic that cannot
  distinguish "absorbed" from "left and came back later" is broken.
- Where you place your probe, and when you look at it, determines what you are
  able to see at all. Some choices make two physically different outcomes
  arrive at the same instant.
- A reference that is itself contaminated by reflection is worse than no
  reference.

### Things you need to know about this codebase

- Pin the context: `lt.Context(torch.device('cpu'), dtype=torch.float64,
  use_native=False)`. A bare `lt.Context()` selects CUDA and native codegen.
- Register boundary conditions in your flow's `post_boundaries`, not
  `pre_boundaries`. The `pre_boundaries` path is currently broken in the
  framework and silently disables streaming.
- Streaming is `torch.roll`, so the domain is periodic by default; a boundary
  takes effect only where `make_no_streaming_mask` blocks the roll.
- There is **no pressure-based observable** in `lettuce`. `Observable` /
  `ObservableReporter` in `lettuce/ext/_reporter/observable_reporter.py` is the
  extension point; the five shipped observables are velocity- and energy-based.
- `Enstrophy` and `torch_gradient` assume periodicity — they use `roll` — so
  they are wrong near a non-periodic face. Do not build your diagnostic on them.
- `Mass` hard-codes a 2-D interior slice and is wrong in 1-D and 3-D.
- The environment may need `MKL_THREADING_LAYER=GNU` set.

---

## Grader notes (not shown to the agent)

**Grade by separation, not by agreement with our number.** A good harness need
not reproduce our `R_L2`; it must resolve boundary conditions that genuinely
differ. Instantiate the submitted harness on four boundary conditions whose true
behaviour we have measured:

| BC | our `R_L2` | role |
|---|---|---|
| no boundary (periodic) | 1.1319 | reflects everything, by re-entry |
| `EquilibriumOutletP` | 1.1477 | reflects everything |
| `AntiBounceBackOutlet` | 1.1500 | reflects everything |
| convective prototype | 0.0047 | absorbs |

**Pass** if the harness separates the absorbing case from the three reflecting
cases by at least a factor of 20, *and* its scalar for the periodic case is in
the reflecting cluster rather than the absorbing one.

**Do not grade on ranking all four.** `AntiBounceBackOutlet` and
`EquilibriumOutletP` are separated by 0.2 % — measured across six configs, the
gap never exceeds 0.4 % and its sign is not even stable (at `A = 1e-2` the order
flips). There is no true ordering between them to recover, and requiring one
would grade noise. This corrects the design originally proposed in
[`../01_idea_assessment.md`](../01_idea_assessment.md) §N-A.

**The decisive check is the periodic case.** A harness scoring periodic as
"absorbing" has built a windowed peak-ratio diagnostic, which is the natural
first idea and is wrong: measured, periodic scores a *perfect* `0.0000` on peak
ratio while scoring `1.13` on a full-trace norm. Expect this to be the single
most common failure, and note in the write-up whether the agent discovered it
unprompted — that is the most interesting signal this task produces.
