# Agent Friction Audit — `lettuce`

**Scope.** What in this codebase would block, mislead, slow down, or overburden an autonomous
coding agent asked to *implement a new boundary condition (or similar extension) and validate it*.

**Method.** Worktree `/home/pspelt3s/lettuce/.claude/worktrees/autoresearch-nrbc` at
`121359d` ("270 efficient hwbb and ibb (#287)"). Environment built from scratch with `uv`
(CPU-only Linux box, no NVIDIA driver; `nvidia-smi` fails).

Findings are tagged:

* ***verified*** — reproduced by executing code in that environment. The exact commands and outputs
  are in [Appendix A](#appendix-a--commands-run).
* ***by inspection*** — established by reading source, with file:line evidence, but not executed
  (typically because it needs a GPU, or because running it would mean changing project files).

Nothing outside `autoresearch/` was modified; the virtualenv lives in the session scratchpad.

**Do not fix anything from this document silently** — several findings are genuine physics bugs
that the maintainers must triage first.

---

## Executive summary

### Ranked top blockers

| # | Blocker | Severity | Verdict |
|---|---|---|---|
| 1 | `no_streaming_mask` is initialised to `collision_index` instead of `0`. With **exactly one** `pre_boundaries` entry, streaming is silently disabled on the entire domain. With **≥2**, every boundary's no-streaming request is silently ignored. Only `len(pre_boundaries) == 0` behaves correctly. | Critical — silent wrong physics | **Fix before publishing** |
| 2 | `lt.Context(use_native=False)` (no `device`) silently selects `cuda:0` on a CPU-only machine and then dies inside `torch` with `AssertionError: Torch not compiled with CUDA enabled`. This is the most natural way to ask for "CPU, no codegen". | Critical — blocks setup, misleading error | **Fix before publishing** |
| 3 | Bare `uv sync` installs a `torch` that cannot be imported (`ImportError: libcudnn.so.9`). The working invocation (`uv sync --extra cpu`) appears only in CI, never in the README. | Critical — blocks setup entirely | **Fix before publishing** (or pin the command in the task prompt) |
| 4 | `requires-python = "==3.12.*"` — an exact pin. `pip install -e .` fails on 3.10/3.11/3.13. No Python 3.12 exists on this machine. | High — blocks setup | **State explicitly in the task prompt** |
| 5 | The `Boundary` ABC has **zero docstrings** on all five abstract methods, and the real contract (`__call__` must return a *full-domain* `f`; masking is done by `Simulation`; some shipped boundaries mutate `flow.f` in place instead) is undocumented and internally inconsistent. | High — misleads | **Fix before publishing** (docstrings only) |
| 6 | `examples/advanced_projects/efficient_bounce_back_obstacle/` contains a **second, competing boundary framework** (`HalfwayBounceBackBoundary`, `SolidBoundaryData`, `EbbSimulation`, a third `post_streaming_boundaries` category) that is not importable (`ModuleNotFoundError: No module named 'solid_boundary_data'`) and is the newest code in the repo — exactly what an agent's search will surface first. | High — misleads | **Fix before publishing** or fence off in the prompt |
| 7 | `Context.convert_to_tensor` silently converts `bool` → `torch.uint8`. A mask built with the obvious helper crashes later in `Simulation` with `RuntimeError: masked_fill_ only supports boolean masks`. | Medium-high — loud but confusing | **State explicitly in the task prompt** |
| 8 | 884 of 1278 tests (69%) skip without CUDA; `tests/native/` is 100% skipped. An agent gets **zero** feedback on `native_generator`. | Medium-high — no signal | **State explicitly in the task prompt** |
| 9 | `flow.torch_stencil.opposite` is a **float64 tensor** and unusable as an index, while `flow.stencil.opposite` (a list) works. The two shipped boundaries use different ones. | Medium — loud trap | Leave as-is, name it in the prompt |
| 10 | Documentation is structurally dead: 15 of 17 `automodule` targets point at modules that no longer exist; `docs/contributing.rst` and `docs/history.rst` include missing files; `docs/installation.rst` recommends `python setup.py install` (there is no `setup.py`). | Medium — wasted agent time | **Fix before publishing** or delete `docs/` |

### Must-fix-before-publishing (short list)

1. **`no_streaming_mask` initialisation** ([`_simulation.py:106`](../lettuce/_simulation.py#L106)) — the benchmark is unusable while a correct pre-boundary silently produces wrong physics.
2. **`Context` device default** ([`_context.py:22`](../lettuce/_context.py#L22)) — agents cannot reliably get a CPU context.
3. **A documented, working CPU install command** — README and CI currently disagree, and the README's commands do not work.
4. **Docstrings on the five `Boundary` abstract methods** ([`_flow.py:31`](../lettuce/_flow.py#L31)) — one paragraph each; without them the task is guesswork, not engineering.
5. **Decide what to do with `examples/advanced_projects/.../boundary/`** — either make it importable or exclude it from the agent's view.
6. **Delete or regenerate `docs/modules.rst`** — it is 100% misleading.

Everything else is defensible as "part of the challenge", provided the task prompt says so.

---

## 1. Setup and environment friction

### 1.1 Bare `uv sync` produces an unimportable torch — *verified, critical*

[`pyproject.toml:44`](../pyproject.toml#L44) declares five mutually-conflicting torch extras
(`cpu`, `cu124`, `cu126`, `cu128`, `cu130`) via `[tool.uv] conflicts`. With **no** extra selected,
`uv.lock` resolves `torch==2.8.0` from PyPI but the conflict markers on the `nvidia-*` packages all
evaluate false, so none are installed:

```
$ uv sync            # 30 packages, zero nvidia-* wheels
$ python -c "import lettuce"
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

`uv sync --extra cpu` fixes it (`torch==2.8.0+cpu`) and everything then works. But:

* The README never mentions `uv sync` at all — it documents `uv pip install .` / `".[cpu]"` / `-e .`
  ([README.rst:67–75](../README.rst#L67)).
* `.github/workflows/CI.yml:40` uses `uv sync --extra <x>` — the only place the working command appears.
* README option 3, `uv pip install -e .`, is labelled the *development* option and carries **no extra**,
  so the documented developer path is the GPU path. There is no documented CPU-development command
  (`uv pip install -e ".[cpu]"` is never written down).

**Why it hurts.** This is the very first thing the agent does. A failing `import lettuce` with a
`libcudnn` error reads like a broken machine, not a broken lockfile; agents typically burn many turns
installing CUDA runtimes that the task does not need.

**Mitigation:** *Fix before publishing.* Make the default resolution CPU-safe, or state the exact
command (`uv sync --extra cpu`) in the task prompt.

### 1.2 `requires-python = "==3.12.*"` is an exact pin — *verified*

[`pyproject.toml:10`](../pyproject.toml#L10). Verified failure on Python 3.11:

```
ERROR: Package 'lettuce' requires a different Python: 3.11.3 not in '==3.12.*'
```

No 3.12 interpreter exists on this machine (system `/usr/bin/python3` is 3.9, conda base is 3.10,
the project's own conda env is 3.11). `uv` rescues this by downloading a standalone CPython 3.12 —
but only if the agent uses `uv`. An agent reaching for `pip`/`venv`/`conda` (a very common default)
hits a hard wall with an error that suggests the *package* is at fault.

README's [`uv venv`](../README.rst#L58) with no `--python 3.12` will also pick the system interpreter
and fail on this machine.

**Mitigation:** *State explicitly in the task prompt* — "use `uv`, the project requires Python 3.12".
Relaxing to `>=3.12` would be a kinder fix.

### 1.3 Versioneer is vestigial and reports a version nobody expects — *verified*

* [`pyproject.toml:7`](../pyproject.toml#L7) hard-codes `version = "0.2.3"` (static, used by pip).
* [`lettuce/__init__.py:9`](../lettuce/__init__.py#L9) computes `__version__` from
  `_version.get_versions()` at import time.
* There is **no `setup.py`**, so versioneer's `cmdclass` is never wired into the build. The
  `[tool.versioneer]` block and the 80 KB [`versioneer.py`](../versioneer.py) at the repo root are dead weight.
* `versioneer.py` is version **0.21**; `pyproject.toml` requires `versioneer>=0.28`
  ([`pyproject.toml:3`](../pyproject.toml#L3)). The vendored copy does not satisfy the declared build requirement.

Result, verified in the worktree:

```
lt.__version__ = '0+untagged.431.g121359d'      # runtime
pip metadata    = 0.2.3                          # build
```

Does versioneer break in a worktree? **No** — it resolved correctly through the `.git` gitdir pointer.
It does *not* break in a fresh clone either, but the repo has **no tags**, so every install reports
`0+untagged.N.g<sha>`.

Two secondary consequences:

* `.gitattributes` is listed in [`.gitignore:3`](../.gitignore#L3), so versioneer's
  `export-subst` keyword substitution can never be committed. A GitHub tarball download yields
  `0+unknown`.
* [`docs/conf.py:60`](../docs/conf.py#L60) sets the Sphinx version from `lettuce.__version__`, so the
  docs would be labelled `0+untagged...`.

**Why it hurts.** An agent trying to be helpful may "fix the versioning" and lose a lot of time on
something orthogonal to the task. A `0.2.3` vs `0+untagged` mismatch also reads as a broken install.

**Mitigation:** *Fix before publishing* (delete `versioneer.py` and the `[tool.versioneer]` block, or
wire it up properly). Low risk, high clarity gain.

### 1.4 `readme = "README.md"` points at a nonexistent file — *verified*

[`pyproject.toml:9`](../pyproject.toml#L9) — only `README.rst` exists. setuptools tolerates this
silently (metadata builds fine, `Description-Content-Type: text/markdown` with an empty body), so it
is cosmetic, but it is one more signal that packaging metadata is not trustworthy.

**Mitigation:** Fix before publishing (one-word change).

### 1.5 No hard CUDA dependency in code, but plenty of hard CUDA assumptions — *verified*

The package itself imports fine on CPU (`import lettuce` in 2.3 s, no GPU). The CUDA coupling is in
defaults and examples, not in imports. See §2.2 and §5.3.

### 1.6 Stale build artifacts

The checked-out worktree is clean (`git status` shows only the untracked `autoresearch/`). `.gitignore`
covers `build/`, `dist/`, `.eggs/`, `*.egg-info/`, `__pycache__/` correctly. Installing produces
`lettuce.egg-info/` in-tree, which is gitignored and harmless.

One machine-specific hazard worth knowing (not a repo problem, but it will bite anyone reusing the
maintainer's conda env): `/home/pspelt3s/miniconda3/envs/lettuce` (Python 3.11.3, torch 2.0.1, no CUDA)
has **three stale `.egg` installs of lettuce plus an editable install pointing at the main checkout**
(`/home/pspelt3s/lettuce`), which would shadow the worktree. It also needs
`MKL_THREADING_LAYER=GNU` on every invocation, otherwise it dies with
`symbol lookup error: .../libmkl_intel_thread.so.2: undefined symbol: __kmpc_global_thread_num`.
With that prefix it does run the suite (independently measured: *400 passed, 884 skipped in 68.16 s*
— matching the clean `uv` environment), but an agent must build a fresh environment rather than reuse
this one.

**Mitigation:** State in the task prompt that a fresh venv is required.

---

## 2. Test-suite friction

### 2.1 Good news: the suite is green and fast — *verified*

```
400 passed, 884 skipped, 26 warnings in 63.10s
```

**There are no pre-existing failures on master (CPU).** This is the single healthiest thing about the
repo as a benchmark substrate. Collection takes 1 s; the whole suite takes ~65 s wall (7m39s CPU —
it parallelises internally via torch). The slowest tests are
`tests/util/test_torch_gradient.py` (~8 s each, 3 of them).

The CLI integration checks that CI runs also pass and give a *real* validation signal:

```
$ lettuce --no-cuda convergence --use-no-cuda_native
 resolution   error (u)   order (u)   error (p)   order (p)
         16    1.91e-02        0.00    2.27e-02        0.00
        ...
        256    5.86e-05        2.07    1.13e-03        1.02
```

Second-order convergence in velocity — a genuinely useful acceptance criterion for a new BC task.

### 2.2 69% of tests skip without a GPU — *verified*

| Directory | passed | skipped |
|---|---:|---:|
| `tests/boundary` | 85 | 281 |
| `tests/collision` | 138 | 358 |
| `tests/flow` | 49 | 130 |
| `tests/moments` | 40 | 28 |
| **`tests/native`** | **0** | **6** |
| `tests/reporter` | 34 | 41 |
| `tests/stencil` | 20 | 0 |
| `tests/unit` | 3 | 0 |
| `tests/util` | 8 | 0 |

`tests/native/__init__.py:4` skips the whole package: *"CUDA is not available on this machine, so
cuda_native will not be tested."*

**Why it hurts.** The agent sees a mostly-`s` progress bar, which reads as "the suite is broken".
More importantly, if the task asks for a native implementation, **there is no way to verify it** on a
CPU-only runner. The agent will write CUDA C++ it cannot compile or test.

**Mitigation:** *State explicitly in the task prompt* — "the grader runs on CPU; do not implement
`cuda_native` codegen, stub `native_available()` to `False`". See §3.4.

### 2.3 `conftest.py` is comprehensible but fragile

[`tests/conftest.py`](../tests/conftest.py) is readable: `fix_stencil`, `fix_dtype`, `fix_device`,
`fix_configuration` (the device × dtype × native cross-product, with CPU+native excluded), plus two
reusable flows `TestFlow` and `DummyFlow`. An agent can copy a test in ~2 minutes. That is a plus.

Three defects:

* **`fix_native` is broken and dead.** [`conftest.py:175`](../tests/conftest.py#L175) does
  `request.param[0]` where `request.param` is a plain `bool` → `TypeError: 'bool' object is not
  subscriptable`. Verified unused: `grep -rn fix_native tests/` returns only the definition. If an
  agent picks this fixture (its name is the most obvious one), it gets a baffling error.
* **It depends on star-import leakage.** `conftest.py:6` is `from lettuce import *`; it then uses the
  bare names `lettuce` ([line 112](../tests/conftest.py#L112)), `Union`, `List`, `Optional`,
  `np`, `torch`. None of these are exported deliberately — `lettuce/__init__.py` has no `__all__`, so
  `import lettuce._version` accidentally binds the name `lettuce` into the package namespace and the
  submodules' `typing` imports leak through. Adding an `__all__` anywhere upstream would break the
  entire test suite in a way that looks unrelated to the change.
* **No pytest configuration exists** — no `pytest.ini`, `setup.cfg`, `tox.ini`, or `[tool.pytest]`
  block. No `testpaths`, no markers, no `--timeout`. `pytest tests` must be run from the repo root
  (`from tests.conftest import *` requires it on `sys.path`).

### 2.4 A test that can never exercise its own branch — *verified by inspection*

[`tests/boundary/test_bc_masks.py:11`](../tests/boundary/test_bc_masks.py#L11):

```python
all_native_boundaries_in_Obstacle = sum([
    _.native_available() for _ in flow.boundaries]) == flow.boundaries
```

This compares an `int` to a `list`, so it is **always `False`**, so the native branch of the test is
always skipped when `native=True`. This is the one test that checks mask plumbing end-to-end, and it
only ever asserts `.any()` — it would pass even with the §3.1 bug.

The same copy-pasted expression appears in **three** files:
[`test_bc_masks.py:10`](../tests/boundary/test_bc_masks.py#L10),
[`test_equilibrium_pressure_outlet.py:43`](../tests/boundary/test_equilibrium_pressure_outlet.py#L43),
[`test_obstacle.py:24`](../tests/flow/test_obstacle.py#L24).

### 2.4b The green suite is partly green because tests are switched off — *verified*

`pytest.skip` appears 40 times across `tests/`. Most are legitimate (CUDA, stencil dimensionality),
but several disable tests whose authors recorded that the code is *wrong*, not merely untestable:

| Test | Line | Skip/tolerance | What it means |
|---|---|---|---|
| `tests/boundary/test_equilibrium_pressure_outlet.py` | [:5](../tests/boundary/test_equilibrium_pressure_outlet.py#L5) | unconditional `pytest.skip("TODO rtol is too big or simulation time too short!")` as the **first statement of the test body** | the only end-to-end test of `EquilibriumOutletP` is fully disabled |
| `tests/boundary/test_equilibrium_bc_outlet_p.py` | [:31](../tests/boundary/test_equilibrium_bc_outlet_p.py#L31) | `pytest.approx(..., rel=1e-2)  # TODO rel is too big!` | 1% tolerance on what should be an exact equilibrium identity |
| `tests/native/test_native_no_streaming_mask.py` | [:8](../tests/native/test_native_no_streaming_mask.py#L8) | skipped without CUDA; docstring is literally `"""test if """` | the one test *named after* the §3.1 mask never runs on CPU |
| `tests/collision/test_collision_fixpoint_2x_MRT.py` | [:9](../tests/collision/test_collision_fixpoint_2x_MRT.py#L9) | unconditional skip, `"Transform() currently still uses Lattice methods."` | dangling v0.1 API, cf. §3.5 |
| `tests/flow/test_pressure_poisson.py` | [:8](../tests/flow/test_pressure_poisson.py#L8) | `"D3Q27 pressure_poisson does not work."` | matches the `TODO(@MCBs)` at [`_flow.py:309`](../lettuce/_flow.py#L309), §4.4 |
| `tests/reporter/test_energy_spectrum.py` | [:9](../tests/reporter/test_energy_spectrum.py#L9) | `"CouetteFlow2D has nan energy spectrum"` | a shipped flow produces NaN |

**Why it hurts an agent.** "400 passed, 0 failed" reads as a trustworthy baseline. It is not: the
boundary subsystem the agent is about to extend has its strictest tests disabled or loosened to 1%.
An agent that calibrates its own test's tolerance against `test_equilibrium_bc_outlet_p.py` will write
a test that cannot detect a wrong boundary.

**Mitigation:** *State explicitly in the task prompt* — give the agent the acceptance criterion and
tolerance rather than letting it infer one from the existing tests.

### 2.5 Existing boundary tests are a misleading template

[`tests/boundary/test_bounceback_bc.py`](../tests/boundary/test_bounceback_bc.py) calls
`bounce_back(flow)` **directly**, never through `Simulation`. So it tests the kernel and never the
mask plumbing — which is exactly where the critical bug lives (§3.1). Worse,
`test_bounce_back_boundary_not_applied_if_mask_empty` (line 22) asserts `flow.f` is unchanged, but
`BounceBackBoundary.__call__` never writes `flow.f` at all, so the assertion is vacuous — it passes
for any non-mutating implementation regardless of the mask.

An agent copying this template will write a test that cannot detect its own integration bugs.

**Mitigation:** *State explicitly in the task prompt* — require at least one test that drives
`Simulation` for N steps and checks a physical invariant.

---

## 3. API discoverability: learning the `Boundary` interface

### 3.1 CRITICAL — `no_streaming_mask` is initialised wrong — *verified*

[`_simulation.py:106`](../lettuce/_simulation.py#L106):

```python
self.no_collision_mask = self.context.full_tensor(
    flow.resolution, self.collision_index, dtype=torch.uint8)
self.no_streaming_mask = self.context.full_tensor(
    [flow.stencil.q, *flow.resolution], self.collision_index, dtype=torch.uint8)
```

Filling `no_collision_mask` with `collision_index` is correct and deliberate — it is an *int* field of
boundary indices, documented at [`_simulation.py:83`](../lettuce/_simulation.py#L83).

Filling **`no_streaming_mask`** with `collision_index` is not. It is a *boolean* field
([`_simulation.py:93`](../lettuce/_simulation.py#L93): *"a boolean mask that will mark all nodes where
populations will not be altered during the streaming-substep"*), it is combined with `|=`
([lines 129](../lettuce/_simulation.py#L129) and [146](../lettuce/_simulation.py#L146)), and it is
consumed as `torch.eq(self.no_streaming_mask[i], 1)`
([`_simulation.py:254`](../lettuce/_simulation.py#L254)).

Measured behaviour across configurations:

| `len(pre)` | `len(post)` | `collision_index` | mask values | streaming blocked everywhere? | BC's no-streaming request honoured? |
|---:|---:|---:|---|---|---|
| 0 | 1 | 0 | `[0, 1]` | no | **yes** ✅ |
| 0 | 2 | 0 | `[0, 1]` | no | **yes** ✅ |
| 1 | 0 | 1 | `[1]` | **YES — whole domain frozen** ❌ | no |
| 1 | 1 | 1 | `[1]` | **YES — whole domain frozen** ❌ | no |
| 2 | 0 | 2 | `[2, 3]` | no | **no — silently ignored** ❌ |
| 2 | 1 | 2 | `[2, 3]` | no | **no — silently ignored** ❌ |
| 3 | 0 | 3 | `[3]` | no | **no — silently ignored** ❌ |

Two distinct failures:

* `collision_index == 1` (exactly one pre-boundary): the mask is `1` everywhere, so `_stream()` keeps
  the old populations everywhere. **The simulation collides but never streams.** No error, no warning.
* `collision_index >= 2`: `2 | True == 3`, and `eq(3, 1)` is `False`, so a boundary's genuine
  no-streaming request is **discarded**. Again silent.

The bug is completely latent today because **every shipped flow uses only `post_boundaries`** —
`grep` over `lettuce/ext/_flows/` finds nine `post_boundaries` and zero `pre_boundaries`. No test
touches the `pre_boundaries` path.

**Why it hurts an agent.** "pre_boundaries" is the natural-sounding place for an inlet/wall BC applied
before collision. An agent that puts its new BC there, correctly implements `make_no_streaming_mask`,
and validates with a plausible test will see *plausible-looking wrong numbers* — the classic worst
case for a benchmark, because it scores the model on a bug it cannot reasonably find.

**Mitigation: fix before publishing.** The fill value for `no_streaming_mask` should be `0`
(or the tensor should be `dtype=torch.bool`). If the maintainers prefer not to touch it, the task
prompt **must** say "register your boundary in `post_boundaries`" and the grader must not depend on
`no_streaming_mask`.

### 3.2 The `Boundary` contract is undocumented and inconsistent

[`_flow.py:31–52`](../lettuce/_flow.py#L31):

```python
class Boundary(ABC):
    @abstractmethod
    def __call__(self, flow: 'Flow'): ...
    @abstractmethod
    def make_no_collision_mask(self, shape, context) -> Optional[torch.Tensor]: ...
    @abstractmethod
    def make_no_streaming_mask(self, shape, context) -> Optional[torch.Tensor]: ...
    @abstractmethod
    def native_available(self) -> bool: ...
    @abstractmethod
    def native_generator(self, index: int) -> 'NativeBoundary': ...
```

**Five abstract methods, zero docstrings, zero class docstring.** Everything below is a silent
contract an agent must reverse-engineer from `_simulation.py` and four example implementations that
do not agree with each other:

1. **`__call__` must return a full-domain `f` of shape `[q, *resolution]`, not just boundary nodes.**
   `Simulation._collide` evaluates `boundary(self.flow)` over the *whole* domain and then selects with
   `torch.where` ([`_simulation.py:291`](../lettuce/_simulation.py#L291)). So the kernel must be
   numerically safe at every node, including deep in the fluid. Nothing says this.
2. **`make_no_collision_mask(shape, ...)` takes the spatial shape** `[*resolution]`
   ([`_simulation.py:117`](../lettuce/_simulation.py#L117): `self.flow.f.shape[1:]`), while
   **`make_no_streaming_mask(shape, ...)` takes the full shape** `[q, *resolution]`
   ([line 126](../lettuce/_simulation.py#L126): `self.flow.f.shape`). Same parameter name `shape`, two
   different meanings, no docstring distinguishing them.
3. **Returning `None` means "no mask"** — legal, used by `BounceBackBoundary`. But the shipped code
   expresses it three ways: `return None`
   ([bounce_back_boundary.py:22](../lettuce/ext/_boundary/bounce_back_boundary.py#L22)), a bare `pass`
   ([equilibrium_boundary_pu.py:92](../lettuce/ext/_boundary/equilibrium_boundary_pu.py#L92)), and a bare
   `pass` with no return annotation
   ([partially_saturated_boundary.py:45](../lettuce/ext/_boundary/partially_saturated_boundary.py#L45)).
4. **Masks must be `torch.bool` on the right device.** A `uint8` mask crashes — see §4.1.
5. **Ordering matters and is unstated.** Boundaries run in list order, before and after collision, and
   their index in `no_collision_mask` depends on their position. The only explanation is a block
   comment at [`_simulation.py:277`](../lettuce/_simulation.py#L277).

**Mitigation:** *Fix before publishing.* Five docstrings. Without them, the exercise measures
archaeology, not engineering.

### 3.3 Pure vs. in-place: the shipped boundaries disagree — *by inspection*

| Boundary | Returns | Mutates `flow.f`? |
|---|---|---|
| [`BounceBackBoundary`](../lettuce/ext/_boundary/bounce_back_boundary.py#L17) | new tensor `flow.f[opposite]` | no |
| [`PartiallySaturatedBC`](../lettuce/ext/_boundary/partially_saturated_boundary.py#L22) | new tensor via `torch.where` (masks *internally*, then `Simulation` masks *again*) | no |
| [`AntiBounceBackOutlet`](../lettuce/ext/_boundary/anti_bounce_back_outlet.py#L82) | `flow.f` itself | **yes, in place** |
| [`EquilibriumOutletP`](../lettuce/ext/_boundary/equilibrium_outlet_p.py#L70) | `einsum` copy, but `feq = flow.f` then `feq[here] = ...` | **yes, in place** |

For the two in-place variants the write to `flow.f` has *already happened* by the time
`torch.where(cond, boundary(flow), flow.f, out=flow.f)` runs, so their `make_no_collision_mask` is
decorative for the write itself. An agent that copies `AntiBounceBackOutlet` inherits in-place
semantics; one that copies `BounceBackBoundary` inherits functional semantics. Both "look right".
There is no statement anywhere about which is correct.

Note also the aliasing in [`_simulation.py:291–304`](../lettuce/_simulation.py#L291):
`torch.where(..., self.flow.f, out=self.flow.f)` uses `flow.f` as both an input and the output buffer.

**Mitigation:** *Fix before publishing* (pick one convention and document it), or
*state explicitly in the task prompt*.

### 3.4 `native_available` / `native_generator`: opting out is cheap, but only on CPU — *verified*

Good news: the ABC forces the agent to write both methods, and the failure is loud and self-explaining:

```
TypeError: Can't instantiate abstract class PartialBC without an implementation
for abstract methods 'native_available', 'native_generator'
```

Opting out is a four-line stub (`return False` / `return None`), exactly as
[`AntiBounceBackOutlet`](../lettuce/ext/_boundary/anti_bounce_back_outlet.py#L105),
`EquilibriumOutletP` and `PartiallySaturatedBC` all do. **So the agent is *not* forced to write CUDA
codegen.**

Bad news: the opt-out is only safe while `context.use_native` is `False`.
[`_simulation.py:200–205`](../lettuce/_simulation.py#L200) calls `boundary.native_generator(i)`
**unconditionally** when `use_native` is set — the `native_available()` check three lines earlier only
*prints* a message ([line 186](../lettuce/_simulation.py#L186)) and does not stop anything. The `None`
then flows into `Generator` → `DefaultCodeGeneration.generate` →
[`_default_code_gen.py:292`](../lettuce/cuda_native/_default_code_gen.py#L292)
`transformer.generate(self)` → `AttributeError: 'NoneType' object has no attribute 'generate'`.

And [`Context.__init__`](../lettuce/_context.py#L17) defaults `use_native=True` whenever CUDA is
available. So on any GPU machine, `lt.Context()` + `lt.Obstacle` (which includes the non-native
`AntiBounceBackOutlet`) is broken out of the box. The agent cannot observe this on a CPU runner.

**Mitigation:** *State explicitly in the task prompt* — "the grader uses `use_native=False`;
`native_available() -> False` and `native_generator() -> None` are acceptable". Separately, the
maintainers should make `native_generator` conditional on `native_available()`.

### 3.5 `lettuce/base.py` is dead code from the v0.1 API — *verified*

[`base.py`](../lettuce/base.py) defines `LatticeBase` with a `lattice` attribute, a `native_available()`
and a `create_native()` — and *has the best docstrings in the repository* for exactly the concepts the
agent needs. It is **never imported anywhere** (`grep -rn "LatticeBase\|from .base"` matches only
`base.py` itself), it is not in `lettuce/__init__.py`, and its API (`create_native`, a `lattice`
constructor arg) does not match the live API (`native_generator`, no `lattice`).

**Why it hurts.** A file literally named `base.py` with the repo's clearest docs on `native_available`
is a near-perfect decoy. An agent grepping for the extension interface will find it first and follow
the wrong contract.

**Mitigation:** *Fix before publishing* — delete it, or move its prose into the live `Boundary` ABC.

### 3.6 Boundary objects are constructed twice and `Simulation.transformer` is a dead duplicate — *verified*

`pre_boundaries` / `post_boundaries` are **properties**. `Simulation.__init__` reads them at
[line 70](../lettuce/_simulation.py#L70) (to build `self.transformer`) and again at
[lines 72–73](../lettuce/_simulation.py#L72). Since e.g.
[`Obstacle.post_boundaries`](../lettuce/ext/_flows/obstacle.py#L108) constructs fresh objects on every
access, the result is:

```
boundary __init__ calls during Simulation.__init__: 2
sim.transformer[-1] is sim.post_boundaries[0]: False
```

`self.transformer` is then **never read again** — `grep -n "self.transformer" lettuce/_simulation.py`
returns only the assignment. So every boundary is constructed once for nothing.

**Why it hurts.** A stateful boundary (force accumulator, time-dependent inlet, cached geometry) is a
completely reasonable thing to write, and it will silently have a doppelgänger. Worse, any external
reference the agent's *test* holds is not the object the simulation uses. Construction cost is also
doubled, which matters for boundaries that precompute neighbour lists.

**Mitigation:** *Fix before publishing* (cache the property reads) — it is a two-line change and
removes a whole class of confusing test failures.

---

## 4. Traps and footguns

### 4.1 `Context.convert_to_tensor` silently turns `bool` into `uint8` — *verified*

[`_context.py:99`](../lettuce/_context.py#L99):

```python
bool_types = [bool, torch.bool, torch.uint8, np.uint8]
if hasattr(array, 'dtype') and array.dtype in bool_types:
    new_dtype = torch.uint8
```

Verified:

```python
ctx.convert_to_tensor(np.zeros((8,8), dtype=bool)).dtype   # torch.uint8
ctx.convert_to_tensor(torch.zeros(8,8, dtype=torch.bool)).dtype  # torch.uint8
```

and the downstream consequence in `Simulation.__init__`
([`_simulation.py:120`](../lettuce/_simulation.py#L120), `self.no_collision_mask[ncm] = i`):

```
RuntimeError: masked_fill_ only supports boolean masks, but got mask with dtype unsigned char
```

The working idiom is `context.zero_tensor(shape, dtype=bool)` — which is what every shipped boundary
uses, but it is only discoverable by reading them. `convert_to_tensor` is the more obvious-sounding
API and it is the wrong one for masks.

**Why it hurts.** It is at least *loud*. But the error surfaces in `Simulation.__init__`, far from the
agent's mask-construction code, and the message ("masked_fill_") does not obviously point at
`convert_to_tensor`.

**Mitigation:** *State explicitly in the task prompt*, or add a `dtype=torch.bool` shortcut.

### 4.2 `Context(use_native=False)` silently picks `cuda:0` — *verified, critical*

[`_context.py:22`](../lettuce/_context.py#L22):

```python
elif device is None:
    assert not use_native or torch.cuda.is_available(), ...
    device = torch.device('cuda:0')     # unconditional
```

Measured on a CPU-only machine:

| Call | resulting device | works? |
|---|---|---|
| `Context()` | `cpu` | ✅ |
| `Context(use_native=False)` | **`cuda:0`** | ❌ `AssertionError: Torch not compiled with CUDA enabled` |
| `Context(dtype=torch.float64, use_native=False)` | **`cuda:0`** | ❌ same |
| `Context(device='cpu')` | `cpu` | ✅ |
| `Context(device='cpu', use_native=False)` | `cpu` | ✅ |

The failure does not happen in the constructor — it happens later, on the first tensor conversion
inside [`TorchStencil.__init__`](../lettuce/_stencil.py#L38), with a torch-internal traceback that
never mentions `Context`.

This is not hypothetical: it is why four of the repo's own examples fail (§5.3), e.g.
[`examples/simple_flows/LambOseenVortex.py:11`](../examples/simple_flows/LambOseenVortex.py#L11)
which reads `lt.Context(dtype=torch.float64, use_native=False)`.

Related dead code in the same file: [`_context.py:13`](../lettuce/_context.py#L13) is
`if device is str:` (identity comparison against the `str` *type*) — always `False`; it should be
`isinstance(device, str)`. And [`line 56`](../lettuce/_context.py#L56) formats an assertion message
with `dtype.__name__`, which `torch.dtype` does not have — the error handler would itself raise.

**Mitigation:** *Fix before publishing.* `device = torch.device('cuda:0' if torch.cuda.is_available()
else 'cpu')` is the obvious intent.

### 4.3 `torch_stencil.opposite` is a float tensor — *verified*

[`_stencil.py:40`](../lettuce/_stencil.py#L40) runs `opposite` through `convert_to_tensor`, which
applies `self.dtype` (a float) because an int list is not in `bool_types`:

```python
flow.stencil.opposite        # [0, 3, 4, 1, 2, 7, 8, 5, 6]   (list[int])  -> f[...] works
flow.torch_stencil.opposite  # tensor([0., 3., ...], float64)             -> IndexError
# IndexError: tensors used as indices must be long, int, byte or bool tensors
```

`BounceBackBoundary` uses `flow.stencil.opposite`;
[`AntiBounceBackOutlet`](../lettuce/ext/_boundary/anti_bounce_back_outlet.py#L82) uses
`context.convert_to_ndarray(flow.torch_stencil.opposite)`. Two shipped boundaries, two different
idioms for the same thing, no comment explaining why.

Also in the same file, [`_stencil.py:38` and `:41`](../lettuce/_stencil.py#L38) are a verbatim
duplicate (`self.e = context.convert_to_tensor(stencil.e)` twice).

And `Stencil` ([`_stencil.py:14`](../lettuce/_stencil.py#L14)) is declared `ABC` but has **no abstract
methods**, so `lt.Stencil()` instantiates happily and fails much later with `AttributeError: 'Stencil'
object has no attribute 'e'`.

**Mitigation:** Leave as-is (the error is loud), but name it in the task prompt.

### 4.4 `initialize_pressure=True` is a no-op — *by inspection*

[`_flow.py:134–141`](../lettuce/_flow.py#L134):

```python
if self.initialize_pressure:
    initial_rho = pressure_poisson(...)
    self.f = self.equilibrium(self, rho=initial_rho, u=initial_u)
self.f = self.equilibrium(self, rho=initial_rho, u=initial_u)   # <- not in an else
```

The second assignment is unconditional. It happens to use the *updated* `initial_rho`, so the result
is the same — meaning line 140 is pure dead code that computes a full equilibrium and throws it away.
Reading it, an agent cannot tell whether this is a bug or intentional; either way it doubles the cost
of pressure-initialised flows. `pressure_poisson` itself carries
[`# TODO(@MCBs): still not working in 3D`](../lettuce/_flow.py#L309) and hard-codes `dim=2`.

### 4.5 Units: `UnitConversion` defaults are a silent-wrong-physics trap

[`_unit.py:21`](../lettuce/_unit.py#L21): `characteristic_length_lu=1` by default. Since
`viscosity_lu = characteristic_length_lu * characteristic_velocity_lu / Re`
([`_unit.py:49`](../lettuce/_unit.py#L49)), forgetting to pass the resolution gives a viscosity too
small by a factor of N and a `relaxation_parameter_lu` near `0.5` — an unstable but *not obviously
wrong* simulation. `conftest.DummyFlow.make_units`
([`conftest.py:253`](../tests/conftest.py#L253)) does exactly this (deliberately, but it is the
copy-paste template an agent will find).

Of the 24 conversion methods in `_unit.py`, **only 4 have docstrings** (the four energy ones).
Nothing states the pu↔lu direction convention beyond the method names, and the naming is
asymmetric: `convert_velocity_to_lu` / `convert_velocity_to_pu` but
`convert_density_lu_to_pressure_pu` / `convert_pressure_pu_to_density_lu` — four different naming
patterns in one class.

**Mitigation:** Leave as-is — this is legitimate domain difficulty and a fair part of the challenge.
But the task prompt should say which quantities the grader checks and in which unit system.

### 4.6 Axis ordering is consistent (a genuine positive)

Every `grid` property uses `torch.meshgrid(*xyz, indexing='ij')` (9 occurrences, no exceptions), and
`f` is uniformly `[q, x, y, (z)]`. No transposition traps found. This is undocumented but at least
self-consistent.

### 4.7 Dead code, duplication and misleading names — inventory

| Item | Location | Note |
|---|---|---|
| `LatticeBase` — entire module unused | [`base.py`](../lettuce/base.py) | v0.1 API decoy, see §3.5 |
| `Simulation.transformer` built, never read | [`_simulation.py:70`](../lettuce/_simulation.py#L70) | costs a duplicate construction of every boundary |
| `Flow.velocity` is a `@property` with a `f` parameter that can never be passed | [`_flow.py:196`](../lettuce/_flow.py#L196) | also duplicates `Flow.u` |
| `Flow.einsum` deprecated wrapper | [`_flow.py:236`](../lettuce/_flow.py#L236) | still exercised by `tests/flow/test_einsum.py` |
| `EquilibriumOutletP.__init__` re-derives everything `super().__init__` just computed | [`equilibrium_outlet_p.py:22–61`](../lettuce/ext/_boundary/equilibrium_outlet_p.py#L22) | ~40 duplicated lines |
| `EquilibriumOutletP(AntiBounceBackOutlet)` | [`equilibrium_outlet_p.py:12`](../lettuce/ext/_boundary/equilibrium_outlet_p.py#L12) | a pressure outlet inheriting from an anti-bounce-back outlet — misleading hierarchy |
| `direction: [List[int]]` — a *list containing a type* | [`anti_bounce_back_outlet.py:22`](../lettuce/ext/_boundary/anti_bounce_back_outlet.py#L22) | nonsense annotation |
| Commented-out collision call + *"not 100% sure about this"* | [`anti_bounce_back_outlet.py:72`](../lettuce/ext/_boundary/anti_bounce_back_outlet.py#L72) | the constructor still takes and stores an unused `collision` |
| Commented-out `EquilibriumOutletP` alternative in `Obstacle.post_boundaries` | [`obstacle.py:119`](../lettuce/ext/_flows/obstacle.py#L119) | invites the agent to "fix" it |
| Commented-out `make_no_collision_mask` in a test helper | [`test_equilibrium_bc_pu.py:26`](../tests/boundary/test_equilibrium_bc_pu.py#L26) | |
| Commented-out `__all__` entries | [`util/__init__.py:23`](../lettuce/util/__init__.py#L23) | `moments` excluded "due to cyclic import issue" |
| Commented-out `collision`/`boundaries` properties | [`_default_code_gen.py:51`](../lettuce/cuda_native/_default_code_gen.py#L51) | |
| `pre_boundaries` / `post_boundaries` annotations commented out | [`_flow.py:71`](../lettuce/_flow.py#L71), [`_simulation.py:46`](../lettuce/_simulation.py#L46) | |
| `@abstractmethod` commented out on all three boundary properties | [`_flow.py:98`](../lettuce/_flow.py#L98) | so a flow can silently define none |
| `ExperimentalWarning: "I am not 100% sure if this equilibrium is correct."` | [`util/moments.py:189`](../lettuce/util/moments.py#L189), `:246` | fires during the test run |
| `"KBCCollision2D is is deprecated"` (typo, and the 3D class emits the 2D message) | [`kbc_collision.py:171`](../lettuce/ext/_collision/kbc_collision.py#L171), `:178` | |
| 8 `TODO`/`FIXME` markers | see Appendix A | |
| `Obstacle(ctx, 16, ...)` fails despite `Union[int, List[int]]` | [`obstacle.py:59`](../lettuce/ext/_flows/obstacle.py#L59) (`resolution[0]`) | `TypeError: 'int' object is not subscriptable` — *verified*; `make_resolution` (line 79) handles the int case that `__init__` already crashed on |

---

## 5. Documentation state

### 5.1 Docstring coverage — *measured by AST walk over `lettuce/`*

| | with docstring | total | missing |
|---|---:|---:|---:|
| Modules | 14 | 79 | **65 (82%)** |
| Classes | 31 | 92 | **61 (66%)** |
| Public functions/methods | 65 | 346 | **281 (81%)** |

Every class the agent must interact with is undocumented: `Boundary`, `Simulation`, `Context`,
`Stencil`, `TorchStencil`, `Collision`, `Equilibrium`, `BounceBackBoundary`, `BGKCollision`,
`QuadraticEquilibrium`, `NativeBoundary`, `NativeTransformer`. The only well-documented class in the
extension path is the dead `LatticeBase` (§3.5).

### 5.2 `docs/` is structurally broken — *verified*

* **15 of 17 `automodule` targets in [`docs/modules.rst`](../docs/modules.rst) do not exist**:
  `lettuce.simulation`, `lettuce.lattices`, `lettuce.stencils`, `lettuce.streaming`,
  `lettuce.collision`, `lettuce.observables`, `lettuce.reporters`, `lettuce.force`,
  `lettuce.equilibrium`, `lettuce.boundary`, `lettuce.flows.{couette,poiseuille,taylorgreen,
  decayingturbulence,obstacle}`. The real modules are `lettuce._simulation`, `lettuce.ext._boundary`,
  `lettuce.ext._flows.*`. The entire published API reference is empty.
* [`docs/contributing.rst`](../docs/contributing.rst) does `.. include:: ../CONTRIBUTING.rst` — **that
  file does not exist**. Same for [`docs/history.rst`](../docs/history.rst) → `../HISTORY.rst`. There
  is therefore **no contributing guide at all**.
* [`docs/installation.rst:29`](../docs/installation.rst#L29) recommends `python setup.py install` —
  there is no `setup.py`, and setuptools would reject it anyway.
* [`docs/README.md`](../docs/README.md) tells you to build docs with `conda install sphinxcontrib-napoleon`
  (a package renamed years ago); `.readthedocs.yaml` instead installs from `requirements.txt`, which
  contains no Sphinx at all — so the RTD build cannot work either.

### 5.3 `README.rst` is partly inaccurate

* Install commands: see §1.1. The badge points at `lettucecfd.readthedocs.io` while the
  "Documentation" link ([README.rst:33](../README.rst#L33)) points at `lettuceboltzmann.readthedocs.io`.
* [README.rst:44](../README.rst#L44) sends newcomers to `examples/` — **5 of the 7 runnable `.py`
  examples fail or hang on CPU** (verified, 180 s timeout):

| Example | Result on CPU |
|---|---|
| `examples/00_simplest_TGV.py` | ✅ runs (0.63 MLUPS) |
| `examples/01b_first_example_obstacle.py` | ❌ `AssertionError: cuda device explicitly requested but cuda is not available!` — hard-codes `torch.device("cuda:0")` at [line 13](../examples/01b_first_example_obstacle.py#L13) |
| `examples/02_converging_obstacle_flow.py` | ⏱ timeout >180 s |
| `examples/03_outputs_TGV.py` | ⏱ timeout >180 s (3-D D3Q27, resolution 80) |
| `examples/simple_flows/LambOseenVortex.py` | ❌ `Torch not compiled with CUDA enabled` — hits the §4.2 `Context` bug |
| `examples/development/example_progress_reporter_based_on_simplest_TGV.py` | ❌ same |
| `examples/advanced_flows/PartiallySaturatedObstacle.py` | ❌ same |
| `examples/advanced_flows/FailingTGVandObstacle.py` | ❌ same |

Several also carry a copy-pasted `# for running on cpu: device='cpu'` comment attached to the wrong
argument (e.g. [`LambOseenVortex.py:13`](../examples/simple_flows/LambOseenVortex.py#L13)), which is
actively wrong advice given §4.2.

### 5.4 `native_cuda_synopsis.md` names classes that no longer exist

[`native_cuda_synopsis.md`](../native_cuda_synopsis.md) is a v0.1→v0.2 migration note. It is the only
root-level design document, so an agent will read it, and it references `lt.Lattice`, `lt.Streaming`,
`lt.StandardStreaming` (all removed) and `lt.UnitConverter` (the real class is `UnitConversion`). Its
one genuinely useful line — *"`no_collision_mask` now includes the boundary's identifier (important
for custom flows)"* — is the closest thing to documentation of the mask contract, and it is buried in
a bullet list of deprecations.

### 5.5 There is no `CLAUDE.md`, `AGENTS.md`, or any agent instructions

None found. Nothing tells an agent how to build, test, or where extension points live.

**Mitigation:** *Fix before publishing* — a 30-line `CLAUDE.md` with the working install command,
`pytest tests`, and a pointer to `lettuce/ext/_boundary/` would remove most §1 friction without
making the actual task easier. (Alternatively put it in the task prompt, which keeps the repo pristine.)

---

## 6. Repo hygiene affecting agents

* **`.gitignore` ignores `Makefile` and `.gitattributes`** ([`.gitignore:1–3`](../.gitignore#L1)).
  Ignoring `Makefile` means any build shortcut is invisible to the agent (none exists today, so this
  is latent). Ignoring `.gitattributes` means versioneer's `export-subst` line can never be committed
  (§1.3). Both look like accidents from a cookiecutter template.
* **`.source/`** contains only `img/logo_lettuce_typo.png` (130 KB), referenced by the README badge.
  Harmless, but the leading-dot name makes it invisible to `ls` and to many file-listing tools — an
  agent may conclude the logo link is broken.
* **No `build/`, `dist/`, `.eggs/` in the repo** — clean. `git status` is clean apart from the
  untracked `autoresearch/`.
* **Repo is small**: 259 tracked files, 5.0 MB. Largest files are notebooks
  (`00_run_parametrized_project.ipynb` 1.6 MB, `DecayingTurbulence.ipynb` 1.0 MB) and `uv.lock`
  (284 KB). Notebooks with large embedded outputs are token-expensive if an agent reads them, but
  nothing is pathological.
* **`requirements.txt` has drifted from `uv.lock`** — `requirements.txt` says `filelock==3.20.3` and
  `fonttools==4.61.0` (from the recent Dependabot commits), while `uv.lock` still pins `3.18.0` and
  `4.59.0`. Two dependency sources of truth that disagree; `requirements.txt` is used only by
  `.readthedocs.yaml`. It also pins the full `nvidia-*` CUDA stack, which is wrong for an RTD build.
* **The `examples/advanced_projects/` parallel framework** — see §6.1.

### 6.1 The competing boundary framework in `examples/` — *verified*

`examples/advanced_projects/efficient_bounce_back_obstacle/` (added in the most recent commit,
`121359d` "270 efficient hwbb and ibb") contains a full second implementation of boundary machinery:

* `boundary/fullway_bounce_back_boundary.py`, `halfway_bounce_back_boundary.py`,
  `linear_interpolated_bounce_back_boundary.py`, `solid_boundary_data.py`
* `simulation/ebb_simulation.py` — an `EbbSimulation(Simulation)` subclass introducing a **third**
  boundary category, `post_streaming_boundaries`, with its own mask plumbing
  ([`ebb_simulation.py:29–57`](../examples/advanced_projects/efficient_bounce_back_obstacle/simulation/ebb_simulation.py#L29))
  that **replicates the §3.1 `no_streaming_mask` bug verbatim**.
* A different constructor convention: `HalfwayBounceBackBoundary(context, flow, solid_boundary_data,
  global_solid_mask, periodicity, calc_force)` vs. the package's `BounceBackBoundary(mask)`.

**It does not import.** `boundary/__init__.py` is empty and the modules use bare top-level imports:

```
$ python -c "import examples.advanced_projects.efficient_bounce_back_obstacle.boundary"
ModuleNotFoundError: No module named 'solid_boundary_data'
```

**Why it hurts.** An agent asked for "a new boundary condition" will grep for `Boundary` and find
*eight* implementations across two incompatible frameworks, with the non-working one being the newest
and the most elaborate (it has the docstrings, the force calculation, the periodicity handling). The
`EbbSimulation` file even contains `# FOR DEBUGGING: print(...)` lines left in
([`ebb_simulation.py:48`](../examples/advanced_projects/efficient_bounce_back_obstacle/simulation/ebb_simulation.py#L48)).
Choosing the wrong framework wastes the entire run.

**Mitigation:** *Fix before publishing* (make it importable, or move it out of the repo), or fence it
off explicitly in the task prompt: "implement against `lettuce.Boundary` in `lettuce/ext/_boundary/`;
ignore `examples/advanced_projects/`."

---

## Recommendations by disposition

### Fix before publishing
1. `no_streaming_mask` fill value ([`_simulation.py:106`](../lettuce/_simulation.py#L106)) — §3.1
2. `Context` device default ([`_context.py:22`](../lettuce/_context.py#L22)) — §4.2
3. A working, documented CPU install path (README ↔ CI reconciliation) — §1.1
4. Docstrings on the five `Boundary` abstract methods ([`_flow.py:31`](../lettuce/_flow.py#L31)) — §3.2
5. Delete `lettuce/base.py` or merge its prose into `Boundary` — §3.5
6. Make `examples/advanced_projects/.../boundary/` importable, or remove it — §6.1
7. Delete or regenerate `docs/modules.rst`; add `CONTRIBUTING.rst`/`HISTORY.rst` or drop the includes — §5.2
8. Cache the `pre_boundaries`/`post_boundaries` property reads in `Simulation.__init__` — §3.6
9. `readme = "README.rst"` in `pyproject.toml`; remove or wire up versioneer — §1.3, §1.4
10. Fix the `int == list` condition in three test files and `conftest.fix_native` (`request.param[0]`) — §2.3, §2.4
11. Triage the unconditionally-skipped boundary tests and the `rel=1e-2` tolerance — §2.4b

### State explicitly in the task prompt
* Exact setup command and Python version (`uv sync --extra cpu`, Python 3.12) — §1.1, §1.2
* "Grader runs on CPU; `native_available() -> False` is acceptable; do not write CUDA codegen" — §2.2, §3.4
* "Register the boundary in `post_boundaries`" (until §3.1 is fixed)
* "Build masks with `context.zero_tensor(shape, dtype=bool)`, not `convert_to_tensor`" — §4.1
* "`__call__` returns a full-domain `f`; `Simulation` applies the mask" — §3.2
* "Implement against `lettuce.Boundary`; ignore `examples/advanced_projects/`" — §6.1
* Which physical quantity is graded and in which unit system — §4.5
* "Add a test that drives `Simulation` end-to-end, not just the boundary kernel" — §2.5
* The acceptance tolerance — do not let the agent infer one from the existing boundary tests,
  which are disabled or loosened to `rel=1e-2` with a `# TODO rel is too big!` — §2.4b

### Leave as-is — part of the challenge
* `torch_stencil.opposite` float-tensor trap (loud `IndexError`) — §4.3
* `UnitConversion` naming asymmetry and pu/lu discipline — §4.5
* The pure-vs-in-place divergence among shipped boundaries, *if* the prompt says which to follow — §3.3
* Low docstring coverage generally (81% of public functions) — §5.1
* Notebook size / example runtimes — §5.3

---

## Appendix A — commands run

Environment was built in the scratchpad, **not** in the worktree, so the repo was left untouched
(`git status` shows only the untracked `autoresearch/` directory this report lives in).

```bash
# --- toolchain ---
pip install --target ./uvtool uv                       # uv 0.12.15 (no uv on PATH)

# --- packaging checks ---
/…/envs/lettuce/bin/pip install -e . --no-deps --no-build-isolation --dry-run
#   -> ERROR: Package 'lettuce' requires a different Python: 3.11.3 not in '==3.12.*'
python -c "from setuptools.build_meta import prepare_metadata_for_build_wheel; ..."
#   -> OK, lettuce-0.2.3.dist-info; missing README.md tolerated silently

# --- environment ---
UV_PROJECT_ENVIRONMENT=$SP/venv uv sync --dry-run      # 30 pkgs, torch 2.8.0, zero nvidia-*
UV_PROJECT_ENVIRONMENT=$SP/venv uv sync
python -c "import lettuce"
#   -> ImportError: libcudnn.so.9: cannot open shared object file
UV_PROJECT_ENVIRONMENT=$SP/venv uv sync --extra cpu    # torch 2.8.0 -> 2.8.0+cpu
python -c "import lettuce as lt; print(lt.__version__)"
#   -> import OK in 2.3 s; 0+untagged.431.g121359d; torch 2.8.0+cpu; cuda False

# --- tests ---
python -m pytest tests -q --collect-only               # 1278 tests collected in 1.04 s
time python -m pytest tests -q --durations=15
#   -> 400 passed, 884 skipped, 26 warnings in 63.10s   (real 1m04s, user 7m39s)
#   -> ZERO failures on master
for d in boundary collision flow moments native reporter stencil unit util; do
    python -m pytest tests/$d -q; done                 # per-directory table in §2.2
python -m pytest tests/native -q -rs
#   -> SKIPPED [6] tests/native/__init__.py:4: CUDA is not available on this machine

# --- CLI integration (as CI runs it) ---
lettuce --no-cuda convergence --use-no-cuda_native     # 2nd-order convergence, exit 0

# --- examples (180 s timeout each) ---
python examples/00_simplest_TGV.py                     # OK, 0.627 MLUPS
python examples/01b_first_example_obstacle.py          # AssertionError: cuda device explicitly requested
python examples/02_converging_obstacle_flow.py         # exit 124 (timeout)
python examples/03_outputs_TGV.py                      # exit 124 (timeout)
python examples/simple_flows/LambOseenVortex.py        # AssertionError: Torch not compiled with CUDA enabled
python examples/development/example_progress_reporter_based_on_simplest_TGV.py   # same
python examples/advanced_flows/PartiallySaturatedObstacle.py                     # same
python examples/advanced_flows/FailingTGVandObstacle.py                          # same

# --- targeted probes (scripts in the scratchpad) ---
# no_streaming_mask across (n_pre, n_post) -> table in §3.1
# Context() constructor matrix            -> table in §4.2
# bool->uint8 conversion + masked_fill_   -> §4.1
# torch_stencil.opposite dtype            -> §4.3
# Boundary ABC partial implementation     -> §3.4
# boundary construction count / identity  -> §3.6:
#     boundary __init__ calls during Simulation.__init__: 2
#     sim.transformer[-1] is sim.post_boundaries[0]: False
# Obstacle(ctx, 16, ...)                  -> TypeError: 'int' object is not subscriptable

# --- static surveys ---
# AST docstring census over lettuce/      -> §5.1 (14/79 modules, 31/92 classes, 65/346 functions)
# importlib check of docs/modules.rst     -> 2 resolve, 15 dead, of 17
grep -rn "TODO\|FIXME\|XXX\|HACK" --include=*.py lettuce/ | grep -v _version.py | wc -l   # 8
grep -rn "LatticeBase\|from .base" --include=*.py .   # only base.py itself -> dead module
grep -rn "pre_boundaries\|post_boundaries" lettuce/ext/_flows/   # 9 post_, 0 pre_
grep -n "self.transformer" lettuce/_simulation.py     # line 70 only (assigned, never read)
grep -rn "pytest.skip\|xfail\|TODO" tests/            # 40 skips; disabled-test table in §2.4b
grep -rn -B2 "all_native_boundaries" tests/           # the int==list bug, in 3 files
python -c "import examples.advanced_projects.efficient_bounce_back_obstacle.boundary"
#   -> ModuleNotFoundError: No module named 'solid_boundary_data'
git ls-files | wc -l                                   # 259 tracked files, 5.0 MB
```

### Reproducing quickly

```bash
git clone <repo> && cd lettuce
uv sync --extra cpu            # the only install command verified to work on CPU
uv run pytest tests -q         # expect 400 passed, 884 skipped, ~65 s
uv run lettuce --no-cuda convergence --use-no-cuda_native
```
