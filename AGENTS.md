# Notes for AI agents working in this repo

## Never guess how long something takes — wait for it

The failure this file exists to prevent: an agent starts a long command, has no
idea when it ends, and invents a duration. It sleeps 60 s, polls, sleeps again;
it burns turns on a job that finished in five seconds, or declares success on
one still running. Worst of all, a command that *never* exits looks exactly like
one that is merely slow.

Use the bundled runner. The contract is a file, not a timer.

```bash
python3 tools/job.py run <name> -- <command...>   # returns immediately
python3 tools/job.py wait <name>                  # blocks until it ends
```

`wait` exits with the job's own exit code and prints its duration and last
output. One call. Correct whether the job takes 2 s or 2 h — no interval to
guess, no polling loop to write.

```bash
python3 tools/job.py wait <name> --timeout 600     # distinguish hung from slow
python3 tools/job.py tail <name>                   # stream the log live
python3 tools/job.py list                          # what ran, what is running
python3 tools/job.py stop <name>                   # kill a runaway
```

Only `wait --timeout` can tell "hung" from "slow": on timeout it reports
`still_running`, so a blocking prompt or an unclosed plot window is a loud,
distinguishable failure instead of a silent wait. Logs land in `.jobs/`
(gitignored) and stream as the job produces them.

## Write the full command; do not use a shell variable

```bash
CLI="python3 tools/job.py"; $CLI list      # BROKEN
```

zsh — the macOS default — does not word-split unquoted expansions, so the whole
string is taken as one command name and you get `command not found: python3
tools/job.py`. It reads like a missing file, not a quoting bug. Each agent tool
call also starts a fresh shell, so the variable would be gone next call anyway.
Type the full path every time.

## Running things here

```bash
pip install -e ".[dev]"
python3 tools/job.py run tests -- pytest -q
python3 tools/job.py wait tests
```

Measured on an M4: the full suite is **a few seconds** (338 passed, 4 xfail).
If it has not finished in a minute, something is wrong — do not wait longer,
look at the log.

`tarhan demo` prints its table and exits. It only opens a plot window when
stdout is a terminal; pass `--show` to force one, `--save out.png` to write the
figure instead. It will not block a CI job or an agent shell.

## Ask the code what it can do, instead of guessing

```bash
tarhan capabilities list                      # every capability and its status
tarhan capabilities show <capability-id>      # limits, evidence, or why it is blocked
tarhan --format json capabilities list        # stdout is ONLY json; notes go to stderr
```

Read this before assuming a capability exists. A record marked `blocked` or
`planned` says *why* and what would unlock it, so "TARHAN cannot do X" is
answerable without reading the source. Capability ids carry the spatial
dimension and the time axis separately — `…1d.steady` and `…1d.transient` are
different capabilities, and the roadmap's eventual "4D" is `3d.transient`.

Two things matter when scripting it:

- **stdout carries the result and nothing else.** Progress, notes, warnings and
  the anvil feedback always go to stderr, in every format including `table`.
  So `tarhan --format json capabilities list | jq` is safe. The global
  flags come BEFORE the subcommand; argparse will not accept them after it.
- **The exit status is the machine-readable half of the answer.** `0` success;
  `2` bad input, including an unknown capability id or two runs that cannot be
  compared; `3` the capability is blocked, merely planned, or validated but not
  wired to `run solve` — the record still prints in full; `4` the solver did not
  converge, and no artifact is written; `5` an internal bug. Do not grep the
  prose — it will be reworded.

## Running something and keeping the result

```bash
tarhan run solve <capability-id> --bias 0.3 --output runs/
tarhan run show <run-id> --output runs/
tarhan compare runs <run-a> <run-b> --output runs/
tarhan compare runs <run-a> <run-b> --allow-build-diff --output runs/
```

A run leaves `runs/<problem-id>-<build-id>/` behind: `manifest.json`,
`input.lock.toml`, `provenance.json`, `metrics.json`, `fields.npz`,
`stdout.log`, `report.md`. The **problem id** hashes the capability, the fully
resolved inputs and the solver contract, so **re-running the same problem
overwrites rather than accumulates** — and a changed tolerance is a different
problem, landing elsewhere. The **build id** hashes what produced it: the
tarhan/python/numpy/scipy versions and a sha256 of the package's own source
bytes. The source hash is the load-bearing part — every commit between two
releases carries one dev version, so a build id derived from version strings
alone lets two different commits share a directory. Every `.py` under the
package counts, including modules no solve imports — the whole package is
treated as the build, deliberately, because which modules a solve imports is
not decidable without running it. The git commit is recorded alongside, with a
`dirty` flag, and is **excluded from the hash**: the same bytes reached from
two branches are one build, and `dirty` is a boolean over the whole tree, so
hashing it would move a solver's build id when a README changed. `git` answers
where the code came from; `source` answers what ran. The
timestamp is recorded but not hashed; include the clock and every run would be
unique by construction, which is the same as having no id at all.

`compare runs` refuses rather than ranks when the comparability contract does
not hold, and exits `2` naming the term that differs. Two solves at different
tolerances produce two numbers you *can* subtract; the difference means nothing.
The contract is capability, inputs, solver **and build**. A different build is
the one waivable term — comparing across code IS a real question — but it needs
`--allow-build-diff`, and every metric is then flagged, because with the inputs
held fixed a delta from a code change looks exactly like a physical effect.

**What the checksums are.** `manifest.json` records a sha256 of every other
file, so a result that changed after the run is caught. That is
accidental-corruption and casual-edit detection, **not** tamper-proofing: the
manifest is unsigned, so anyone who edits `metrics.json` and updates the digest
beside it passes. Directories written before checksums existed (schema v1) are
read, but `run show` and `compare runs` say plainly that nothing has verified
their contents.

`tarhan demo` keeps its own older 0/1 contract and is untouched by the above.

## GPU: no at today's scale, an open question at 2D/3D scale

Measured on an M4 at TARHAN's **current** 1D working size (n=2e4): MLX takes
0.631 ms against NumPy's 0.012 ms — about **50x slower**. Host-device transfer
dominates completely at this size. So for the code that exists today, GPU is a
pessimisation, and that is a measurement rather than an opinion.

**This verdict is about scale, not about GPUs.** It does not carry to 2D or 3D:

| | nodes |
|---|---|
| today, 1D | 20,000 |
| 2D, 256x256 mesh | 65,536 |
| 3D, 128^3 mesh | 2,097,152 |

At 3D sizes the arithmetic finally dwarfs the transfer, and two parts of the
solve behave very differently:

- **Assembly** (the edge loop computing Scharfetter-Gummel fluxes) is a
  gather/scatter over ~10^5-10^6 edges with no data dependence. That is a good
  GPU workload and MLX can express it.
- **The linear solve** is the dominant cost, and it is the blocker: **MLX has no
  sparse support today** (`mx.sparse` does not exist; `mx.linalg` is dense only,
  as of 0.31.1). A dense solve is not an option — 65k nodes dense in float64 is
  ~34 GB. So a GPU path needs either a matrix-free iterative solver (CG/GMRES
  with the stencil applied directly, no assembled matrix) or hand-rolled sparse
  matvec via gather/scatter.

So the honest position: **this has not been re-measured since 2D landed.** The
verdict above is a 1D measurement and nothing more. 2D is now built as far as
stage 2D-3′ and nobody has run the comparison again at 2D node counts. Expect
the answer to depend on the solver choice made in `docs/DESIGN-2D.md` rather
than on the hardware — a direct sparse LU stays on the CPU, while a matrix-free
iterative scheme is where a GPU could pay.

(This paragraph promised a re-measurement once 2D arrived, and kept promising it
after 2D arrived. The documentation test caught it.)

## Scope — do not overclaim

0D and 1D: `pemfc0d`, `pn1d`, `sofc1d`, `chronoamp1d`, `diffusion1d`.

**2D exists but is partial, and the difference matters.** `numerics/mesh.py`,
`numerics/assemble.py`, `backend.solve_sparse` and `models/pn2d.py` are built,
and stages 2D-0 through 2D-3′ of `docs/DESIGN-2D.md` §5 are validated against
DEVSIM: a 1D problem on a 2D mesh, the equilibrium pn junction, diode I–V with
ideality 1.012, and electrostatic contact charge. **2D-3 and 2D-4 are BLOCKED**,
for reasons recorded there. Not built at all: AC or small-signal, circuit
coupling, mesh generation, 3D, and any 2D entry on the MCP tool surface. §5's
table is the authority — quote it rather than this paragraph if the two drift.

There is no 3D solver in this repo. DEVSIM (installed in the venv as the
reference simulator) ships its own 2D/3D examples under
`devsim_data/testing/` — those belong to DEVSIM, not to TARHAN. Do not describe
them as this project's capability.
