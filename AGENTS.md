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

Measured on an M4: the full suite is **a few seconds** (152 tests, 4 xfail,
1 skip). If it has not finished in a minute, something is wrong — do not wait
longer, look at the log.

`tarhan demo` prints its table and exits. It only opens a plot window when
stdout is a terminal; pass `--show` to force one, `--save out.png` to write the
figure instead. It will not block a CI job or an agent shell.

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
