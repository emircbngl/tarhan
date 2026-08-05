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

## Scope — do not overclaim

This is 0D/1D only: `pemfc0d`, `pn1d`, `sofc1d`, `chronoamp1d`, `diffusion1d`.
There is no 2D or 3D solver in this repo. DEVSIM (installed in the venv as the
reference simulator) ships its own 2D/3D examples under
`devsim_data/testing/` — those belong to DEVSIM, not to TARHAN. Do not describe
them as this project's capability.
