# Contributing to TARHAN

Contributions are welcome. Two things are different here from most projects, and
both are worth knowing before you spend time.

## 1. The acceptance bar is evidence, not opinion

TARHAN is built validation-first. A number does not enter this codebase because it
looks right — it enters because it is tied to something checkable. Concretely, a
new numeric threshold must be one of:

- **source-pinned** — the last printed digit of a published number, with the
  citation (book/paper, section, page) in the code;
- **an order or limit** — the asserted quantity *is* a measured convergence order
  or a Richardson-extrapolated limit;
- **exact structural** — a machine-precision identity, conservation law,
  monotonicity, sign, or domain guard;
- **measured-margin** — a regression guard set with deliberate margin over a value
  you measured *and recorded in the code next to the assertion*.

A tolerance chosen so the test would pass is not acceptable. The README section
"The honesty model" states how many thresholds currently fall in each class,
including the ones that are still underived — that count is published on purpose,
and pull requests should not make it worse silently.

Physics functions in `tarhan.physics` carry an explicit honesty tier
(`first-principles (oracle-verified)` / `textbook (reproduced)` / `empirical fit`).
New formulas need one.

If you find that an existing claim in this repository is wrong, that is a valuable
contribution on its own. Open an issue with the evidence. It has happened before
and the corrections are recorded rather than quietly fixed.

## 2. The Contributor License Agreement

TARHAN is released under **AGPL-3.0-or-later**, and a **separate commercial
license** is offered to users who cannot accept AGPL terms.

Keeping that possible requires a license grant from contributors. Without it, one
merged pull request would permanently end the project's ability to offer
commercial terms, because every past contributor would have to be tracked down
and asked.

So: before your first pull request is merged, please read [`CLA.md`](CLA.md) and
include this line in the pull request description:

```
I have read the CLA.md document and I hereby sign the CLA.
```

**You keep the copyright to your work.** The CLA is a license grant, not an
assignment. It is one page and says so explicitly.

You sign once; it covers all your later contributions here. If you are
contributing on behalf of an employer, say so in the pull request.

There is deliberately no bot enforcing this — the two established CLA bots are
both unmaintained, and the GitHub Action variant requires write permissions on
pull requests from forks, which is not a trade this project is willing to make for
its current contribution volume. The pull request template carries the checkbox
instead.

## Practical steps

```bash
git clone https://github.com/emircbngl/tarhan
cd tarhan
pip install -e ".[dev]"
pytest                      # the Layer-0 validation catalog must stay green
tarhan demo                 # zero-config smoke check
tarhan demo --case diode    # flagship drift-diffusion solver
```

Optional extras: `.[oracle]` adds the DEVSIM cross-validation tests,
`.[mcp]` adds the MCP server.

CI runs the suite on Linux, macOS and Windows across Python 3.11 and 3.13, then
builds a wheel and installs it into a clean environment. If it passes locally but
fails there, the difference is usually a platform assumption — say so in the pull
request rather than pinning around it.

## Reporting a problem

Include what you ran, what you expected, and what happened. For a numerical
disagreement, the most useful report gives the case parameters and the two numbers
being compared — a difference is only meaningful next to the quantity it is a
difference *of*.
