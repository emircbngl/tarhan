"""Layer-0 for the prose: documentation claims checked against the tree.

Why this exists. Twice in review, the defect was not code but a sentence I wrote
that my own later work made false — "there is no 2D solver in this repo" three
paragraphs under a header announcing 2D was partly built; "EdgeGeometry cannot
express this today" about a field implemented two commits later; AGENTS.md still
promising to "re-measure when 2D lands" after it had landed. Each time the
structured record (the §5 table) was updated and the surrounding prose was
trusted to have kept up. It had not.

What this can and cannot catch, stated plainly because the boundary matters:

* CATCHES a capability claim that contradicts the code, a documented source path
  that does not exist, and a stage marked DONE with no test behind it.
* DOES NOT CATCH a wrong *measurement*. In the same review round a merge count
  of "2954 to 2517" was carried across from a different mesh — a number never
  measured, presented as measured. No static check can tell a plausible wrong
  number from a right one; only re-running it, or an independent reviewer, can.
  This file is not a substitute for either.

Historical sections are exempt: a section whose heading or opening paragraph
says "Historical" describes the past on purpose, and forbidding that would make
the documentation erase its own reasoning.

One consequence, learned the first time this test ran. These documents correct
themselves in place, leaving the old claim beside the correction — and this test
cannot tell a corrected quotation from a live claim, because both are the same
string. Teaching it to would be guesswork about quotation marks. So the
discipline is the other way round: when correcting a sentence, DESCRIBE the old
wording rather than reproducing it. The test's bluntness is the feature; a
checker that tried to be clever about intent would be one more thing that can be
wrong without saying so.
"""
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
#: The roadmap is in this list on purpose. A plan document is the likeliest of
#: them all to go stale, because it describes things that do not exist yet and
#: nothing breaks on the day one of them quietly starts existing.
DOCS = ["README.md", "AGENTS.md", "CHANGELOG.md", "CONTRIBUTING.md",
        "docs/DESIGN-2D.md", "docs/ROADMAP-CLI-TUI.md"]


def _sections(text):
    """Split a markdown file into (heading, body, first-line-number) triples."""
    out, heading, buf, start = [], "(preamble)", [], 1
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("#"):
            out.append((heading, "\n".join(buf), start))
            heading, buf, start = line.lstrip("# ").strip(), [], lineno
        else:
            buf.append(line)
    out.append((heading, "\n".join(buf), start))
    return out


def _live_sections(doc):
    """Sections that claim to describe the present."""
    text = (REPO / doc).read_text(encoding="utf-8")
    for heading, body, lineno in _sections(text):
        opening = body.strip().split("\n\n")[0] if body.strip() else ""
        if "Historical" in heading or "Historical" in opening:
            continue
        yield heading, body, lineno


def _has_2d_solver():
    return (REPO / "src/tarhan/models/pn2d.py").exists()


def _edge_geometry_has_shares():
    sys.path.insert(0, str(REPO / "src"))
    from tarhan.numerics.mesh import EdgeGeometry
    return "facet_shares" in EdgeGeometry.__dataclass_fields__


#: Each entry pairs a phrase with a LIVE check rather than a constant, so the
#: test starts failing when the code catches up with the prose — which is the
#: moment the prose goes stale, and the moment nobody is looking.
STALE_CLAIMS = [
    (r"no 2D anything", _has_2d_solver),
    (r"There is no 2D or 3D solver", _has_2d_solver),
    (r"0D/1D only", _has_2d_solver),
    (r"No 2D code exists", _has_2d_solver),
    (r"when 2D lands", _has_2d_solver),
    (r"`?EdgeGeometry`? cannot express", _edge_geometry_has_shares),
]


@pytest.mark.parametrize("doc", DOCS)
def test_no_live_section_denies_something_the_code_has(doc):
    """A capability claim must not survive the capability arriving."""
    offences = []
    for heading, body, lineno in _live_sections(doc):
        for pattern, still_true in STALE_CLAIMS:
            if re.search(pattern, body) and still_true():
                offences.append(f"{doc} §{heading!r} (from line {lineno}) "
                                f"still says {pattern!r}")
    assert not offences, (
        "documentation denies something the code now has:\n  "
        + "\n  ".join(offences)
        + "\n(if the section describes the past on purpose, mark it Historical)")


@pytest.mark.parametrize("doc", DOCS)
def test_every_source_path_mentioned_in_the_docs_exists(doc):
    """A renamed module must not be discoverable only through a reader's confusion.

    Prose names modules the way people say them — `mesh.py`, not
    `src/tarhan/numerics/mesh.py` — so a bare basename is resolved anywhere
    under the package. That is the point: it catches a module that was renamed
    or deleted, which is what a reader would trip over, without forcing the
    documentation to spell out full paths it has no reason to.

    DEVSIM's own case files are referenced too. Those resolve against the
    installed data directory and are skipped when it is absent, since they are
    not this repository's to guarantee.
    """
    text = (REPO / doc).read_text(encoding="utf-8")
    devsim_data = pathlib.Path(sys.prefix) / "devsim_data"
    ours = {p.name for p in REPO.rglob("*.py")
            if ".venv" not in p.parts and "__pycache__" not in p.parts}
    missing = []
    for match in re.finditer(r"`([\w./-]+\.py)`", text):
        rel = match.group(1)
        if any((REPO / base / rel).exists()
               for base in ("", "src", "src/tarhan")):
            continue
        if rel.split("/")[-1] in ours:
            continue
        if devsim_data.exists():
            if list(devsim_data.rglob(rel.split("/")[-1])):
                continue
        else:
            continue          # cannot judge DEVSIM's files without DEVSIM
        missing.append(rel)
    assert not missing, f"{doc} references source files that do not exist: {missing}"


def test_every_stage_marked_done_has_a_test_behind_it():
    """The §5 table is the authority, so it must not be able to lie.

    A row saying DONE with no test is exactly the failure this repository
    refuses everywhere else: a claim with no evidence attached.
    """
    text = (REPO / "docs/DESIGN-2D.md").read_text(encoding="utf-8")
    tests = {p.name for p in (REPO / "validation").rglob("test_*.py")}
    expected = {
        "2D-0": ("test_assemble.py", "test_pn2d_equilibrium.py"),
        "2D-1": ("test_2d1_pn2d_equilibrium.py",),
        "2D-2": ("test_2d2_pn2d_iv.py",),
        "2D-3′": ("test_2d3_capacitance.py",),
    }
    done = [line.split("|")[1].strip() for line in text.splitlines()
            if line.startswith("| 2D-") and "**DONE**" in line]
    assert done, "no stage is marked DONE; has the table format changed?"
    for stage in done:
        assert stage in expected, (
            f"stage {stage} is marked DONE but this test does not know which "
            "test backs it; add it to `expected` rather than deleting the row")
        for name in expected[stage]:
            assert name in tests, (
                f"stage {stage} is marked DONE but {name} does not exist")


def test_blocked_stages_say_why_in_the_same_document():
    """A blocked row must point at its reason, not merely wear the label.

    "BLOCKED" with no explanation is how a scope decision decays into folklore.
    """
    text = (REPO / "docs/DESIGN-2D.md").read_text(encoding="utf-8")
    blocked = [line for line in text.splitlines()
               if line.startswith("| 2D-") and "BLOCKED" in line]
    assert blocked, "no blocked stage found; has the table format changed?"
    for line in blocked:
        stage = line.split("|")[1].strip().replace("~~", "")
        assert re.search(rf"###[^\n]*{re.escape(stage)}[^\n]*blocked", text,
                         re.IGNORECASE), (
            f"stage {stage} is marked BLOCKED but no section explains why")
