#!/usr/bin/env python3
"""A file:line citation this tree SHIPS must land on what it names.

The EDC manifest reconcile shipped explanatory comment carrying six single-line citations, two of
them wrong: one pointed at line 31 of pipeline/tier_routing.py, which is BLANK (TIERS is three
lines earlier), and one pointed at line 41 of pipeline/domain.py, the FileNotFoundError raise,
when the manifest re-read it meant is the line after. All five gate commands were green while the
prose pointed at nothing, and a reader who lands on a blank line concludes the file moved under
them. A comment hunk also SHIFTS the file it edits, so a citation into that same file must be
measured after the edit, not before.

Two layers, both live:
  (i)  an exhaustive blank/out-of-range guard over every single-line citation in the files this
       change ships — no per-citation authoring, so it cannot fall behind them. Its REACH is held
       by two literal registries, POLICED (the file set) and MIN_CITATIONS (the per-file floor),
       because a scanner can be neutered by shrinking what it scans as easily as by loosening how
       it scans, and only the second of those looks like an edit to this file;
  (ii) an anchor map for the citations whose target is a NAMED construct, which is the half a
       line-number check cannot do: a non-blank cited line does not make it `TIERS`.

WHAT THIS FILE DOES NOT SEE, stated because a guard's blind spots are the next stale claim. It
polices `file.py:NNN` and nothing else. Line RANGES (`foo.py:12-30`) are deliberately out of scope —
a range names a region, and asserting a token on its first line would be a guess, so they are
verified by reading. So is every MEASURED-CLAIM sentence these files ship: the counts, shares and
distributions in the manifest comments are exactly as unguarded as the citation that motivated this
file, and one such claim shipped false in the very change that added this guard. That is a separate
instrument, and it is not this one.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_shipped_citations.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C

# Every file this change ships or edits that can carry prose. Adding a file here costs nothing and
# is the only thing that keeps layer (i) exhaustive rather than sampled.
CITING = (
    "configs/domains/edc.yaml",
    "scripts/plan_install.py",
    "scripts/pipeline/test_recipe_profiles.py",
    "scripts/pipeline/test_stage_lockstep.py",
    "scripts/pipeline/test_shipped_citations.py",
)

# The files layer (i) MUST be reading, and the minimum each one must still carry. Both registries
# are LITERALS and that is the whole point: `POLICED = set(CITING)` was measured and it does not
# work, because an assertion derived from the thing it polices shrinks with it — dropping the two
# zero-citation entries from CITING left the derived form GREEN with an out-of-range citation
# planted in a dropped file. A floors-only check misses the same mutation independently, because
# the dropped files carry no citations to fall below a floor. Only the literal enumeration reds.
POLICED = frozenset({
    "configs/domains/edc.yaml",
    "scripts/plan_install.py",
    "scripts/pipeline/test_recipe_profiles.py",
    "scripts/pipeline/test_stage_lockstep.py",
    "scripts/pipeline/test_shipped_citations.py",
})
MIN_CITATIONS = {
    "configs/domains/edc.yaml": 3,
    "scripts/pipeline/test_recipe_profiles.py": 6,
    "scripts/pipeline/test_shipped_citations.py": 2,
}

# `(?![\d-])` is load-bearing: without it a RANGE like `derive_plan.py:499-515` is truncated to its
# first two digits and read as a single-line citation nobody wrote, so this file would police a line
# at random. Proven in test_both_layers_can_actually_fail. Note the prose above is deliberately the
# only place a range appears — spelling the truncated form out would itself become a citation here,
# which is the same trap one level up.
_CITATION = re.compile(r"([\w./-]+\.py):(\d+)(?![\d-])")

# (citing file, regex with ONE capture group for the line, cited file, token that must be on it).
ANCHORS = [
    # The load-bearing one, and the one this file originally missed: this citation is the SOLE
    # evidence for the directive its own comment issues ("only rejects UNKNOWN keys; it never
    # requires completeness ... Do not 'complete' it"). A net-zero-line-count refactor of
    # plan_install.py drifts line 145 onto `if profile not in RECIPE_PROFILES:` — a check that DOES
    # reject — so a maintainer following the pointer reads the OPPOSITE of the invariant, while
    # every gate command stays green. Layer (i) is structurally blind to that: the drifted line is
    # neither blank nor out of range. Only a construct anchor catches it.
    # The joiner is `\s*(?:\n\s*#\s*)?` and not a bare `\n\s*#\s*` on purpose: the citation and the
    # phrase sit on two comment lines today, but re-wrapping them onto one is a cosmetic edit that
    # must not red a guard about line numbers. Measured both ways: the two-line form and the
    # reflowed one-line form both match, and the net-zero-line-count refactor described above
    # (which walks line 145 onto `if profile not in RECIPE_PROFILES:`) is still caught under
    # EITHER layout.
    ("configs/domains/edc.yaml",
     r"plan_install\.py:(\d+)\s*(?:\n\s*#\s*)?only rejects UNKNOWN keys",
     "scripts/plan_install.py", 'set(RECIPE_PROFILES["generic_facts"])'),
    ("configs/domains/edc.yaml", r"tier_routing\.py:(\d+) accepts",
     "scripts/pipeline/tier_routing.py", "TIERS = frozenset"),
    ("configs/domains/edc.yaml", r"extract_structured\.py:(\d+) compares",
     "scripts/extract_structured.py", "tier_for"),
    ("scripts/pipeline/test_recipe_profiles.py", r"construction \(domain\.py:(\d+)\)",
     "scripts/pipeline/domain.py", "yaml.safe_load"),
    ("scripts/pipeline/test_recipe_profiles.py", r"unioned by plan_install\.py:(\d+)",
     "scripts/plan_install.py", 'domain.spec.get("tiers")'),
    # The two raise sites the tier-vocabulary fixture names in its own failure messages: if these
    # drift, that fixture reports a red against a line that no longer raises anything.
    ("scripts/pipeline/test_recipe_profiles.py", r"per-predicate \(tier_routing\.py:(\d+)\)",
     "scripts/pipeline/tier_routing.py", "has illegal tier"),
    ("scripts/pipeline/test_recipe_profiles.py", r"tiers block \(tier_routing\.py:(\d+)\)",
     "scripts/pipeline/tier_routing.py", "tiers block:"),
]


def _resolve(name: str) -> Path:
    """Cited path or basename -> the one file it can mean. Ambiguity is a failure, not a coin flip.

    The first version of this function matched on the BASENAME ALONE, which made a whole class of
    citation unwriteable: it discarded the directory the author wrote and then refused the leftover
    basename as ambiguous, so a correct, full, verified path reded the commit gate with "cite a
    path, not a basename" — at an author who had cited a path. That was not hypothetical. Seven
    duplicated basenames already exist under scripts/ (__init__.py, __main__.py, mfw.py, news.py,
    official.py, reddit.py, youtube.py), and the two __main__ files are the entry points of EDC's
    OWN declared scrape and extract stages, so the two files this manifest was most likely to cite
    were exactly the two it could not.

    The repair is the fourth line below: the search keeps every directory component the author DID
    write, as a COMPONENT-wise suffix match — plus a GUARDED direct lookup ahead of it. The first
    version of this docstring called the direct prepend dead code; two reviewers then proved it
    live from two angles: a citation to a file OUTSIDE the scripts/**-plus-repo-root search scope
    (configs/…, docs/…) reded as "no such file" even though the path was correct, and a full-path
    citation whose basename collides elsewhere reded as ambiguous even though exactly one file
    matches the written path. The prepend is guarded (no absolute paths, no `..`) so the refusals
    below keep their teeth — an UNGUARDED `C.ROOT / name` resolves absolute citations, because
    `Path("/a") / "/abs/b"` is `/abs/b`. Components rather than a string `endswith` because a
    string suffix accepts a TRUNCATED leading component: `ne/tier_routing.py` is a string suffix of
    `scripts/pipeline/tier_routing.py` and is not a path anybody meant. What the one line does, in
    the four shapes authors write:
      * a full repo-relative path resolves, however many files share its basename;
      * a bare `tier_routing.py` still resolves — with no directories written the filter is a no-op,
        which is why every citation this tree already ships is unaffected;
      * a scripts-relative `pipeline/tier_routing.py` resolves, since PYTHONPATH=scripts makes that
        the form a reader of the module path would naturally write;
      * a WRONG-directory path resolves to nothing and says so — `scripts/pipeline/plan_install.py`
        must not be satisfied by `scripts/plan_install.py`, because certifying a path a reader
        cannot follow is the same "the pointer lands on nothing" failure this file exists to stop.
        The same line refuses an absolute path and one containing `..`, so a citation can never be
        checked against a file outside this tree.

    The search scope is scripts/** plus the repo root, deliberately: rglob over the whole repository
    would walk data/. A citation to a .py outside that scope reds, and the message says why.
    """
    wrote = Path(name).parts
    if not Path(name).is_absolute() and ".." not in wrote:
        direct = C.ROOT / name
        if direct.is_file():
            return direct
    matches = [p for p in (C.ROOT / "scripts").rglob(wrote[-1]) if p.is_file()]
    matches += [p for p in C.ROOT.glob(wrote[-1]) if p.is_file()]
    matches = [p for p in matches if p.relative_to(C.ROOT).parts[-len(wrote):] == wrote]
    assert matches, (
        f"citation names {name!r} but no such file exists under scripts/ or the repo root — the "
        f"path may name a directory the file is not in. Cite a repo-relative path, or a basename.")
    exact = [p for p in matches if p.relative_to(C.ROOT).parts == wrote]
    if len(exact) == 1:
        return exact[0]
    assert len(matches) == 1, (
        f"citation names {name!r}, which is ambiguous: "
        f"{sorted(str(p.relative_to(C.ROOT)) for p in matches)} — cite a path, not a basename")
    return matches[0]


def _line(path: Path, number: int) -> str | None:
    """The cited line, or None when the citation is out of range."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[number - 1] if 0 < number <= len(lines) else None


def _code(line: str | None) -> str:
    """The line with any trailing comment removed. A comment naming a construct is documentation."""
    return (line or "").split("#", 1)[0]


def test_no_shipped_citation_points_at_a_blank_or_absent_line() -> None:
    for citing in CITING:
        text = (C.ROOT / citing).read_text(encoding="utf-8")
        for name, number in _CITATION.findall(text):
            cited = _resolve(name)
            index = int(number)
            line = _line(cited, index)
            assert line is not None, (citing, f"{name}:{index}", "out of range")
            assert line.strip(), (citing, f"{name}:{index}", "cites a BLANK line")


def test_shipped_citations_land_on_what_they_name() -> None:
    for citing, pattern, cited, want in ANCHORS:
        text = (C.ROOT / citing).read_text(encoding="utf-8")
        match = re.search(pattern, text)
        assert match, f"{citing}: citation pattern {pattern!r} no longer present"
        number = int(match.group(1))
        path = C.ROOT / cited
        got = _line(path, number)
        assert got is not None and want in got, (
            f"{citing} cites {cited}:{number} but {want!r} is not there; line {number} is "
            f"{(got if got is not None else '<out of range>')!r}; {want!r} lives at "
            f"{[i + 1 for i, line in enumerate(path.read_text(encoding='utf-8').splitlines()) if want in line]}")


def test_both_layers_can_actually_fail() -> None:
    """Guard the guards: a text scanner is easy to write so that it matches nothing.

    Layer (i) is a real check only if _CITATION finds the citations that are present and rejects a
    blank target; layer (ii) only if the SAME comparison rejects a neighbouring line. Both are
    proven here in-process against the real tree, so a later edit that empties CITING, loosens the
    regex, or drops the blank-line assertion is not silently green.
    """
    found = _CITATION.findall("see tier_routing.py:28 and domain.py:42")
    assert found == [("tier_routing.py", "28"), ("domain.py", "42")], found
    assert _CITATION.findall("range derive_plan.py:499-515 is out of scope") == [], "range leaked in"

    # _resolve's seven shapes, driven in-process rather than by planting citations in a real file —
    # which keeps the manifest free of citations that exist only to feed a fixture. Every one is a
    # form a future author will write, and the first version of this function got the first of them
    # WRONG: it threw the directory away and then refused the basename as ambiguous, so the two
    # files most likely to be cited here were the two that could not be. Measured: deleting the
    # single filter line in _resolve reds five of these seven, which is what makes it a check and
    # not a decoration. The two that survive are the bare basename and the scripts-relative path,
    # which is expected — neither has a directory component the filter could discriminate on.
    duplicated = "scripts/extract/__main__.py"          # basename shared with scripts/scrape/
    assert _resolve(duplicated) == C.ROOT / duplicated, "a full path must resolve as written"
    assert _resolve("pipeline/tier_routing.py") == C.ROOT / "scripts/pipeline/tier_routing.py", (
        "a scripts-relative path must resolve — its directories are a suffix of the real ones")
    # The last two are the absolute and the `..` form of a path that DOES exist and IS in this tree.
    # They must still be refused: a citation is repo-relative or it is not checkable.
    for bad, expect in (("__main__.py", "ambiguous"),                    # genuinely bare and shared
                        ("scripts/pipeline/plan_install.py", "no such file"),  # wrong directory
                        ("ne/tier_routing.py", "no such file"),          # truncated leading component
                        (str(C.ROOT / "scripts/pipeline/tier_routing.py"), "no such file"),
                        ("scripts/../scripts/pipeline/tier_routing.py", "no such file")):
        try:
            _resolve(bad)
        except AssertionError as error:
            assert expect in str(error), (bad, expect, str(error))
        else:
            raise AssertionError(f"_resolve({bad!r}) resolved — it must refuse")

    # Layer (i)'s REACH is pinned as a registry, not as a total. The total was `assert total == 11`
    # and it was wrong in both directions, measured:
    #   * it OVER-fired. Any new CORRECT citation in a policed file reded the commit gate with a
    #     bare per-file dict and no sentence saying what was wrong — so the practice this file
    #     establishes, documenting an invariant with an accurate file:line pointer, broke the build.
    #   * it UNDER-fired on the drift it existed to catch. Deleting two policed citations and adding
    #     two elsewhere keeps the sum at 11, so the published receipt went wrong in three of five
    #     entries with every fixture green.
    # POLICED and MIN_CITATIONS replace it: erosion of the FILE SET and erosion of each file's COUNT
    # are both named, and growth of either is free.
    per_file = {c: len(_CITATION.findall((C.ROOT / c).read_text(encoding="utf-8"))) for c in CITING}
    total = sum(per_file.values())
    assert POLICED <= set(CITING), (
        sorted(POLICED - set(CITING)),
        "these files were dropped from CITING, so layer (i) stopped reading them — restore them, "
        "or remove them from POLICED on purpose")
    # MIN_CITATIONS ⊆ POLICED ⊆ CITING, so per_file is guaranteed to have every floored key: a
    # mistyped floor is an AssertionError naming it, never a KeyError from the loop below.
    assert set(MIN_CITATIONS) <= POLICED, sorted(set(MIN_CITATIONS) - POLICED)
    for path, floor in MIN_CITATIONS.items():
        assert per_file[path] >= floor, (
            path, per_file[path], floor,
            "a policed citation was DELETED, not merely moved — restore it or lower the floor on purpose")
    # Two cross-layer floors, and they fail on DIFFERENT things. The first reads the OBSERVED total,
    # so it fires when layer (ii) claims more targets than layer (i) can currently see; because the
    # floors below already sum to 11, that needs ANCHORS to grow past 11, and it is the looser of
    # the two — kept only because it is the one that reads the live count. The second reads the
    # FLOORS, so it fires when they stop GUARANTEEING a target per anchor, which is the only thing
    # standing between MIN_CITATIONS = {} and a green fixture that polices nothing.
    assert total >= len(ANCHORS), (total, len(ANCHORS), "layer (i) sees fewer citations than layer (ii)")
    assert sum(MIN_CITATIONS.values()) >= len(ANCHORS), (
        sorted(MIN_CITATIONS.items()), len(ANCHORS),
        "the floors no longer guarantee a citation per anchor — layer (i) can now erode to nothing "
        "while every fixture stays green")

    tier_routing = C.ROOT / "scripts/pipeline/tier_routing.py"
    lines = tier_routing.read_text(encoding="utf-8").splitlines()
    blank = next(i + 1 for i, line in enumerate(lines) if not line.strip())
    assert _line(tier_routing, blank) is not None and not _line(tier_routing, blank).strip(), (
        "layer (i)'s blank-line assertion cannot fire — no blank line found to prove it")
    assert _line(tier_routing, len(lines) + 1) is None, "layer (i)'s out-of-range branch is dead"

    # Layer (ii) must discriminate by LINE, not merely find the token somewhere in the file — and
    # that has to hold for EVERY anchor. Proving it for ANCHORS[0] alone let the other entries read
    # as coverage without ever being shown to discriminate, which is how an anchor whose token also
    # sits on line n±1 would ship as protection against an off-by-one it cannot detect.
    checked = 0
    for citing, pattern, cited, want in ANCHORS:
        match = re.search(pattern, (C.ROOT / citing).read_text(encoding="utf-8"))
        assert match, f"{citing}: citation pattern {pattern!r} no longer present"
        number = int(match.group(1))
        path = C.ROOT / cited
        # The bound MUST be recomputed per cited file. `lines` above belongs to tier_routing.py
        # (322 lines); reusing it here would bound domain.py, which has 107, by 322 — the anchors
        # happen to survive that today, which is exactly why it has to be fixed while it is cheap.
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        assert want in (_line(path, number) or ""), (citing, cited, number, want)
        neighbours = [n for n in (number - 1, number + 1) if 0 < n <= n_lines]
        assert neighbours, (citing, cited, number, "no neighbouring line to prove the anchor discriminates")
        # Compared against the CODE of each neighbour, not its raw text. A comment that names the
        # cited construct is documentation, and documenting a construct beside it must not red a
        # guard about line numbers: writing `# TIERS = frozenset — the routing vocabulary` on the
        # blank line above TIERS shifts nothing, leaves every citation exactly correct, and reded
        # this fixture with "this anchor cannot detect an off-by-one". `_code` handles the whole-line
        # comment AND the trailing one; a line-level "starts with #" FILTER of the neighbour list
        # handles only the first, and it introduces a second false red of its own — for an anchor
        # whose neighbours are BOTH comments the filtered list is empty and `assert neighbours`
        # fires. Which is why the assertion above survives rather than being deleted: transforming
        # the compared TEXT drops nothing from the list, so both-comments cannot empty it (measured
        # green), and `assert neighbours` is left as what it always was — a fail-closed guard for a
        # cited one-line file, reachable only there. `_code` splits on the first '#', so a '#'
        # inside a string literal would truncate a neighbour early; that can only make the probe
        # more permissive, never red it falsely, and no anchor neighbour holds one today.
        #
        # DISCLOSED LIMIT, because this probe's promise is narrower than it reads: it proves the
        # anchor discriminates against a neighbour's CODE. If the cited line itself were to become a
        # comment that names the construct, `want in _line(...)` above would still pass. Catching
        # that needs comment-aware parsing of the anchored line, which is a different instrument;
        # the failure it would catch (a pointer landing on prose ABOUT the construct rather than the
        # construct) is also a far weaker one than the inversion ANCHORS[0] exists for.
        assert all(want not in _code(_line(path, n)) for n in neighbours), (
            f"{want!r} is on a neighbour of {cited}:{number} too — this anchor cannot detect an off-by-one")
        checked += 1

    # Layer (ii) must not be able to go VACUOUS, and looping is what made that reachable. The line
    # this loop replaced was `citing, pattern, cited, want = ANCHORS[0]`, so an emptied ANCHORS
    # raised IndexError and the fixture went red BY ACCIDENT. Measured on the looped version:
    # `ANCHORS = []` leaves every assertion above satisfied (the reach check is `11 >= 0`) and the
    # whole file GREEN. Widening the coverage therefore has to re-assert non-emptiness on purpose.
    #
    # Three floors, and none of them is redundant BECAUSE OF THE ORDER. `checked == len(ANCHORS)`
    # catches a skipped entry at any size; `assert ANCHORS` is the only one that can fire at zero
    # and is what names that case; `checked >= 7` is the only one that can fire at one through six.
    # Erosion is gradual, not binary — measured: ANCHORS cut from 7 entries to 1 leaves all three
    # fixtures green without a count floor.
    #
    # An earlier version of this file rejected the count floor on the grounds that it charges the
    # sanctioned revert of the predicate-tier hunk in configs/domains/edc.yaml an extra edit, since
    # two anchors cite from inside that hunk. That rationale was measured and it is FALSE: reverting
    # the hunk ALREADY reds this fixture, on `citation pattern ... no longer present`, whether or not
    # a count floor exists. The floor costs the revert nothing, and the revert's cost is unchanged at
    # exactly two test files.
    assert checked == len(ANCHORS), (checked, len(ANCHORS), "an anchor was skipped, not checked")
    assert ANCHORS, "layer (ii) is empty — every construct anchor was dropped"
    assert checked >= 7, (checked, "the construct-anchor layer eroded")


if __name__ == "__main__":
    test_no_shipped_citation_points_at_a_blank_or_absent_line()
    test_shipped_citations_land_on_what_they_name()
    test_both_layers_can_actually_fail()
    print("OK — every shipped file:line citation lands on what it names")
