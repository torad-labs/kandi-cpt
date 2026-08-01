#!/usr/bin/env python3
"""Red-green proofs for alias+inverse-closed split_v2, and EDC split-coverage assertions:
each registry unit is covered (split + non-empty holdout + partitions its current graph)
or declared by name below."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.split as S  # noqa: E402


def _t(s, p, o):
    return {"subject_id": s, "predicate": p, "object": o}


def test_v2_alias_closed_no_leak():
    # Same person under two ids that resolve to one canonical
    id_map = {"dom:character:gio_boss": "dom:character:giovanni"}
    triples = [
        _t("dom:character:giovanni", "leads", "dom:org:tr"),
        _t("dom:character:gio_boss", "member_of", "dom:org:tr"),
        _t("dom:character:proton", "member_of", "dom:org:tr"),
        _t("dom:character:petrel", "member_of", "dom:org:tr"),
        _t("dom:character:archer", "member_of", "dom:org:tr"),
        _t("dom:character:ariana", "member_of", "dom:org:tr"),
        _t("dom:location:viridian", "located_in", "dom:region:kanto"),
        _t("dom:location:pewter", "located_in", "dom:region:kanto"),
        _t("dom:location:cerulean", "located_in", "dom:region:kanto"),
        _t("dom:location:vermilion", "located_in", "dom:region:kanto"),
    ]
    train, holdout, meta = S.make_split_v2(triples, holdout_frac=0.3, seed=0, id_map=id_map)
    assert meta["grouping"] == "alias+inverse"
    leak = S.alias_leak_report(triples, train, holdout, id_map)
    assert leak["zero_alias_leaks"], leak["alias_leaks"]
    # The two giovanni ids must be on the same side
    g1 = S.triple_id(triples[0])
    g2 = S.triple_id(triples[1])
    assert (g1 in holdout) == (g2 in holdout)


def test_v2_inverse_closed():
    triples = [
        _t("dom:character:gio", "leads", "dom:org:tr"),
        _t("dom:org:tr", "led_by", "dom:character:gio"),
        _t("dom:character:a", "member_of", "dom:org:tr"),
        _t("dom:character:b", "member_of", "dom:org:tr"),
        _t("dom:character:c", "member_of", "dom:org:tr"),
        _t("dom:character:d", "member_of", "dom:org:tr"),
        _t("dom:location:x", "located_in", "dom:region:k"),
        _t("dom:location:y", "located_in", "dom:region:k"),
        _t("dom:location:z", "located_in", "dom:region:k"),
        _t("dom:location:w", "located_in", "dom:region:k"),
    ]
    train, holdout, meta = S.make_split_v2(triples, holdout_frac=0.25, seed=1, id_map={})
    t_leads = S.triple_id(triples[0])
    t_led = S.triple_id(triples[1])
    assert (t_leads in holdout) == (t_led in holdout)
    leak = S.alias_leak_report(triples, train, holdout, {})
    assert leak["zero_alias_leaks"]


def test_v1_still_subject_grouped():
    triples = [
        _t("s1", "p", "o1"),
        _t("s1", "p", "o2"),
        _t("s2", "p", "o3"),
        _t("s3", "p", "o4"),
        _t("s4", "p", "o5"),
        _t("s5", "p", "o6"),
        _t("s6", "p", "o7"),
        _t("s7", "p", "o8"),
        _t("s8", "p", "o9"),
        _t("s9", "p", "o10"),
    ]
    train, holdout = S.make_split(triples, 0.2, 0)
    # s1's two triples travel together
    t0, t1 = S.triple_id(triples[0]), S.triple_id(triples[1])
    assert (t0 in holdout) == (t1 in holdout)


def test_nested_unit_slug_is_not_just_dirname():
    """RED that caught the EDC loader bug: ed_dir.name is the YEAR, the slug is city/year."""
    # Off-tree path still has a `data/` segment — must yield city/year, not year alone.
    assert S._domain_and_slug(Path("/x/data/edc/las_vegas/2021")) == ("edc", "las_vegas/2021")
    import config as C
    ed_dir = Path(C.ROOT) / "data" / "edc" / "las_vegas" / "2021"
    if not ed_dir.is_dir():
        return
    d, s = S._domain_and_slug(ed_dir, domain="edc")
    assert d == "edc"
    assert s == "las_vegas/2021", s
    assert s != ed_dir.name, "year-only slug is the bug under test"


def test_split_and_audit_loaders_agree_on_nested_unit():
    """The class that produced the 8 EDC alias overlaps: two loaders, two maps, one unit.

    GREEN only when both resolve the same non-empty map for a nested EDC unit that has a
    resolve artifact. Before the unification, split returned {} (keyed on year) and
    audit_recall returned the real map (keyed on city_year).
    """
    import config as C
    from pipeline import audit_recall as A

    ed_dir = Path(C.ROOT) / "data" / "edc" / "las_vegas" / "2021"
    resolve = Path(C.ROOT) / "data" / "_build" / "resolve" / "edc" / "las_vegas_2021" / "id_map.json"
    if not (ed_dir.is_dir() and resolve.is_file()):
        print("  skip  test_split_and_audit_loaders_agree_on_nested_unit (no EDC fixture on disk)")
        return
    m_split = S.load_id_map(ed_dir, domain="edc")
    m_audit = A.load_id_map("edc", "las_vegas/2021")
    assert m_split, "split loader must find the nested-unit resolve map (was {} under year-only key)"
    assert m_audit, "audit loader must find the nested-unit resolve map"
    assert m_split == m_audit, (
        f"loaders disagree on las_vegas/2021: split={len(m_split)} audit={len(m_audit)}"
    )
    # And the path both digests bind to is the same file.
    p_split = S.id_map_path(ed_dir, domain="edc")
    p_audit = A.id_map_path("edc", "las_vegas/2021")
    assert p_split is not None and p_audit is not None
    assert p_split.resolve() == p_audit.resolve()


def test_load_id_map_without_domain_still_finds_resolve_map():
    """domain=None used to skip the resolve/ tree entirely and only check 15_structured/."""
    import config as C

    ed_dir = Path(C.ROOT) / "data" / "pokemon-lore" / "kanto"
    resolve = Path(C.ROOT) / "data" / "_build" / "resolve" / "pokemon-lore" / "kanto" / "id_map.json"
    if not (ed_dir.is_dir() and resolve.is_file()):
        print("  skip  test_load_id_map_without_domain_still_finds_resolve_map")
        return
    m = S.load_id_map(ed_dir)  # no domain kwarg
    assert len(m) > 0, "domain inference from data/<domain>/<unit> must reach the resolve map"


# ---------------------------------------------------------------------------
# EDC SPLIT COVERAGE -- measurability, not file existence.
#
# A unit is COVERED iff it (a) resolves a split, (b) carries a NON-EMPTY holdout, and (c) that
# manifest PARTITIONS the unit's CURRENT triple id set -- covering it (nothing unassigned, no
# orphans) AND splitting it (train and holdout disjoint). A has-a-split-FILE predicate certifies
# units that contribute nothing to any install-strength estimand: six EDC units carry a manifest
# whose holdout is empty (generic_probes takes the `if sp:` branch and stamps them `held_out`
# with zero gold triples -- strictly worse than an honest `trained`), and one carries a manifest
# cut 25 minutes before its graph was rewritten. Both shapes are silent under file existence.
#
# Measured over the 98 (2026-07-31): 90 covered, 8 not. Every uncovered unit is DECLARED BY NAME
# below with its reason and its arithmetic, and every declaration is TIGHTNESS-asserted, so it can
# neither outlive its condition nor be padded with a unit that never exhibited it. Each condition
# is checked against its OWN declaration set: declaring a unit splitless/zero-holdout must never
# exempt it from the other checks, and the DISJOINTNESS check consults no declaration set at all.
#
# The pool is pinned by a FROZEN CONSTANT, because a count is a PROXY for the pool and membership
# is the thing. EDC_REGISTRY_COUNT is an EQUALITY, not a floor -- a floor is a ratchet nobody
# ratchets: grow to 110, shrink back to 98, and the shrinkage is invisible. But equality on a COUNT
# still lets one unit leave while another arrives, holding the size and silently dropping the
# departed unit from the estimand. An earlier revision answered that with a SECOND LIVE READER of
# the registry, and it was a measured NO-OP: `iter_editions(only=None, with_raw=False)` applies no
# filter and yields exactly one Ed per `_registry_units` tuple, so both readers were the same
# computation and an edit to configs/editions_edc.json moved both together -- the coverage
# assertion stayed GREEN with the departed unit named nowhere. Membership is therefore pinned by
# EDC_REGISTRY_SLUGS, a frozen 98-entry constant that a registry edit CANNOT move;
# EDC_REGISTRY_DIGEST pins that constant against a one-line edit inside a 98-entry list; and
# split_coverage_gaps takes expected_slugs and reports departures and arrivals BY NAME.
#
# Every reader here resolves the WINNING manifest -- the one `load_split` returns and every runtime
# consumer opens -- never a superseded version file. Split versions are immutable, so a repaired
# unit keeps its failing old manifest forever: grading superseded files makes the declared-debt set
# a function of files outside the estimand, and it deadlocked an earlier revision (the coverage
# assertion demanded an edit its sibling assertion then rejected, with no satisfying declaration).
#
# There is deliberately NO SKIP GUARD in these coverage assertions, and the pool checks run before
# any other check. A guard keyed on "some split exists" looks benign while one split survives and
# silently suppresses the three maximal gaming states (pool shrunk to the one exempt unit, registry
# emptied, every split purged) -- see test_maximal_gaming_states_are_not_suppressed, which ships
# those three states as permanent probes. A purged data tree is the DESIRED red, not a reason to
# stand down: configs/editions_edc.json still enumerates 98 after any purge, so a purged tree
# reports 97 UNDECLARED_SPLITLESS. THE PRECISE RULE, stated because this file DOES contain live
# data-root guards and a later reader will decide from this banner whether a new one is allowed:
# the three pre-existing r9 loader tests return early on a missing fixture
# (test_nested_unit_slug_is_not_just_dirname, test_split_and_audit_loaders_agree_on_nested_unit,
# test_load_id_map_without_domain_still_finds_resolve_map -- cited by NAME because inserting this
# block shifts every line number in the file, which is how the previous revision came to cite three
# stale ones), and the biconditional below skips when NO split artifact exists at all -- keyed on a
# condition independent of what it measures, and backed by the re-derivation vacuity assert.
# The coverage, re-derivation, digest, postcondition-agreement and registry-swap assertions carry
# no skip guard of any kind; the registry-swap probe's one hard `assert donor.is_dir()` is a
# PRECONDITION that fails loudly, never a skip.
#
# Nothing here re-cuts anything. Split versions are immutable and load_split prefers v3 > v2 > v1,
# so authoring a split CHANGES which holdout every downstream consumer reads -- an eval-series
# change, never a patch. The job of this block is to convert silent debt into gate-visible debt.
# ---------------------------------------------------------------------------

EDC_REGISTRY_COUNT = 98
# EXACT, not a floor. iter_editions('edc') and plan_install both enumerate 98 city/year units from
# configs/editions_edc.json (measured 2026-07-31). Growth AND shrinkage each require an edit here.

EDC_REGISTRY_SLUGS = frozenset({
    "chiba/2018", "chiba/2019", "chicago/2013", "dallas/2010", "dallas/2011", "denver/2008",
    "denver/2009", "denver/2010", "denver/2011", "edsea/2023", "edsea/2024", "edsea/2025",
    "edsea/2026", "incheon/2025", "incheon/2026", "las_vegas/2011", "las_vegas/2012",
    "las_vegas/2013", "las_vegas/2014", "las_vegas/2015", "las_vegas/2016", "las_vegas/2017",
    "las_vegas/2018", "las_vegas/2019", "las_vegas/2021", "las_vegas/2022", "las_vegas/2023",
    "las_vegas/2024", "las_vegas/2025", "las_vegas/2026", "london/2013", "los_angeles/1997",
    "los_angeles/1999", "los_angeles/2000", "los_angeles/2001", "los_angeles/2002",
    "los_angeles/2007", "los_angeles/2008", "los_angeles/2009", "los_angeles/2010",
    "medellin/2026", "mexico_city/2014", "mexico_city/2015", "mexico_city/2016",
    "mexico_city/2017", "mexico_city/2018", "mexico_city/2019", "mexico_city/2020",
    "mexico_city/2022", "mexico_city/2023", "mexico_city/2024", "mexico_city/2025",
    "mexico_city/2026", "milton_keynes/2014", "milton_keynes/2015", "milton_keynes/2016",
    "new_delhi/2016", "new_york_nj/2012", "new_york_nj/2013", "new_york_nj/2014",
    "new_york_nj/2015", "new_york_nj/2016", "orlando/2011", "orlando/2012", "orlando/2013",
    "orlando/2014", "orlando/2015", "orlando/2016", "orlando/2017", "orlando/2018",
    "orlando/2019", "orlando/2021", "orlando/2022", "orlando/2023", "orlando/2024",
    "orlando/2025", "orlando/2026", "phuket/2025", "phuket/2026", "portimao/2021",
    "portimao/2022", "san_bernardino/2003", "san_bernardino/2004", "san_bernardino/2005",
    "san_bernardino/2006", "san_juan/2009", "san_juan/2010", "san_juan/2011", "san_juan/2012",
    "san_juan/2013", "san_juan/2014", "san_juan/2015", "sao_paulo/2015", "seoul/2019",
    "shanghai/2018", "suzhou/2023", "suzhou/2025", "zhuhai/2018",
})
# The 98 slugs this campaign measured, FROZEN -- the pool a coverage claim is ABOUT. It may NOT
# be re-read from the registry at run time. Measured: `iter_editions(only=None, with_raw=False)`
# applies no filter and yields exactly one Ed per `_registry_units(C.load_editions('edc'))`
# tuple, so a live "second reader" is the SAME computation one call deeper -- edit
# configs/editions_edc.json and both sides move together, which is a swapped registry proving
# itself. A constant is the only thing a registry edit cannot move.
# GROWTH, SHRINKAGE AND SWAPS ARE ALL PERMITTED -- they are merely no longer SILENT. Each needs
# an edit to this set AND to the two constants below, and the failure message names the units
# that left and arrived so the edit is reviewable instead of arithmetic.

EDC_REGISTRY_DIGEST = "9a490fa577736290"
# TWO JOBS, both mechanical. (1) It pins EDC_REGISTRY_SLUGS against a one-line edit inside a
# 98-entry list -- a mutated slug changes the frozen set's own sha1 and the two constants stop
# agreeing. (2) It pins the LIVE registry, which is the reader that notices a registry move at all;
# the frozen set alone cannot, since nothing recomputes it. sha1("\n".join(sorted(ed.slug for ed in
# iter_editions("edc", with_raw=False)))).hexdigest()[:16]; measured 2026-07-31 at engine HEAD
# 9cd16822, n=98, identical for with_raw=True. The command is in the receipt's reproduce block.
# The two assertions are deliberately NOT derived from each other: computing the live digest from
# the frozen set would make the second one x == x.

EDC_DECLARED_SPLITLESS = {"las_vegas/2026"}
# 859 triples in 15_structured, no splits/ dir, no data/_build/resolve/edc/las_vegas_2026. Two
# independent blockers, either one disqualifying: (1) the resolve stage that a non-inert cut needs
# is declared `gpu: true` (configs/domains/edc.yaml:96) -- cutting without it is alias-inert and
# recreates the r9 defect; (2) the unit is `protected_editions` (edc.yaml:234), the hand-curated
# gold -- run_stage.py's PROTECTED_UNITS refusal rejects any stage writing under
# data/edc/las_vegas/2026/. Cutting it is a campaign with its own receipt, not a file drop.

EDC_DECLARED_ZERO_HOLDOUT = {
    "los_angeles/2000",       # len(triples)=2
    "los_angeles/2008",       # len(triples)=4
    "san_bernardino/2005",    # len(triples)=5
    "san_bernardino/2006",    # len(triples)=7
    "denver/2008",            # len(triples)=5
    "edsea/2026",             # len(triples)=2
}
# make_split_v2 takes `target = int(len(triples) * holdout_frac)`; at the declared frac 0.10 every
# n above yields target 0, so NO re-cut at the declared frac can ever give these units a holdout.
# The repair is a frac/floor POLICY decision or a permanent out-of-estimand declaration -- never a
# re-cut, because the arithmetic is not a bug in the cut.
#
# THIS DECLARATION IS A DELIBERATE DISAGREEMENT WITH THE STAGE CONTRACT, not an observation that
# nothing else cares. `run_stage._split_integrity_ok` rejects an empty holdout outright (the
# `not holdout` clause; def at run_stage.py:524, the clause at :540 as of engine HEAD 9cd16822),
# execution_contract.py:293 wires it as the split stage's postcondition kind "split_integrity",
# and test_typed_postconditions.py:125 already red-green proves it with an ("empty holdout",
# holdout_ids=[]) mutation. build_contract('edc') emits one split_integrity postcondition per unit
# for the split stage, and that checker rejects exactly these six of the 97 WINNING manifests --
# the file `load_split` resolves for each unit. Denominators, because an earlier revision stated
# this one wrong and the wrong number is what hid a version-scope mismatch: 98 registry units, 97
# of them split-carrying, 147 version FILES on disk (97 v2 + 50 v3), 97 WINNING manifests.
# So the split stage is UN-PASSABLE for EDC until a frac/floor policy lands -- these six are not
# cost-free debt. test_declared_zero_holdout_matches_the_engines_own_split_postcondition pins the
# two readers together so the disagreement cannot silently drift.
#
# READ BEFORE EDITING THIS SET: `_split_integrity_ok` has SEVEN `return False` sites (measured over
# the body at run_stage.py:524) -- not a dict; train/holdout missing or holdout empty; train and
# holdout overlapping; a missing decontamination block on v2/v3; a non-zero alias-leak count; a
# coverage shortfall against a recorded n_unique_triple_ids; and a realized frac outside the
# 0.5x-2.0x band. A new name in its verdict is therefore NOT necessarily a seventh empty holdout.
# The set is named ZERO_HOLDOUT but bound to a seven-way predicate; narrowing that binding to a
# witness (`not manifest["holdout_ids"]`) is a recorded follow-up, not this block's job.

EDC_DECLARED_STALE_SPLIT = {"milton_keynes/2015"}

# Units whose WINNING manifest the split stage's own postcondition rejects for a reason OTHER than
# an empty holdout (frac band, missing decontamination, alias leaks, coverage shortfall, overlap,
# not-a-dict). Empty at HEAD: today all six postcondition rejections ARE empty holdouts. This set
# exists so a non-holdout rejection has a NAMED, DECLARABLE home (UNDECLARED_POSTCOND_REJECT in the
# agreement test below) instead of deadlocking EDC_DECLARED_ZERO_HOLDOUT: a floored v3 whose
# realized frac lands outside the 0.5x-2.0x band gives a unit a NON-empty holdout the tightness
# guard rightly retires, while the whole-verdict comparison kept demanding the unit stay declared.
# Measured on 4 of the 6 declared units (los_angeles/2000+2008, san_bernardino/2005, edsea/2026):
# their floored v3 fixtures realize fracs 0.25-0.50 against the [0.05, 0.20] band, so under the
# old whole-verdict binding NO declaration state was green — the deadlock this set ends.
EDC_DECLARED_POSTCOND_DEBT: set[str] = set()
# splits/split_v3.json written 2026-07-25 15:13; 15_structured/triples.jsonl rewritten 15:38 -- the
# graph moved 25 minutes AFTER the cut. 39 of 82 current triple ids are in NEITHER side, 0 orphans.
# Its resolve map is live (2 entries), so a re-cut is a NEW eval series, not a patch; this
# declaration is the placeholder until that campaign runs. Nothing else catches the CURRENT-GRAPH
# half: run_stage._split_integrity_ok's population-coverage sub-check reads
# `manifest.get("n_unique_triple_ids")` and skips when it is absent (run_stage.py:554 at HEAD
# 9cd16822) -- 0 of the 97 on-disk EDC manifests carry that key -- and it grades against the
# manifest's own frozen denominator rather than the current graph even when it does fire. The
# DISJOINTNESS half is a different story: that checker does own it, and so does this block, which
# is why the leak check below consults no declaration set.

EDC_SPLIT_SEED = 42
EDC_SPLIT_HOLDOUT_FRAC = 0.10
# The cut parameters every EDC manifest records. split.py's argparse defaults live INSIDE main()
# (split.py:366-367) and are not importable, so these literals are pinned deliberately. The
# reproduction claim does NOT rest on them: re-derivation below feeds each manifest its OWN
# recorded seed/holdout_frac. These two only catch a manifest that drifted off the declared cut.


def _leaked_holdout_gap(slug, ver, n_overlap):
    """THE ONE OWNER of the LEAKED_HOLDOUT text. Two readers emit this class -- the pure verdict
    function below and the re-derivation test -- and an earlier revision authored the sentence
    twice, independently, with nothing comparing them: the two copies rendered identically only by
    coincidence of authorship, so editing one was silent. One owner, and both readers are asserted
    to hold no private copy of the literal (see the RC-8 block in the failure-shape test).
    """
    return (f"LEAKED_HOLDOUT {slug} ({ver}): {n_overlap} ids are in BOTH train and holdout - the"
            " manifest does not PARTITION the graph and every such 'held-out' fact was trained on")


def split_coverage_gaps(rows, splitless, zero_holdout, stale, expected_pool, expected_slugs):
    """Every way an EDC unit falls out of the install-strength estimand, as a list of strings.

    rows: {slug: {"has_split", "ver", "holdout", "overlap", "unassigned", "orphan"}}. PURE -- no
    disk, no imports, no early return of any kind. The two pool checks are FIRST and unconditional
    precisely so that moving the denominator -- by shrinking it OR by swapping members at constant
    size -- can never suppress the per-unit checks underneath it.

    expected_slugs is REQUIRED, with no default: a default would let a caller silently fall back
    to "whatever arrived is what I meant to measure", which is the defect this argument exists to
    close.
    """
    gaps = []
    if len(rows) != expected_pool:
        gaps.append(
            f"POOL_MISMATCH: {len(rows)} units enumerated, EDC_REGISTRY_COUNT is {expected_pool}"
            " - a coverage claim may not move its own denominator; growth and shrinkage both"
            " require an explicit constant edit"
        )
    left = sorted(set(expected_slugs) - set(rows))
    arrived = sorted(set(rows) - set(expected_slugs))
    if left or arrived:
        gaps.append(
            f"POOL_MEMBERSHIP_DRIFT: left={left} arrived={arrived} - a coverage claim may not"
            " move its own denominator; a swap holds the COUNT and still drops units from the"
            " estimand"
        )
    for set_name, declared in (
        ("EDC_DECLARED_SPLITLESS", splitless),
        ("EDC_DECLARED_ZERO_HOLDOUT", zero_holdout),
        ("EDC_DECLARED_STALE_SPLIT", stale),
    ):
        for slug in sorted(declared):
            if slug not in rows:
                gaps.append(
                    f"DECLARED_UNIT_ABSENT {slug}: declared in {set_name} but the registry no"
                    " longer enumerates it - a gap may not be closed by deleting the unit"
                )
    # Per-unit. Each condition consults its OWN set, so a declaration in one set never buys
    # exemption from another (padding EDC_DECLARED_ZERO_HOLDOUT cannot silence a stale manifest),
    # and LEAKED_HOLDOUT consults NO set at all -- a manifest whose holdout is also trained on is
    # not declarable debt, it is a split that measures nothing while reporting a number.
    #
    # LEAKED_HOLDOUT is NOT the first owner of that predicate and does not claim to be. Two others
    # exist: run_stage._split_integrity_ok (`if set(train) & set(holdout): return False`), which
    # runs only when the split stage EXECUTES, and scripts/eval_probes.py:138, which raises only at
    # eval time, only over units carrying probes, and is not declared in edc.yaml at all. Neither
    # enumerates the registry and neither runs at commit time. This one does, over all 98.
    for slug in sorted(rows):
        row = rows[slug]
        if not row["has_split"]:
            if slug not in splitless:
                gaps.append(
                    f"UNDECLARED_SPLITLESS {slug}: load_split resolves to None and the slug is"
                    " not in EDC_DECLARED_SPLITLESS"
                )
            continue
        if row["overlap"]:
            gaps.append(_leaked_holdout_gap(slug, row["ver"], row["overlap"]))
        if row["holdout"] == 0 and slug not in zero_holdout:
            gaps.append(
                f"UNDECLARED_ZERO_HOLDOUT {slug} ({row['ver']}): carries a split manifest with an"
                " EMPTY holdout - a file-existence check calls it covered while it contributes"
                " zero to any install-strength estimand"
            )
        if (row["unassigned"] or row["orphan"]) and slug not in stale:
            gaps.append(
                f"UNDECLARED_STALE_SPLIT {slug} ({row['ver']}): {row['unassigned']} current triple"
                f" ids in NEITHER side, {row['orphan']} manifest ids no longer in the graph"
            )
    # TIGHTNESS. A declaration may not outlive its condition, and a set may not be padded with a
    # unit that does not exhibit it. Both directions are one line of bookkeeping to fix, and the
    # message names the exact edit.
    for slug in sorted(splitless):
        row = rows.get(slug)
        if row is not None and row["has_split"]:
            gaps.append(
                f"STALE_EXEMPTION {slug}: now carries a split - remove it from"
                " EDC_DECLARED_SPLITLESS"
            )
    for slug in sorted(zero_holdout):
        row = rows.get(slug)
        # `not row["overlap"]` for exactly the reason the stale predicate carries it: a holdout
        # whose every id is also a train id is not a non-empty holdout in any sense that retires
        # the declaration. The guard protects the MESSAGE, not the gap -- the leak is already
        # reported above and is undeclarable, so nothing is suppressed by adding it. The two
        # tightness predicates were asymmetric here and this block emitted "now has a non-empty
        # holdout - remove it from EDC_DECLARED_ZERO_HOLDOUT" about a manifest every held-out fact
        # of which was trained on: wrong advice, on the exact class the sibling guard exists for.
        if (row is not None and row["has_split"] and row["holdout"] > 0
                and not row["overlap"]):
            gaps.append(
                f"STALE_EXEMPTION {slug}: now has a non-empty holdout - remove it from"
                " EDC_DECLARED_ZERO_HOLDOUT"
            )
    for slug in sorted(stale):
        row = rows.get(slug)
        # "Partitions" here means the same thing it means everywhere else in this block: covers
        # the graph AND splits it. Without the overlap term this message tells an operator to
        # drop the declaration on a manifest whose holdout is entirely trained on -- it would
        # read "now partitions its graph" about a unit that does not partition anything.
        if (row is not None and row["has_split"] and not row["overlap"]
                and not row["unassigned"] and not row["orphan"]):
            gaps.append(
                f"STALE_EXEMPTION {slug}: now partitions its graph - remove it from"
                " EDC_DECLARED_STALE_SPLIT"
            )
    return gaps


def _winning_split_path(ed_dir):
    """THE ONE OWNER of "which manifest is this unit's split", for every reader in this block.

    `load_split` resolves v3 > v2 > v1 and returns the manifest; a reader that needs the PATH (to
    hand to a file-taking checker) must resolve the same way or it is grading a different file set.
    Enumerating ("v2", "v3") instead opens 147 version files where the estimand holds 97 winning
    manifests, and because split versions are IMMUTABLE the superseded file never improves: a unit
    repaired by a v3 keeps its failing v2 forever. The two readers are pinned to agree by
    test_declared_zero_holdout_matches_the_engines_own_split_postcondition.
    """
    for ver in S.SPLIT_VERSION_PREFERENCE:
        p = S.split_path(ed_dir, ver)
        if p.exists():
            return p, ver
    return None, None


def _split_row(manifest, current_ids):
    """One coverage row from one manifest and the unit's CURRENT triple ids. PURE.

    Extracted so the row FIELDS have a red-green proof of their own. Three of the four -- overlap,
    unassigned, orphan -- are 0 for every unit on the live tree, so a tree-reading assertion can
    never show they are derived from the manifest at all: force any of them to a constant and the
    whole suite stays green. Here a synthetic manifest proves each one moves. See the row-builder
    block in test_split_coverage_gaps_detects_each_failure_shape.
    """
    if manifest is None:
        return {"has_split": False, "ver": None, "holdout": 0, "overlap": 0,
                "unassigned": 0, "orphan": 0}
    train, holdout = set(manifest["train_ids"]), set(manifest["holdout_ids"])
    return {
        "has_split": True,
        "ver": manifest.get("version"),
        "holdout": len(manifest["holdout_ids"]),
        # A partition has to SPLIT as well as cover. `train | holdout` is a UNION and hides an
        # overlap perfectly: a manifest whose every holdout id is also a train id reads
        # unassigned=0, orphan=0 and looks like the healthiest unit in the corpus.
        "overlap": len(train & holdout),
        "unassigned": len(current_ids - (train | holdout)),
        "orphan": len((train | holdout) - current_ids),
    }


def _edc_coverage_rows():
    """{slug: row} over EVERY registry unit, graded against the CURRENT graph.

    Deliberately uncached and re-measured per call: a shared cache would make one test's
    monkeypatch invisible to the next and quietly turn a probe green. Graded against the CURRENT
    triple ids -- never the manifest's own frozen n_triples and never n_unique_triple_ids, which
    0 of the 97 on-disk EDC manifests carry.
    """
    from pipeline.editions import iter_editions
    from pipeline.generic_expand import load_all_triples

    rows = {}
    for ed in iter_editions("edc", with_raw=False):
        manifest = S.load_split(ed.dir)
        current = ({S.triple_id(t) for t in load_all_triples(ed)}
                   if manifest is not None else set())
        rows[ed.slug] = _split_row(manifest, current)
    return rows


def test_split_coverage_gaps_detects_each_failure_shape():
    """Red-green for the verdict function itself. No tree read.

    Every shape is proven reachable ALONE -- a gap that only ever fires alongside another one is
    not proven at all -- plus the cross-set case that makes the declaration sets unpaddable.
    """
    ok = {"has_split": True, "ver": "v2", "holdout": 4, "overlap": 0,
          "unassigned": 0, "orphan": 0}

    def unit(**kw):
        row = dict(ok)
        row.update(kw)
        return row

    # THE ROW BUILDER, PURE. `split_coverage_gaps` is only as honest as the rows handed to it, and
    # overlap / unassigned / orphan are ALL 0 for every unit on the live tree -- so no tree-reading
    # assertion can show any of the three is derived from the manifest at all. Force one to a
    # constant and every other test in this file stays green. These cases prove each field moves,
    # and they are the only place `orphan` is exercised by anything in this repo.
    def manifest(train, holdout, ver="v2"):
        return {"version": ver, "train_ids": list(train), "holdout_ids": list(holdout)}

    assert _split_row(None, set()) == {"has_split": False, "ver": None, "holdout": 0,
                                       "overlap": 0, "unassigned": 0, "orphan": 0}
    assert _split_row(manifest("abc", "d", ver="v3"), set("abcd")) == {
        "has_split": True, "ver": "v3", "holdout": 1, "overlap": 0,
        "unassigned": 0, "orphan": 0}, "the healthy shape, and `ver` is read from the manifest"
    assert _split_row(manifest("abcd", "cd"), set("abcd"))["overlap"] == 2, (
        "overlap must come from train & holdout - a manifest whose holdout is also trained on "
        "reads unassigned=0 and orphan=0 and is otherwise indistinguishable from the healthiest "
        "unit in the corpus")
    assert _split_row(manifest("ab", "c"), set("abcde"))["unassigned"] == 2, (
        "unassigned must come from the CURRENT graph minus the manifest, never from the "
        "manifest's own frozen n_triples")
    assert _split_row(manifest("abc", "dz"), set("abcd"))["orphan"] == 1, (
        "orphan must come from the manifest minus the CURRENT graph")

    # COMPLIANT: two covered units plus one declared splitless. No gaps.
    assert split_coverage_gaps(
        {"a/1": unit(), "a/2": unit(),
         "b/1": unit(has_split=False, ver=None, holdout=0)},
        {"b/1"}, set(), set(), 3, {"a/1", "a/2", "b/1"}) == []

    def only(rows, splitless=(), zero=(), stale=(), pool=None, expected=None):
        return split_coverage_gaps(rows, set(splitless), set(zero), set(stale),
                                   len(rows) if pool is None else pool,
                                   set(rows) if expected is None else set(expected))

    g = only({"a/1": unit(has_split=False, ver=None, holdout=0)})
    assert len(g) == 1 and g[0].startswith("UNDECLARED_SPLITLESS a/1"), g

    g = only({"a/1": unit(holdout=0)})
    assert len(g) == 1 and g[0].startswith("UNDECLARED_ZERO_HOLDOUT a/1 (v2)"), g

    g = only({"a/1": unit(unassigned=39)})
    assert g == ["UNDECLARED_STALE_SPLIT a/1 (v2): 39 current triple ids in NEITHER side,"
                 " 0 manifest ids no longer in the graph"], g

    # ORPHAN alone -- a manifest id that is no longer in the graph, with nothing unassigned.
    g = only({"a/1": unit(orphan=7)})
    assert g == ["UNDECLARED_STALE_SPLIT a/1 (v2): 0 current triple ids in NEITHER side,"
                 " 7 manifest ids no longer in the graph"], g

    g = only({"a/1": unit()}, pool=2)
    assert len(g) == 1 and g[0].startswith("POOL_MISMATCH: 1 units enumerated,"
                                           " EDC_REGISTRY_COUNT is 2"), g
    g = only({"a/1": unit(), "a/2": unit(), "a/3": unit()}, pool=2)
    assert len(g) == 1 and g[0].startswith("POOL_MISMATCH: 3 units enumerated,"
                                           " EDC_REGISTRY_COUNT is 2"), g

    # MEMBERSHIP DRIFT AT CONSTANT SIZE -- the counterexample the count check cannot see. 98 units
    # in, 98 units out, one swapped. POOL_MISMATCH stays silent and this must not.
    base = {f"c{i}/2020": unit() for i in range(98)}
    swapped = {k: v for k, v in base.items() if k != "c0/2020"}
    swapped["newcity/2027"] = unit()
    assert len(swapped) == EDC_REGISTRY_COUNT == len(base), (len(swapped), len(base))
    g = split_coverage_gaps(swapped, set(), set(), set(), 98, frozenset(base))
    assert len(g) == 1 and g[0] == (
        "POOL_MEMBERSHIP_DRIFT: left=['c0/2020'] arrived=['newcity/2027'] - a coverage claim may"
        " not move its own denominator; a swap holds the COUNT and still drops units from the"
        " estimand"), g
    # ...and the un-swapped pool is not a gap, so the check keys on membership, not on size.
    assert split_coverage_gaps(base, set(), set(), set(), 98, frozenset(base)) == []

    # LEAKED_HOLDOUT -- a manifest that covers the graph and does not PARTITION it. `assigned` is
    # a union, so unassigned/orphan are both 0 and every other check reads this unit as healthy.
    g = only({"a/1": unit(overlap=4)})
    assert g == [_leaked_holdout_gap("a/1", "v2", 4)], g
    # Declaring the SAME row stale must not buy exemption from the leak: no declaration set does.
    g = only({"a/1": unit(overlap=4, unassigned=39)}, stale=("a/1",))
    assert len(g) == 1 and g[0].startswith("LEAKED_HOLDOUT a/1 (v2): 4 ids"), g
    # ...nor may a leak be laundered into a TIGHTNESS gap. Declared stale, graph fully covered,
    # holdout entirely trained on: the only honest verdict is LEAKED_HOLDOUT, never
    # "now partitions its graph - remove it from EDC_DECLARED_STALE_SPLIT".
    g = only({"a/1": unit(overlap=4, holdout=0)}, zero=("a/1",), stale=("a/1",))
    assert len(g) == 1 and g[0].startswith("LEAKED_HOLDOUT a/1 (v2): 4 ids"), g

    # RC-3: THE TWO TIGHTNESS PREDICATES MUST BE SYMMETRIC under a 100%-leaked holdout. A row with
    # holdout=4 and overlap=4 covers its graph, splits nothing, and is declared in one set or the
    # other. Either way the ONLY honest verdict is the leak; "now has a non-empty holdout - remove
    # it from EDC_DECLARED_ZERO_HOLDOUT" is wrong advice about a manifest that partitions nothing,
    # and it is the exact sentence the sibling STALE predicate already guards against.
    leaked = unit(holdout=4, overlap=4)
    for set_name, kw in (("EDC_DECLARED_ZERO_HOLDOUT", {"zero": ("a/1",)}),
                         ("EDC_DECLARED_STALE_SPLIT", {"stale": ("a/1",)})):
        g = only({"a/1": dict(leaked)}, **kw)
        assert g == [_leaked_holdout_gap("a/1", "v2", 4)], (set_name, g)
        assert not any("now has a non-empty holdout" in x for x in g), (set_name, g)
        assert not any("now partitions its graph" in x for x in g), (set_name, g)

    # (The two both-declared rows above and below are NOT in tension with the pairwise-disjointness
    # assertion on the real declaration sets. This function is pure and must stay indifferent to
    # who declared what; the disjointness constraint is on the three module CONSTANTS, where a unit
    # in two sets hides which condition is actually true. Enforcing it here would delete the
    # cross-set proofs that keep the sets unpaddable.)

    # RC-8: the LEAKED_HOLDOUT sentence has EXACTLY ONE OWNER. Two readers emit it -- this verdict
    # function and test_edc_splits_reproduce_under_their_recorded_parameters -- and they used to
    # hold independently authored f-strings that rendered identically only by coincidence, with
    # nothing comparing them: a one-sided edit was silent. Pinned two ways. (a) the owner renders
    # the canonical text; (b) NEITHER reader holds a string literal naming the class, so a
    # re-inlined copy is a RED rather than a second source of truth. Comments are not Constants,
    # so prose about the class is unaffected.
    assert _leaked_holdout_gap("a/1", "v2", 4) == (
        "LEAKED_HOLDOUT a/1 (v2): 4 ids are in BOTH train and holdout - the manifest does not"
        " PARTITION the graph and every such 'held-out' fact was trained on"), (
        "the LEAKED_HOLDOUT sentence moved. It is pinned here CHARACTER FOR CHARACTER on purpose: "
        "every other assertion in this file compares the emission against _leaked_holdout_gap "
        "itself, which is x == x and would follow the owner anywhere it went")
    import ast
    tree = ast.parse(Path(__file__).read_text())
    for fn in ("split_coverage_gaps", "test_edc_splits_reproduce_under_their_recorded_parameters"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        private = [c.value for c in ast.walk(node)
                   if isinstance(c, ast.Constant) and isinstance(c.value, str)
                   and "LEAKED_HOLDOUT" in c.value]
        assert not private, (
            f"{fn} authors its own copy of the LEAKED_HOLDOUT message: {private} - both readers "
            "must call _leaked_holdout_gap, or the two sentences drift apart the moment one is "
            "edited (which is how this defect shipped the first time)")

    g = only({"a/1": unit()}, splitless=("gone/1",))
    assert len(g) == 1 and g[0].startswith("DECLARED_UNIT_ABSENT gone/1"), g

    # TIGHTNESS, one per set.
    g = only({"a/1": unit()}, splitless=("a/1",))
    assert g == ["STALE_EXEMPTION a/1: now carries a split - remove it from"
                 " EDC_DECLARED_SPLITLESS"], g
    g = only({"a/1": unit()}, zero=("a/1",))
    assert g == ["STALE_EXEMPTION a/1: now has a non-empty holdout - remove it from"
                 " EDC_DECLARED_ZERO_HOLDOUT"], g
    g = only({"a/1": unit()}, stale=("a/1",))
    assert g == ["STALE_EXEMPTION a/1: now partitions its graph - remove it from"
                 " EDC_DECLARED_STALE_SPLIT"], g

    # CROSS-SET: a unit declared zero-holdout that ALSO stops partitioning its graph still goes
    # red on staleness. Declaring a unit in one set buys exemption from that set only.
    g = only({"a/1": unit(holdout=0, unassigned=39)}, zero=("a/1",))
    assert len(g) == 1 and g[0].startswith("UNDECLARED_STALE_SPLIT a/1"), g
    # ...and symmetrically.
    g = only({"a/1": unit(holdout=0, unassigned=39)}, stale=("a/1",))
    assert len(g) == 1 and g[0].startswith("UNDECLARED_ZERO_HOLDOUT a/1"), g
    # Declared in both, exhibiting both: covered debt, no gap.
    assert only({"a/1": unit(holdout=0, unassigned=39)}, zero=("a/1",), stale=("a/1",)) == []


def test_the_live_coverage_reader_routes_through_the_proven_row_builder():
    """`_split_row` is red-green proven; NOTHING proved the LIVE reader uses it.

    overlap and orphan are 0 for every EDC unit on disk, so re-inlining the row dict
    inside `_edc_coverage_rows` with either field pinned to a constant leaves every
    other assertion in this file GREEN -- the exact state the extraction was made to
    end. The proof of the builder is a proof of the corpus only if the corpus is read
    through it, so bind the caller to the callee.
    """
    import sys as _sys
    me = _sys.modules[__name__]
    marker = {"has_split": True, "ver": "SENTINEL", "holdout": -1, "overlap": -1,
              "unassigned": -1, "orphan": -1}
    real = me._split_row
    me._split_row = lambda manifest, current_ids: dict(marker)
    try:
        rows = me._edc_coverage_rows()
    finally:
        me._split_row = real
    assert rows, "VACUOUS: _edc_coverage_rows returned no rows to attribute"
    off = sorted(s for s, r in rows.items() if r != marker)
    assert not off, (
        "_edc_coverage_rows does not build its rows with _split_row: "
        f"{len(off)} of {len(rows)} rows survived the substitution ({off[:3]}...). "
        "overlap and orphan are 0 for every unit on the live tree, so an inlined row "
        "dict here can pin either one to a constant and every tree-reading assertion "
        "in this file stays GREEN")


def test_maximal_gaming_states_are_not_suppressed():
    """The three states a `if not any(has_split): return` guard silently swallowed.

    Shipped as permanent probes so the guard cannot come back, in any spelling. Each state is a
    way to make a coverage claim true by destroying what it measures, and each must be LOUD. The
    boundary case (exactly one split alive) is kept alongside: one-alive/zero-alive is the pair
    that pins where such a guard stops working, and the guard passed the one-alive probe.
    """
    live = {"has_split": True, "ver": "v2", "holdout": 4, "overlap": 0,
            "unassigned": 0, "orphan": 0}
    dead = {"has_split": False, "ver": None, "holdout": 0, "overlap": 0,
            "unassigned": 0, "orphan": 0}
    pool = 98
    slugs = [f"c{i}/2020" for i in range(97)] + ["exempt/2026"]
    expected = frozenset(slugs)
    declared = {"exempt/2026"}

    # BASELINE: full pool, the one declared unit splitless, everything else covered. GREEN.
    rows = {s: dict(live) for s in slugs}
    rows["exempt/2026"] = dict(dead)
    assert split_coverage_gaps(rows, declared, set(), set(), pool, expected) == []

    # (1) POOL SHRUNK TO THE ONE EXEMPT UNIT -- 97 units removed.
    g = split_coverage_gaps({"exempt/2026": dict(dead)}, declared, set(), set(), pool, expected)
    assert any(x.startswith("POOL_MISMATCH: 1 units enumerated,"
                            " EDC_REGISTRY_COUNT is 98") for x in g), g
    assert any(x.startswith("POOL_MEMBERSHIP_DRIFT: left=['c0/2020',") for x in g), g

    # (2) REGISTRY EMPTIED.
    g = split_coverage_gaps({}, declared, set(), set(), pool, expected)
    assert any(x.startswith("POOL_MISMATCH: 0 units enumerated,"
                            " EDC_REGISTRY_COUNT is 98") for x in g), g
    assert any(x.startswith("POOL_MEMBERSHIP_DRIFT: left=['c0/2020',") for x in g), g
    assert any(x.startswith("DECLARED_UNIT_ABSENT exempt/2026") for x in g), g

    # (3) EVERY SPLIT PURGED, registry intact.
    g = split_coverage_gaps({s: dict(dead) for s in slugs}, declared, set(), set(), pool, expected)
    undeclared = [x for x in g if x.startswith("UNDECLARED_SPLITLESS")]
    assert len(undeclared) == 97, len(undeclared)
    assert not any("exempt/2026" in x for x in undeclared), g

    # (4) BOUNDARY: exactly one split alive. The old guard passed this one, which is why the
    # three above were never reached.
    rows = {s: dict(dead) for s in slugs}
    rows["c0/2020"] = dict(live)
    g = split_coverage_gaps(rows, declared, set(), set(), pool, expected)
    assert len([x for x in g if x.startswith("UNDECLARED_SPLITLESS")]) == 96, g

    # (5) THE SWAP: 97 real units silently replaced one at a time would each be a gap, but the
    # cheap version replaces one and keeps the size. Both pool checks are computed BEFORE any
    # per-unit check precisely so this cannot be suppressed by what the incoming unit looks like.
    rows = {s: dict(live) for s in slugs}
    rows["exempt/2026"] = dict(dead)
    del rows["c0/2020"]
    rows["newcity/2027"] = dict(live)
    g = split_coverage_gaps(rows, declared, set(), set(), pool, expected)
    assert g == ["POOL_MEMBERSHIP_DRIFT: left=['c0/2020'] arrived=['newcity/2027'] - a coverage"
                 " claim may not move its own denominator; a swap holds the COUNT and still drops"
                 " units from the estimand"], g


def test_edc_registry_membership_is_pinned_by_digest():
    """The pool is pinned by MEMBERSHIP, not just cardinality. Reads the live registry.

    Three assertions, three different failures. The first two pin the FROZEN constants to each
    other, so a one-line edit inside a 98-entry frozenset cannot pass as arithmetic. The third
    reads the LIVE registry: it is the only reader here that notices a registry MOVE at all, since
    nothing recomputes the frozen set. Deliberately not derived from each other -- computing the
    live digest from EDC_REGISTRY_SLUGS would make the third assertion x == x, which is the exact
    circularity that made an earlier "second live reader" a no-op.

    The digest names nothing. It fires on any move and reports two hex strings; WHICH unit left is
    reported by POOL_MEMBERSHIP_DRIFT, proven by
    test_a_constant_size_registry_swap_names_the_departed_unit. Growth, shrinkage and swaps are all
    permitted -- reviewable constant edits, not walls -- they are merely no longer SILENT.
    """
    import hashlib
    from pipeline.editions import iter_editions

    slugs = sorted(ed.slug for ed in iter_editions("edc", with_raw=False))
    digest = hashlib.sha1("\n".join(slugs).encode("utf-8")).hexdigest()[:16]
    assert len(slugs) == EDC_REGISTRY_COUNT, (
        f"the registry enumerates {len(slugs)} units, EDC_REGISTRY_COUNT is {EDC_REGISTRY_COUNT}"
        f" - edit BOTH EDC_REGISTRY_COUNT and EDC_REGISTRY_DIGEST (now {digest}) and record which"
        " units left and arrived")
    frozen = hashlib.sha1("\n".join(sorted(EDC_REGISTRY_SLUGS)).encode("utf-8")).hexdigest()[:16]
    assert frozen == EDC_REGISTRY_DIGEST, (
        f"EDC_REGISTRY_SLUGS and EDC_REGISTRY_DIGEST have drifted apart: the frozen slug set hashes"
        f" to {frozen}, EDC_REGISTRY_DIGEST is {EDC_REGISTRY_DIGEST} - the two constants pin each"
        " other precisely so a one-line edit inside a 98-entry frozenset cannot pass as arithmetic")
    assert len(EDC_REGISTRY_SLUGS) == EDC_REGISTRY_COUNT, (
        f"EDC_REGISTRY_SLUGS holds {len(EDC_REGISTRY_SLUGS)} slugs, EDC_REGISTRY_COUNT is"
        f" {EDC_REGISTRY_COUNT} - the frozen set IS the pool, so the count may not be edited"
        " without it")
    assert digest == EDC_REGISTRY_DIGEST, (
        f"registry MEMBERSHIP moved while the count held at {len(slugs)}: sha1 of the sorted slugs"
        f" is {digest}, EDC_REGISTRY_DIGEST is {EDC_REGISTRY_DIGEST} - edit BOTH"
        " EDC_REGISTRY_DIGEST and EDC_REGISTRY_COUNT, and name the units that left and arrived")


def _swap_one_registry_unit(depart, arrive):
    """A registry EDIT, applied the way configs/editions_edc.json applies one: through
    `config.load_editions`, which is the single call both `_registry_units` and `iter_editions`
    consume. Patching `iter_editions` instead would prove nothing -- a real edit moves every live
    reader together, which is precisely why the expected slug set is a frozen constant.

    Returns a restore callable; the caller must call it in a finally.
    """
    import copy

    import config as C

    real = C.load_editions
    d_city, d_year = depart.split("/")
    a_city, a_year = arrive.split("/")

    def edited(festival="edc"):
        reg = copy.deepcopy(real(festival))
        for entry in reg["editions"]:
            if entry.get("city") == d_city:
                entry["years"] = [y for y in entry["years"] if str(int(y)) != d_year]
        reg["editions"].append({"city": a_city, "years": [int(a_year)]})
        return reg

    C.load_editions = edited
    return lambda: setattr(C, "load_editions", real)


def test_a_constant_size_registry_swap_names_the_departed_unit():
    """RC-1. One unit leaves, one arrives, the count holds at 98 -- the coverage claim must NAME
    the unit that left, in the assertion an operator actually reads.

    This is the case that failed on the revision this block replaces. That revision computed the
    expected slug set from a live "second reader" one call below `iter_editions`; a registry edit
    moved both readers together, the symmetric difference was empty, and the coverage assertion
    stayed GREEN with orlando/2014 silently out of the estimand. Only the digest fired, and the
    digest reports two hex strings and names nothing. Shipped here as a PERMANENT probe because the
    failure mode is a green test, not a red one, and nothing else in this file would notice.

    Everything happens in a throwaway DATA root of symlinks; the engine tree is never written.
    """
    import os
    import shutil
    import tempfile

    import config as C

    depart, arrive = "orlando/2014", "newcity/2027"
    donor = C.DATA / "edc" / "orlando" / "2014"
    assert donor.is_dir(), (
        f"the RC-1 fixture needs {donor} on disk to give the ARRIVING unit real, genuinely covered "
        "artifacts - without it the swap would go red for a missing-data reason and prove nothing")

    real_data, restore_registry = C.DATA, None
    tmp = tempfile.mkdtemp(prefix="edc_registry_swap_")
    try:
        root = Path(tmp) / "data"
        root.mkdir()
        for entry in real_data.iterdir():          # everything but data/edc, by symlink
            if entry.name != "edc":
                os.symlink(entry, root / entry.name)
        edc = root / "edc"
        edc.mkdir()
        for city in (real_data / "edc").iterdir():
            os.symlink(city, edc / city.name)
        a_city, a_year = arrive.split("/")
        assert not (edc / a_city).exists(), (
            f"the registry now enumerates a real {a_city} - pick another synthetic arrival for "
            "this probe; the fixture must not shadow a live unit")
        newcity = edc / a_city / a_year
        newcity.mkdir(parents=True)
        # THE INTERLOCK. Every unit dir above is a SYMLINK into the live tree, so a path bug here
        # would land a mkdir/symlink inside data/edc. resolve() follows the links, so this fires
        # BEFORE anything is created under a real unit. (shutil.rmtree below never follows a
        # symlink into a directory either -- it unlinks it -- so the live tree survives cleanup.)
        assert not newcity.resolve().is_relative_to(real_data), newcity
        for sub in donor.iterdir():                # the arrival is COVERED, not a hollow shell
            os.symlink(sub, newcity / sub.name)

        C.DATA = root
        restore_registry = _swap_one_registry_unit(depart, arrive)
        rows = _edc_coverage_rows()
        assert len(rows) == EDC_REGISTRY_COUNT and depart not in rows and arrive in rows, (
            f"the RC-1 fixture did not perform a CONSTANT-SIZE swap: {len(rows)} rows, "
            f"{depart} present={depart in rows}, {arrive} present={arrive in rows}")
        try:
            test_edc_split_coverage_has_no_undeclared_gap()
        except AssertionError as exc:
            coverage = str(exc)
        else:
            raise AssertionError(
                f"POOL_MEMBERSHIP_DRIFT IS A NO-OP: {depart} left the registry and the coverage "
                f"assertion stayed GREEN. expected_slugs must be a FROZEN constant; any live "
                "re-read of the registry moves with the edit and proves itself")
        try:
            test_edc_registry_membership_is_pinned_by_digest()
        except AssertionError as exc:
            digest = str(exc)
        else:
            raise AssertionError("the live-registry digest assertion did not fire on a swap")
    finally:
        C.DATA = real_data
        if restore_registry is not None:
            restore_registry()
        shutil.rmtree(tmp, ignore_errors=True)

    assert f"POOL_MEMBERSHIP_DRIFT: left=['{depart}'] arrived=['{arrive}']" in coverage, coverage
    assert depart in coverage, coverage
    # The digest fires too, and that second red is NOT a substitute: it names neither unit.
    assert "registry MEMBERSHIP moved" in digest, digest
    assert depart not in digest and arrive not in digest, digest


def test_edc_split_coverage_has_no_undeclared_gap():
    """Every registry unit is COVERED or DECLARED by name. No skip guard -- see the block banner."""
    gaps = split_coverage_gaps(_edc_coverage_rows(), EDC_DECLARED_SPLITLESS,
                               EDC_DECLARED_ZERO_HOLDOUT, EDC_DECLARED_STALE_SPLIT,
                               EDC_REGISTRY_COUNT, EDC_REGISTRY_SLUGS)
    assert not gaps, "EDC split coverage:\n  " + "\n  ".join(gaps)


def test_declared_zero_holdout_matches_the_engines_own_split_postcondition():
    """The six units this file declares as debt are the six the DECLARED split stage rejects.

    Two mechanical readers of THE SAME 97 FILES -- one per unit, each the manifest `load_split`
    resolves, which is the file every runtime consumer opens. Both sides go through
    `_winning_split_path`, and the (slug, version) pairs the two resolve are asserted equal below,
    so the agreement is over one file set rather than over whatever each happened to open.

    This block is NOT the first owner of the empty-holdout predicate -- `_split_integrity_ok` in
    run_stage rejects an empty holdout outright and execution_contract.py:293 wires it as the
    split stage's "split_integrity" postcondition -- but that checker is unreachable at gate
    time: it runs only when the split stage EXECUTES, over that stage's declared output patterns.
    It never enumerates the registry, never sees the splitless unit, and never compares a manifest
    to the current graph. What it does own, this test binds to: the declared-debt set may not drift
    away from the engine's own verdict on the same six files.
    """
    import run_stage
    from pipeline.editions import iter_editions

    # SUPERSEDED MANIFESTS ARE NOT GRADED. Enumerating ("v2", "v3") reads 147 version files while
    # the estimand holds 97 winning manifests, and since split versions are IMMUTABLE the failing
    # older file survives every repair: a floor-policy v3 for one of the six would retire it from
    # the coverage side (its holdout is no longer empty) while the postcondition side kept rejecting
    # its frozen v2, and NO value of EDC_DECLARED_ZERO_HOLDOUT made both green. Measured no-op at
    # HEAD -- both reads return the identical six -- and the only fix for that deadlock.
    resolved, zero_holdout_witness, other_reject = {}, set(), set()
    for ed in iter_editions("edc", with_raw=False):
        path, ver = _winning_split_path(ed.dir)
        if path is None:
            continue
        resolved[ed.slug] = ver
        if run_stage._split_integrity_ok(path):
            continue
        manifest = run_stage._load_json(path)
        holdout_empty = isinstance(manifest, dict) and not manifest.get("holdout_ids")
        (zero_holdout_witness if holdout_empty else other_reject).add(ed.slug)
    rejected = zero_holdout_witness | other_reject
    # RC-4, the standing pin against reintroducing the version-scope mismatch: the path reader and
    # the manifest reader must resolve the SAME version for every unit. Restore a hand-ordered
    # ("v2", "v3") walk in `_winning_split_path` and this names every dual-series unit.
    by_manifest = {ed.slug: (S.load_split(ed.dir) or {}).get("version")
                   for ed in iter_editions("edc", with_raw=False)
                   if S.load_split(ed.dir) is not None}
    assert resolved, (
        "VACUOUS: 0 EDC units resolved a split manifest - the two readers cannot be compared over "
        "an empty set")
    assert set(resolved.items()) == set(by_manifest.items()), (
        "the two split readers resolve DIFFERENT versions -- _winning_split_path and "
        "S.load_split must open one and the same file per unit, or every agreement below is "
        f"between different file sets: {sorted(set(resolved.items()) ^ set(by_manifest.items()))}")
    assert rejected, (
        "VACUOUS: the split postcondition rejected nothing - it cannot be a witness. Either no"
        " EDC split manifest survives on disk or _split_integrity_ok stopped rejecting anything")
    # THE WITNESS BINDING. EDC_DECLARED_ZERO_HOLDOUT means what it says: the units whose winning
    # manifest carries an EMPTY holdout — exactly the negation of the tightness guard's "now has a
    # non-empty holdout". Comparing against the whole seven-way verdict instead is the measured
    # deadlock: a floored v3 outside the frac band flips the tightness side without flipping the
    # verdict side, and no declaration state is green.
    assert zero_holdout_witness == EDC_DECLARED_ZERO_HOLDOUT, (
        "the declared zero-holdout set and the winning manifests' actual holdouts have diverged: "
        f"empty-holdout witness {sorted(zero_holdout_witness)}, EDC_DECLARED_ZERO_HOLDOUT is "
        f"{sorted(EDC_DECLARED_ZERO_HOLDOUT)}. A unit here gained or lost an empty holdout on its "
        "WINNING manifest - update the declaration, and if the new manifest is rejected for some "
        "OTHER reason it belongs in EDC_DECLARED_POSTCOND_DEBT, not here")
    assert other_reject == EDC_DECLARED_POSTCOND_DEBT, (
        "UNDECLARED_POSTCOND_REJECT: the split postcondition rejects these units for a reason "
        f"OTHER than an empty holdout: {sorted(other_reject - EDC_DECLARED_POSTCOND_DEBT)} "
        f"(stale declarations: {sorted(EDC_DECLARED_POSTCOND_DEBT - other_reject)}). "
        "_split_integrity_ok has SEVEN return-False sites (not a dict; train/holdout missing or "
        "holdout empty; train and holdout overlapping; a missing decontamination block on v2/v3; a "
        "non-zero alias-leak count; a coverage shortfall against a recorded n_unique_triple_ids; a "
        "realized frac outside the 0.5x-2.0x band) - open the manifest, name the reason, and "
        "declare the unit in EDC_DECLARED_POSTCOND_DEBT so the debt is visible, never silent")


def test_edc_splits_reproduce_under_their_recorded_parameters():
    """Re-derive EVERY split-carrying unit -- v3 closure units included, not just the empty-id_map
    half where canonical_subject is the identity and the assertion is f(x)==f(x).

    Each unit is re-cut with its OWN recorded seed/holdout_frac and the LIVE id_map, so the claim
    does not rest on the pinned literals. PARAM_DRIFT and the train/holdout disjointness check run
    over ALL of them, closure units and declared-stale units too.
    """
    from pipeline.editions import iter_editions
    from pipeline.generic_expand import load_all_triples

    reproduced, problems, checked = {}, [], 0
    leak_checked = set()
    for ed in iter_editions("edc", with_raw=False):
        manifest = S.load_split(ed.dir)
        if manifest is None:
            continue
        ver = manifest.get("version")
        if (manifest.get("seed") != EDC_SPLIT_SEED
                or manifest.get("holdout_frac") != EDC_SPLIT_HOLDOUT_FRAC):
            problems.append(
                f"PARAM_DRIFT {ed.slug} ({ver}): manifest records seed={manifest.get('seed')} "
                f"holdout_frac={manifest.get('holdout_frac')}, the declared cut is "
                f"seed={EDC_SPLIT_SEED} holdout_frac={EDC_SPLIT_HOLDOUT_FRAC} - a different "
                f"experiment wearing the same filename")
        # DISJOINTNESS runs for EVERY manifest, BEFORE any declaration-keyed skip. A blanket
        # `continue` here used to exempt a declared-stale unit from re-derivation AND from every
        # content check after it, so the one set that is documented as a growing queue handed out
        # total content blindness. A leak needs no re-cut to detect and is meaningful on a
        # manifest whose graph moved, so the skip below is narrowed to what it is FOR.
        # (split_coverage_gaps emits the same class from the row side; the redundancy is
        # deliberate -- this is the reader whose skip created the hole.)
        overlap = set(manifest["train_ids"]) & set(manifest["holdout_ids"])
        leak_checked.add(ed.slug)
        if overlap:
            # ONE OWNER for the sentence. This reader used to author its own copy of it.
            problems.append(_leaked_holdout_gap(ed.slug, ver, len(overlap)))
        if ed.slug in EDC_DECLARED_STALE_SPLIT:
            continue                    # its graph moved after the cut; declared, not re-derived
        checked += 1
        triples = load_all_triples(ed)
        train, holdout, _meta = S.make_split_v2(
            triples, manifest["holdout_frac"], manifest["seed"],
            S.load_id_map(ed.dir, domain="edc"))
        if train == set(manifest["train_ids"]) and holdout == set(manifest["holdout_ids"]):
            reproduced[ver] = reproduced.get(ver, 0) + 1
        else:
            problems.append(
                f"WOULD_DIFFER {ed.slug} ({ver}): re-cutting with the live id_map and the "
                f"manifest's own recorded parameters moves "
                f"{len(set(manifest['train_ids']) ^ train)} train / "
                f"{len(set(manifest['holdout_ids']) ^ holdout)} holdout ids")
    assert not problems, "EDC split re-derivation:\n  " + "\n  ".join(problems)
    # VACUITY, the mirror of the biconditional's. Padding EDC_DECLARED_STALE_SPLIT until nothing
    # is re-derived is the cheapest way to silence WOULD_DIFFER, so an empty class is itself loud.
    assert checked, (
        "VACUOUS: 0 EDC units were re-derived - either no split manifest survives on disk or "
        "EDC_DECLARED_STALE_SPLIT has been padded until the measured class is empty")
    assert leak_checked, (
        "VACUOUS: 0 EDC manifests were checked for train/holdout disjointness - the check must "
        "run over every manifest on disk and before every declaration-keyed skip")
    # PAIRWISE DISJOINT, asserted BEFORE the ordering guard so the right message wins. The guard
    # below used to read `<= leak_checked | EDC_DECLARED_SPLITLESS`, and that union term existed
    # for exactly one reason: to tolerate a unit declared BOTH stale and splitless, an incoherent
    # state nothing else reports. Tolerating it also blunted the guard (a splitless unit can never
    # be disjointness-checked, so the union let any such unit through). The sets are disjoint
    # today; make the incoherence loud instead of absorbing it, and the guard reads plainly.
    for a_name, a, b_name, b in (
        ("EDC_DECLARED_SPLITLESS", EDC_DECLARED_SPLITLESS,
         "EDC_DECLARED_ZERO_HOLDOUT", EDC_DECLARED_ZERO_HOLDOUT),
        ("EDC_DECLARED_SPLITLESS", EDC_DECLARED_SPLITLESS,
         "EDC_DECLARED_STALE_SPLIT", EDC_DECLARED_STALE_SPLIT),
        ("EDC_DECLARED_ZERO_HOLDOUT", EDC_DECLARED_ZERO_HOLDOUT,
         "EDC_DECLARED_STALE_SPLIT", EDC_DECLARED_STALE_SPLIT),
        # A postcondition-debt unit's manifest EXISTS and has a NON-empty holdout (else it belongs
        # in ZERO_HOLDOUT), so overlap with either of those two hides which condition is true.
        ("EDC_DECLARED_POSTCOND_DEBT", EDC_DECLARED_POSTCOND_DEBT,
         "EDC_DECLARED_SPLITLESS", EDC_DECLARED_SPLITLESS),
        ("EDC_DECLARED_POSTCOND_DEBT", EDC_DECLARED_POSTCOND_DEBT,
         "EDC_DECLARED_ZERO_HOLDOUT", EDC_DECLARED_ZERO_HOLDOUT),
    ):
        assert not (a & b), (
            f"DECLARATION SETS OVERLAP: {sorted(a & b)} is declared in BOTH {a_name} and {b_name} "
            "- a unit exhibits one condition or the other; declaring both hides which is true and "
            "makes the tightness assertions unable to retire either declaration")
    assert EDC_DECLARED_STALE_SPLIT <= leak_checked, (
        "a unit declared STALE carries a manifest that was never disjointness-checked: "
        f"{sorted(EDC_DECLARED_STALE_SPLIT - leak_checked)} - the "
        "declaration-keyed skip has moved back above the leak check")
    print(f"      re-derived under their own recorded parameters: v2={reproduced.get('v2', 0)} "
          f"v3={reproduced.get('v3', 0)} ({checked} checked, "
          f"{len(EDC_DECLARED_STALE_SPLIT)} declared stale); train/holdout disjointness measured "
          f"over {len(leak_checked)} manifests")


def test_a_full_alias_closure_that_zeroes_the_holdout_is_a_gap():
    """The zero-holdout class is FORWARD-REACHABLE -- this is the guard the next campaign needs.

    Run resolve and cut v3 (exactly the named next campaign for las_vegas/2026) and a full alias
    closure can collapse a healthy 40-fact holdout to zero: one component holding every triple is
    rejected wholesale by make_split_v2's per-predicate train-side guard, so the cut writes
    holdout_ids: [] and a file-existence predicate calls the unit covered. Pure -- no tree, no GPU.
    """
    triples = [{"subject_id": f"s{i % 40}", "predicate": f"p{i % 7}", "object": f"o{i}"}
               for i in range(400)]
    id_map = {f"s{i}": "canon" for i in range(40)}
    train_c, holdout_c, meta_c = S.make_split_v2(triples, 0.10, 42, id_map)
    train_e, holdout_e, meta_e = S.make_split_v2(triples, 0.10, 42, {})
    assert (meta_c["n_groups"], len(train_c), len(holdout_c)) == (1, 400, 0), meta_c
    assert (meta_e["n_groups"], len(train_e), len(holdout_e)) == (40, 360, 40), meta_e
    # A fresh cut partitions by construction (`train = {triple_id(t) for t in triples} - holdout`),
    # so overlap is 0 on both and the zero-holdout class is isolated from the leak class.
    assert not (train_c & holdout_c) and not (train_e & holdout_e)
    row = {"x/1": {"has_split": True, "ver": "v3", "holdout": len(holdout_c), "overlap": 0,
                   "unassigned": 0, "orphan": 0}}
    gaps = split_coverage_gaps(row, set(), set(), set(), 1, {"x/1"})
    assert len(gaps) == 1 and gaps[0].startswith("UNDECLARED_ZERO_HOLDOUT x/1 (v3)"), gaps
    row["x/1"]["holdout"] = len(holdout_e)
    assert split_coverage_gaps(row, set(), set(), set(), 1, {"x/1"}) == [], (
        "the healthy cut is not a gap")


def test_edc_v3_covers_exactly_the_units_with_a_live_alias_closure():
    """The r10 class invariant, both directions. 0 both ways today; asserted, not remembered."""
    from pipeline.editions import iter_editions

    eds = list(iter_editions("edc", with_raw=False))
    n_split = sum(1 for ed in eds if S.load_split(ed.dir) is not None)
    if not n_split:
        # Keyed on split ARTIFACTS existing -- a condition independent of what this test measures
        # (the fix probe 10 forced). A tree with no split at all is already loud elsewhere:
        # test_edc_split_coverage_has_no_undeclared_gap reports every unit UNDECLARED_SPLITLESS.
        print("  skip  test_edc_v3_covers_exactly_the_units_with_a_live_alias_closure"
              " (no EDC splits on disk)")
        return
    with_closure, with_v3, bad = [], [], []
    for ed in eds:
        n_map = len(S.load_id_map(ed.dir, domain="edc"))
        has_v3 = S.split_path(ed.dir, "v3").exists()
        if n_map:
            with_closure.append(ed.slug)
        if has_v3:
            with_v3.append(ed.slug)
        if n_map and not has_v3:
            bad.append(f"CLOSURE_WITHOUT_V3 {ed.slug}: live id_map has {n_map} entries but no "
                       f"split_v3 - the r10 alias-closure class is incomplete")
        if has_v3 and not n_map:
            bad.append(f"V3_WITHOUT_CLOSURE {ed.slug}: split_v3 exists but the live id_map is "
                       f"empty - most likely data/_build/resolve/edc was purged for disk; "
                       f"otherwise the id_map loader regressed to the r9 year-only key")
    # HARD assert in both directions. A disk sweep of data/_build/resolve/edc trips all 50 v3
    # units at once, and that is the intended loudness: losing the ability to audit 50 split
    # manifests is exactly the thing this engine should refuse to do quietly.
    assert not bad, "EDC v3/alias-closure biconditional:\n  " + "\n  ".join(bad)
    assert with_closure and with_v3, (
        f"VACUOUS PASS over {n_split} split-carrying units: {len(with_closure)} have a live alias "
        f"closure, {len(with_v3)} have a split_v3 - the biconditional held only because there was "
        f"nothing left to compare")


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} SPLIT-V2 TESTS PASS")
