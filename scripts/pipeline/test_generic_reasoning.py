"""Gate — generic_reasoning: the reasoning-linked surface class.

Two halves, tested as two halves. The DETERMINISTIC half (entity classes, walk enumeration, estimand
guards, budget) contains no generator at all. The LLM half renders through the SAME injectable
`generate_fn` seam generic_synth uses, so every case here runs with a stub and the whole suite costs
ZERO provider calls.

Every guard is proven in BOTH directions: the violation is dropped, and a sibling case that must
survive does — a guard that suppresses everything reports the same zero as a guard that works.
Neutral synthetic fixtures throughout: no case reads data/, so writing relations_reasoning.jsonl
later cannot turn this suite red.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_generic_reasoning.py
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pipeline.generic_reasoning import (
    OUT_NAME,
    SURFACE_FORM,
    assert_channel_healthy,
    assert_not_form_collapsed,
    assert_train_only,
    base_surface_lines,
    budget_for_fraction,
    build_prompt,
    class_aliases,
    class_names,
    entity_classes,
    enumerate_walks,
    form_metrics,
    grounding_violations,
    holdout_bridge_pairs,
    missing_endpoints,
    negation_markers,
    numeral_violations,
    parse_variations,
    plan_walks,
    reasoning_records,
    render_records,
    universe_matcher,
    walk_context,
    walk_shape,
)
from pipeline.split import triple_id

REL_Q = {
    "leads": ["Which organization does {s} lead?", "{s} leads {o}."],
    "located_in": ["Which region is {s} located in?", "{s} is located in {o}."],
    "member_of": ["Which organization is {s} a member of?", "{s} is a member of {o}."],
    "owns": ["What does {s} own?", "{s} owns {o}."],
    "rival_of": ["Who is {s} a rival of?", "{s} is a rival of {o}."],
    "needs_n": ["?", "{s} and {o} met at {n}."],
    "unadmitted_pred": ["Q?", "{s} is unadmittedly tied to {o}."],
    "no_declarative": ["Only a question?"],
}


def _t(subject_id, predicate, object_id, subject, obj, doc=None):
    """One triple in the shape `generic_expand.load_all_triples` hands downstream."""
    rec = {"subject_id": subject_id, "subject": subject, "predicate": predicate,
           "object_id": object_id, "object": obj, "attrs": {}}
    if doc:
        rec["provenance"] = {"doc_id": doc}
    return rec


def _index(triples):
    out = {}
    for t in triples:
        out.setdefault(triple_id(t), t)
    return out


def _frames(triples, id_map=None):
    classes = entity_classes(triples, id_map or {})
    aliases = class_aliases(triples, classes)
    universe = set()
    for names in aliases.values():
        universe |= names
    return {"classes": classes, "names_of": class_names(triples, classes),
            "aliases_of": aliases, "universe": universe}


# The canonical fixture: a 3-edge simple path Giovanni -> Team Rocket -> Kanto, plus a second
# region so shared-object and shared-subject shapes exist.
CHAIN = [
    _t("x:c:giovanni", "leads", "x:o:rocket", "Giovanni", "Team Rocket", doc="lore/rocket.md"),
    _t("x:o:rocket", "located_in", "x:r:kanto", "Team Rocket", "Kanto"),
]
SHARED_OBJECT = [
    _t("x:l:ecruteak", "located_in", "x:r:johto", "Ecruteak City", "Johto"),
    _t("x:l:goldenrod", "located_in", "x:r:johto", "Goldenrod City", "Johto"),
]
SHARED_SUBJECT = [
    _t("x:c:green", "leads", "x:o:gym", "Green", "Viridian Gym"),
    _t("x:c:green", "rival_of", "x:c:red", "Green", "Red"),
]
# A disconnected component: it puts "Red" in the unit's entity vocabulary without joining any walk,
# so a foreign-entity case can be isolated to exactly one walk.
ELSEWHERE = [_t("q:c:red", "rival_of", "q:c:blue", "Red", "Blue")]


def _stub_from_context(prefix="", extra_line=None, empty_for=None):
    """A deterministic stand-in for the model: it echoes the walk's own stated facts in a handful of
    shapes derived from the prompt. It is a PLUMBING fixture, never evidence about diversity."""

    def stub(system, user, **kw):
        facts = [line.strip() for line in system.splitlines() if line.startswith("    ")]
        head = system.splitlines()
        who = [ln for ln in head if "link" in ln and " to " in ln]
        subject_line = who[0] if who else ""
        # endpoints are named in the "Read together they link X to Y" line
        try:
            middle = subject_line.split("they link", 1)[1]
            x = middle.split(" to ", 1)[0].strip()
            y = middle.split(" to ", 1)[1].split(" through ", 1)[0].rstrip(":").strip()
        except (IndexError, ValueError):
            x, y = "", ""
        if empty_for and empty_for in system:
            return ""
        body = " ".join(facts)
        lines = [
            f"{prefix}Question: How are {x} and {y} related? Answer: {body}",
            f"{prefix}{body} That is the trail running from {x} all the way to {y}.",
            f"{prefix}A reader asks about {x}; the record shows {body} which reaches {y}.",
            f"{prefix}Trace it step by step: {body} Endpoint {x}, endpoint {y}.",
        ]
        if extra_line:
            lines.insert(0, extra_line.format(x=x, y=y))
        return "\n".join(lines)

    return stub


def _render(triples, stub, *, variations=4, budget=None, id_map=None, holdout_ids=frozenset(),
            max_walk=3, domain=None):
    uniq = _index(triples)
    frames = _frames(list(uniq.values()), id_map)
    return reasoning_records(
        uniq, uniq, REL_Q, "the unit", domain, frames["universe"], classes=frames["classes"],
        names_of=frames["names_of"], aliases_of=frames["aliases_of"], holdout_ids=holdout_ids,
        generate_fn=stub, variations=variations, budget=budget, max_walk=max_walk)


# --------------------------------------------------------------------- entity identity


def test_entity_classes_merges_two_ids_that_render_one_display_name():
    """kanto carries 42 display names under two node ids. Keyed on the raw id, a walk 'connects' an
    entity to itself through its own alias."""
    triples = [_t("pokemon_lore:species:pikachu", "owns", "x:c:ash", "Pikachu", "Ash Ketchum"),
               _t("pikachu:species:pikachu", "rival_of", "x:c:james", "Pikachu", "James")]
    classes = entity_classes(triples, {})
    assert classes["pokemon_lore:species:pikachu"] == classes["pikachu:species:pikachu"], classes


def test_entity_classes_merges_ids_resolve_already_merged():
    """The second equivalence: `resolve` merged them even though they render differently."""
    triples = [_t("a:o:above_and_beyond", "leads", "x:o:label", "Above and Beyond", "Anjunabeats"),
               _t("x:o:aab", "located_in", "x:r:uk", "Above & Beyond", "UK")]
    id_map = {"a:o:above_and_beyond": "x:o:aab"}
    classes = entity_classes(triples, id_map)
    assert classes["a:o:above_and_beyond"] == classes["x:o:aab"], classes


def test_entity_classes_representative_is_the_class_minimum():
    triples = [_t("zzz:c:n", "leads", "x:o:o", "N", "Team Plasma"),
               _t("aaa:c:n", "rival_of", "x:c:red", "N", "Red")]
    classes = entity_classes(triples, {})
    assert classes["zzz:c:n"] == "aaa:c:n", classes


def test_entity_classes_does_not_merge_genuinely_distinct_entities():
    """Not blanket collapse: different names, no id_map entry → different classes."""
    triples = CHAIN
    classes = entity_classes(triples, {})
    assert len({classes["x:c:giovanni"], classes["x:o:rocket"], classes["x:r:kanto"]}) == 3, classes


def test_class_name_is_deterministic_when_a_class_carries_two_spellings():
    triples = [_t("x:o:aab", "leads", "x:o:label", "Above & Beyond", "Anjunabeats"),
               _t("a:o:ab", "located_in", "x:r:uk", "Above and Beyond", "UK"),
               _t("x:o:aab", "rival_of", "x:c:rival", "Above & Beyond", "Rival")]
    classes = entity_classes(triples, {"a:o:ab": "x:o:aab"})
    names = class_names(triples, classes)
    cls = classes["x:o:aab"]
    assert names[cls] == "Above & Beyond", names[cls]        # most frequent wins, ties lexicographic
    assert class_aliases(triples, classes)[cls] == {"Above & Beyond", "Above and Beyond"}


def test_alias_self_connection_is_impossible_on_the_merged_graph():
    """The reviewer's counterexample as a required case: two Pikachu ids and a third entity produce
    'What links Pikachu to Pikachu?' on the RAW graph. On the class graph the two ids are one node,
    so the set is no longer a simple path and is never enumerated."""
    triples = [_t("pokemon_lore:species:pikachu", "owns", "x:c:ash", "Pikachu", "Ash Ketchum"),
               _t("x:c:ash", "rival_of", "x:c:james", "Ash Ketchum", "James"),
               _t("x:c:james", "rival_of", "pikachu:species:pikachu", "James", "Pikachu")]
    uniq = _index(triples)
    raw_classes = {i: i for t in triples for i in (t["subject_id"], t["object_id"])}
    raw_walks, _ = enumerate_walks(uniq, raw_classes, max_walk=3)
    merged = _frames(triples)["classes"]
    merged_walks, _ = enumerate_walks(uniq, merged, max_walk=3)
    raw3 = [w for w in raw_walks if len(w) == 3]
    assert raw3, "the raw-id graph must produce the degenerate 3-walk this guard exists for"
    assert [w for w in merged_walks if len(w) == 3] == [], merged_walks
    assert merged_walks, "not blanket suppression — the 2-edge walks still enumerate"


def test_degenerate_display_name_walk_is_dropped_even_without_an_alias_table():
    """Belt and braces: alias tables are never complete, so `walk_context` re-checks the RENDERED
    names and refuses a walk whose nodes are not distinct as written."""
    triples = [_t("id:1", "leads", "id:2", "Alpha", "Beta"),
               _t("id:2", "located_in", "id:3", "Beta", "Alpha")]        # id:3 renders "Alpha" again
    uniq = _index(triples)
    classes = {i: i for i in ("id:1", "id:2", "id:3")}
    names = {"id:1": "Alpha", "id:2": "Beta", "id:3": "Alpha"}
    aliases = {k: {v} for k, v in names.items()}
    walks, _ = enumerate_walks(uniq, classes, max_walk=2)
    assert walks, "the fixture must enumerate a walk for the guard to reject"
    ctx, reason = walk_context(walks[0], uniq, REL_Q, "u", classes, names, aliases)
    assert ctx is None and reason == "degenerate", (ctx, reason)


# --------------------------------------------------------------------- estimand


def test_holdout_bridge_walk_is_dropped():
    """Fact-level leak the id check cannot see: the walk's endpoints ARE a held-out triple's pair."""
    triples = CHAIN + [_t("x:c:giovanni", "located_in", "x:r:kanto", "Giovanni", "Kanto")]
    uniq = _index(triples)
    holdout = {triple_id(triples[2])}
    train = {tid: t for tid, t in uniq.items() if tid not in holdout}
    frames = _frames(list(uniq.values()))
    pairs = holdout_bridge_pairs(uniq, holdout, frames["classes"])
    assert pairs, "the fixture must produce a holdout endpoint pair"
    selected, stats = plan_walks(train, uniq, REL_Q, "u", None, classes=frames["classes"],
                                 names_of=frames["names_of"], aliases_of=frames["aliases_of"],
                                 holdout_ids=holdout, max_walk=3)
    assert stats["drops"]["holdout_bridge"] == 1, stats["drops"]
    assert selected == [], selected


def test_holdout_bridge_covers_internal_pairs_not_just_endpoints():
    """The guard must reject a walk that STATES a holdout pair anywhere, not only at its endpoints.

    Both round-2 reviewers converged on this independently. Checking `nodes[0], nodes[-1]` alone
    lets a longer walk narrate the held-out fact between two interior nodes: here the outer pair is
    (Oak, Red), which is not held out, while the walk still tells the reader that Agatha and Red are
    connected — which is exactly what the holdout triple (Red|opposes|Agatha) is probed on.
    Adjacent pairs are the walk's own TRAIN edges and stay exempt; only j - i >= 2 is a claim the
    walk makes without declaring it.
    """
    triples = [
        _t("x:c:oak", "rival_of", "x:c:agatha", "Professor Oak", "Agatha"),
        _t("x:c:agatha", "member_of", "x:o:elite", "Agatha", "Elite Four"),
        _t("x:o:elite", "leads", "x:c:red", "Elite Four", "Red"),
        _t("x:c:red", "rival_of", "x:c:agatha", "Red", "Agatha"),          # HELD OUT
    ]
    uniq = _index(triples)
    holdout = {triple_id(triples[3])}
    train = {tid: t for tid, t in uniq.items() if tid not in holdout}
    frames = _frames(list(uniq.values()))
    pairs = holdout_bridge_pairs(uniq, holdout, frames["classes"])
    assert pairs, "the fixture must produce a holdout pair"
    selected, stats = plan_walks(train, uniq, REL_Q, "u", None, classes=frames["classes"],
                                 names_of=frames["names_of"], aliases_of=frames["aliases_of"],
                                 holdout_ids=holdout, max_walk=3)
    leaked = [ctx for ctx in selected
              if {"Agatha", "Red"} <= set(ctx["rendered"])]
    assert not leaked, (
        f"walk states the held-out (Agatha, Red) connection between interior nodes: "
        f"{[c['rendered'] for c in leaked]}")
    assert stats["drops"]["holdout_bridge"] >= 1, stats["drops"]


def test_holdout_bridge_drop_is_not_blanket_suppression():
    """Same enumerator, a walk whose endpoints are NOT a holdout pair, survives."""
    triples = CHAIN + [_t("x:c:giovanni", "rival_of", "x:c:red", "Giovanni", "Red")]
    uniq = _index(triples)
    holdout = {triple_id(triples[2])}
    train = {tid: t for tid, t in uniq.items() if tid not in holdout}
    frames = _frames(list(uniq.values()))
    selected, stats = plan_walks(train, uniq, REL_Q, "u", None, classes=frames["classes"],
                                 names_of=frames["names_of"], aliases_of=frames["aliases_of"],
                                 holdout_ids=holdout, max_walk=3)
    assert stats["drops"]["holdout_bridge"] == 0, stats["drops"]
    assert len(selected) == 1, selected


def test_assert_train_only_fires_on_a_holdout_id_and_on_an_unsplit_id():
    """Positive membership: `not in holdout_ids` silently admits the ids in NEITHER side (140 of
    kanto's 736). Both failure modes are calibrated, not assumed."""
    rec = {"text": "t", "source_triple_ids": ["aaa", "bbb"]}
    for train_ids, holdout_ids, marker in (({"aaa"}, {"bbb"}, "HOLDOUT"), ({"aaa"}, set(), "UNSPLIT")):
        try:
            assert_train_only([rec], train_ids, holdout_ids)
        except SystemExit as exit_error:
            assert marker in str(exit_error), str(exit_error)
        else:
            raise AssertionError(f"{marker} leak was not detected")


def test_assert_train_only_passes_on_clean_records():
    assert_train_only([{"text": "t", "source_triple_ids": ["aaa"]}], {"aaa"}, {"bbb"}) is None


# --------------------------------------------------------------------- the LLM channel


def test_surface_text_comes_from_the_generator_not_from_a_template():
    """THE channel test. The emitted text is byte-identical to what the injected generator returned;
    no fixed question skeleton is synthesised in this module."""
    marker = "ZZQ Giovanni leads Team Rocket, and Team Rocket sits in Kanto, so Giovanni reaches Kanto."

    def stub(system, user, **kw):
        return marker

    records, stats = _render(CHAIN, stub, variations=4)
    assert [r["text"] for r in records] == [marker], records
    assert stats["llm_calls"] == 1, stats


def test_module_holds_no_surface_template_literal():
    """The class of the defect, not its instance: the repaired module must contain no string literal
    capable of BEING a surface. Checked over the AST with docstrings excluded, so prose describing
    the removed renderer does not launder a real one back in."""
    source = Path(__file__).with_name("generic_reasoning.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = {ast.get_docstring(node, clean=False) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docstrings]
    banned = ("Answer:", "connected to", "is connected", "What links", "How are")
    offenders = [(lit[:60], b) for lit in literals for b in banned if b in lit]
    assert offenders == [], offenders
    assert "FORMS" not in source, "the fixed-skeleton renderer constant is back"
    assert '"text": line,' in source, "the record text must be the generator's line, verbatim"


def test_prompt_states_the_walk_edges_the_preamble_and_nothing_else():
    seen = {}

    def stub(system, user, **kw):
        seen["system"] = system
        return "Giovanni leads Team Rocket and Team Rocket is located in Kanto, linking him to Kanto."

    uniq = _index(CHAIN)
    frames = _frames(CHAIN)
    reasoning_records(uniq, uniq, REL_Q, "the unit", None, frames["universe"],
                      classes=frames["classes"], names_of=frames["names_of"],
                      aliases_of=frames["aliases_of"], generate_fn=stub, variations=3,
                      preamble="DOMAIN PREAMBLE.")
    system = seen["system"]
    assert system.startswith("DOMAIN PREAMBLE."), system[:60]
    assert "Giovanni leads Team Rocket." in system and "Team Rocket is located in Kanto." in system
    assert "Kanto" in system and "Giovanni" in system
    assert "Red" not in system and "Johto" not in system      # nothing outside the walk is stated


def test_build_prompt_names_the_bridges_and_only_the_walk():
    uniq = _index(CHAIN)
    frames = _frames(CHAIN)
    walks, _ = enumerate_walks(uniq, frames["classes"], max_walk=2)
    ctx, _reason = walk_context(walks[0], uniq, REL_Q, "the unit", frames["classes"],
                                frames["names_of"], frames["aliases_of"])
    prompt = build_prompt(ctx, "", "PERSONA LINE.", 3)
    assert prompt.startswith("PERSONA LINE."), prompt[:40]
    assert "through Team Rocket" in prompt, prompt
    assert "Giovanni leads Team Rocket." in prompt and "Team Rocket is located in Kanto." in prompt


def test_render_half_is_usable_alone_on_planned_walks():
    """The two halves are separable, which is what makes --dry-run structurally call-free."""
    uniq = _index(CHAIN)
    frames = _frames(CHAIN)
    selected, plan_stats = plan_walks(uniq, uniq, REL_Q, "the unit", None, classes=frames["classes"],
                                      names_of=frames["names_of"], aliases_of=frames["aliases_of"])
    records, render_stats = render_records(selected, None, frames["universe"],
                                           generate_fn=_stub_from_context(), variations=2)
    assert plan_stats["llm_calls_planned"] == render_stats["llm_calls"] == 1, (plan_stats, render_stats)
    assert len(records) == 2, records


def test_planning_half_makes_zero_generator_calls():
    """`--dry-run` reports the call count a real run would make; it must be structurally unable to
    call. plan_walks takes no generate_fn at all — this pins that."""
    uniq = _index(CHAIN)
    frames = _frames(CHAIN)
    selected, stats = plan_walks(uniq, uniq, REL_Q, "u", None, classes=frames["classes"],
                                 names_of=frames["names_of"], aliases_of=frames["aliases_of"])
    assert stats["llm_calls_planned"] == len(selected) == 1, stats
    assert "llm_calls" not in stats, stats


def test_one_call_per_selected_walk_and_walks_are_ceil_budget_over_variations():
    calls = {"n": 0}

    def stub(system, user, **kw):
        calls["n"] += 1
        return _stub_from_context()(system, user, **kw)

    triples = CHAIN + SHARED_OBJECT + SHARED_SUBJECT
    records, stats = _render(triples, stub, variations=4, budget=9)
    assert stats["llm_calls"] == calls["n"], stats
    assert calls["n"] == 3, calls                 # ceil(9 / 4)
    assert len(records) <= 9, len(records)


def test_generator_receives_the_rotated_persona_and_temperature():
    seen = []

    def stub(system, user, temperature=None, **kw):
        seen.append((system.splitlines()[0], temperature))
        return _stub_from_context()(system, user)

    triples = CHAIN + SHARED_OBJECT + SHARED_SUBJECT
    _render(triples, stub, variations=1, budget=6)
    assert len({t for _, t in seen}) > 1, seen        # temperature schedule rotates
    assert len({p for p, _ in seen}) > 1, seen        # persona roster rotates


def test_a_two_argument_stub_still_works():
    """generic_synth's TypeError fallback, kept: a stub taking only (system, user) must not break."""
    records, _ = _render(CHAIN, lambda system, user: "Giovanni leads Team Rocket into Kanto here.")
    assert len(records) == 1, records


# --------------------------------------------------------------------- the grounding gate


def _one_bad_line(triples, extra_line, reason):
    """One walk, five candidate lines, the first violating `reason`. Asserts BOTH directions at once:
    the offender is dropped with its reason counted, and its four clean siblings from the SAME
    response survive — a guard that suppressed everything would fail the second half."""
    records, stats = _render(triples, _stub_from_context(extra_line=extra_line), variations=4)
    assert stats["drops"][reason] == 1, stats["drops"]
    assert sum(stats["drops"].values()) == 1, stats["drops"]
    assert len(records) == 4, (len(records), stats["drops"])
    assert not any(line_fragment in r["text"] for r in records
                   for line_fragment in (extra_line.split("{")[0][:12],) if line_fragment.strip())
    return stats


def test_generated_line_naming_a_foreign_entity_is_dropped():
    stats = _one_bad_line(CHAIN + ELSEWHERE, "{x} reaches {y}, a route Red also patrols.", "ungrounded")
    assert stats["examples"]["ungrounded"][0]["foreign_entities"] == ["Red"], stats["examples"]


def test_generated_line_inventing_a_numeral_is_dropped():
    stats = _one_bad_line(CHAIN, "{x} has led toward {y} since 1996.", "foreign_numeral")
    assert stats["examples"]["foreign_numeral"][0]["numerals"] == ["1996"], stats["examples"]


def test_generated_line_missing_an_endpoint_is_dropped():
    stats = _one_bad_line(CHAIN, "{x} leads Team Rocket, which is a large organisation.",
                          "missing_endpoint")
    assert stats["examples"]["missing_endpoint"][0]["missing"] == ["y"], stats["examples"]


def test_generated_line_that_denies_something_is_dropped():
    _one_bad_line(CHAIN, "{x} does not lead Team Rocket, yet reaches {y}.", "negated")


def test_repeated_generated_line_is_emitted_once():
    def stub(system, user, **kw):
        line = "Giovanni leads Team Rocket, which is located in Kanto, so he reaches Kanto."
        return "\n".join([line, line, line])

    records, stats = _render(CHAIN, stub, variations=4, budget=None)
    assert len(records) == 1, records
    assert stats["drops"]["duplicate"] == 2, stats["drops"]


def test_short_fragment_is_dropped():
    records, stats = _render(CHAIN, lambda s, u: "Giovanni.\nKanto.", variations=4)
    assert records == [], records
    assert stats["drops"]["too_short"] == 2, stats["drops"]


def test_every_guard_lets_a_grounded_line_through():
    """Not blanket suppression: the same enumerator, the same guards, four clean lines survive."""
    records, stats = _render(CHAIN, _stub_from_context(), variations=4)
    assert len(records) == 4, (len(records), stats["drops"])
    assert all(v == 0 for v in stats["drops"].values()), stats["drops"]
    rec = records[0]
    assert rec["surface_form"] == SURFACE_FORM and rec["split"] == "cpt"
    assert rec["relation_type"] == "leads"
    assert rec["entity_ids"] == ["Giovanni", "Team Rocket", "Kanto"], rec["entity_ids"]
    assert len(rec["source_triple_ids"]) == 2 and all(len(i) == 16 for i in rec["source_triple_ids"])
    assert rec["source_doc_ids"] == ["lore/rocket.md"], rec
    assert rec["synth_persona"], rec


def test_allowed_long_name_is_not_reported_as_its_shorter_substring():
    matcher = universe_matcher({"Kanto", "Kanto Region"})
    assert grounding_violations("Ecruteak is in the Kanto Region.", {"Kanto Region"}, matcher) == []
    assert grounding_violations("Ecruteak is in Kanto.", {"Kanto Region"}, matcher) == ["Kanto"]


def test_foreign_name_containing_a_licensed_one_is_not_masked_away():
    """The MIRROR of the test above, and the direction that was never covered.

    Masking the allowed names out before the scan closed the false-positive direction and opened
    its opposite: a FOREIGN name that CONTAINS a licensed one was destroyed by the mask before the
    matcher saw it. Found by round-3 review by swapping the `allowed` argument between the two
    members of the test above's own pair. On real kanto, 235 of 453 selected walks license a name
    that is a token-boundary substring of some other entity class.
    """
    matcher = universe_matcher({"Kanto", "Kanto Region"})
    assert grounding_violations("Ecruteak is in the Kanto Region.", {"Kanto"}, matcher) == \
        ["Kanto Region"]
    # …and the original direction must NOT regress.
    assert grounding_violations("Ecruteak is in the Kanto Region.", {"Kanto Region"}, matcher) == []


def test_a_distinct_entity_sharing_a_licensed_prefix_is_ungrounded():
    """The shipped counterexample, reduced. `Blue Oak` is a different class from `Blue`.

    Round 3 drove this end-to-end through render_records on real kanto: the selected walk
    ['Blue','Celio','Green','Ditty'] accepted "Blue is linked to Ditty, and Blue Oak belongs to the
    same chain." with ZERO drops, shipping a fabricated fact attributed to three real train triples.
    """
    matcher = universe_matcher({"Blue", "Blue Oak", "Ditty"})
    assert grounding_violations(
        "Blue is linked to Ditty, and Blue Oak belongs to the same chain.",
        {"Blue", "Ditty"}, matcher) == ["Blue Oak"]
    # Calibration, so a future edit cannot make this pass by disabling the gate: a wholly foreign
    # name is still caught, and a fully licensed line is still clean.
    assert grounding_violations("Blue is linked to Ditty via Absol.", {"Blue", "Ditty"},
                                universe_matcher({"Blue", "Ditty", "Absol"})) == ["Absol"]
    assert grounding_violations("Blue is linked to Ditty.", {"Blue", "Ditty"}, matcher) == []


def test_endpoint_is_not_satisfied_by_a_longer_name_containing_it():
    """`missing_endpoints` used a bare boundary search, so a mention of a DIFFERENT entity whose
    name contains the endpoint satisfied the endpoint check."""
    matcher = universe_matcher({"Blue", "Blue Oak", "Ditty"})
    assert missing_endpoints("Blue Oak is linked to Ditty.", {"Blue"}, {"Ditty"}, matcher) == ["x"]
    assert missing_endpoints("Blue is linked to Ditty.", {"Blue"}, {"Ditty"}, matcher) == []


def test_holdout_stated_as_the_walks_own_adjacent_edge_is_NOT_dropped_and_why():
    """PROVEN NO, kept as the record of a reverted fix.

    Round 3 added adjacent pairs to the holdout-bridge scan: a fact held out in one DIRECTION sits
    on the train side in the other, under a different triple_id, and both collapse to one class
    pair — kanto has (Lt. Surge|opposes|Red) TRAIN and (Red|opposes|Lt. Surge) HELD OUT.

    Round 4 measured that arm and it was INERT. An adjacent pair IS one of the walk's own TRAIN
    triples, and the deterministic floor renders those verbatim regardless, so refusing to render
    the walk cannot change whether the corpus states that holdout class pair. Both pairs the arm
    fired on were already stated by the floor (2 and 3 lines each), while the arm cost 1,542 walks
    and 4 of 521 train triples. Reverted.

    This test therefore asserts the CURRENT behaviour — the walk survives — so that anyone
    re-adding the arm has to come here and argue with the measurement rather than rediscover it.
    The leak is real but lives UPSTREAM in the floor, which states 2 of 72 holdout class pairs.
    """
    triples = [
        _t("x:c:surge", "rival_of", "x:c:red", "Lt. Surge", "Red"),          # TRAIN
        _t("x:c:red", "member_of", "x:o:elite", "Red", "Elite Four"),        # TRAIN
        _t("x:c:red", "rival_of", "x:c:surge", "Red", "Lt. Surge"),          # HELD OUT, reversed
    ]
    uniq = _index(triples)
    holdout = {triple_id(triples[2])}
    train = {tid: t for tid, t in uniq.items() if tid not in holdout}
    frames = _frames(list(uniq.values()))
    pairs = holdout_bridge_pairs(uniq, holdout, frames["classes"])
    assert pairs, "the fixture must produce a holdout class pair"
    selected, stats = plan_walks(train, uniq, REL_Q, "u", None, classes=frames["classes"],
                                 names_of=frames["names_of"], aliases_of=frames["aliases_of"],
                                 holdout_ids=holdout, max_walk=3)
    adjacent = [c for c in selected if {"Lt. Surge", "Red"} <= set(c["rendered"])]
    assert adjacent, (
        "the adjacency arm is reverted, so a walk whose OWN EDGE is the holdout class pair must "
        "still render — the floor states that fact anyway and dropping the walk buys nothing")
    # …while a NON-adjacent statement of a holdout pair is still dropped: the revert narrowed the
    # scan back to distance >= 2, it did not disable it.
    assert "holdout_bridge" in stats["drops"], stats["drops"]


def test_an_alias_spelling_of_a_merged_class_is_licensed_not_foreign():
    """A resolve-merged class can be written two ways; the stated edge may use either spelling, so
    BOTH must be licensed or the gate would reject its own prompt."""
    triples = [_t("x:o:aab", "leads", "x:o:label", "Above & Beyond", "Anjunabeats"),
               _t("a:o:ab", "located_in", "x:r:uk", "Above and Beyond", "UK")]
    id_map = {"a:o:ab": "x:o:aab"}
    uniq = _index(triples)
    frames = _frames(triples, id_map)
    walks, _ = enumerate_walks(uniq, frames["classes"], max_walk=2)
    ctx, reason = walk_context(walks[0], uniq, REL_Q, "u", frames["classes"], frames["names_of"],
                               frames["aliases_of"])
    assert ctx is not None, reason
    assert {"Above & Beyond", "Above and Beyond"} <= ctx["licensed_names"], ctx["licensed_names"]
    matcher = universe_matcher(frames["universe"])
    assert grounding_violations("Above and Beyond leads Anjunabeats from the UK.",
                                ctx["licensed_names"], matcher) == []


def test_numeral_and_negation_and_endpoint_helpers_are_calibrated():
    assert numeral_violations("met in 1996 at 3 pm", {"3"}) == ["1996"]
    assert numeral_violations("met in 1996", {"1996"}) == []
    assert negation_markers("this is not that") == ["not"]
    assert negation_markers("this is that") == []
    assert missing_endpoints("A and B", {"A"}, {"B"}) == []
    assert missing_endpoints("A only", {"A"}, {"B"}) == ["y"]


def test_parse_variations_strips_markers_and_quotes():
    raw = '- one line here\n2) another line here\n"a quoted line here"\n\n'
    assert parse_variations(raw) == ["one line here", "another line here", "a quoted line here"]


# --------------------------------------------------------------------- channel health


def test_empty_completion_is_counted_not_silently_successful():
    """`content or ""` turns an empty completion into a successful-looking empty string; qwen3-family
    models return exactly that without /no_think. It is counted at the call site."""
    records, stats = _render(CHAIN, lambda s, u: "", variations=4)
    assert records == [], records
    assert stats["drops"]["empty_completion"] == 1, stats["drops"]


def test_mostly_empty_channel_aborts_the_stage():
    try:
        assert_channel_healthy({"llm_calls": 10, "drops": {"empty_completion": 6}, "emitted": 3})
    except SystemExit as exit_error:
        assert "mostly empty" in str(exit_error), str(exit_error)
    else:
        raise AssertionError("a 60%-empty channel was accepted")
    assert_channel_healthy({"llm_calls": 10, "drops": {"empty_completion": 4}, "emitted": 3})


def test_zero_admissible_surfaces_aborts_rather_than_writing_an_empty_corpus():
    try:
        assert_channel_healthy({"llm_calls": 5, "drops": {"empty_completion": 0}, "emitted": 0})
    except SystemExit as exit_error:
        assert "0 admissible surfaces" in str(exit_error), str(exit_error)
    else:
        raise AssertionError("an empty corpus was accepted")


# --------------------------------------------------------------------- the form gate


def _skeleton_texts(n=120):
    """What the renderer this repair REPLACES produced: one skeleton, names swapped."""
    return [f"Question: How are E{i} and F{i} connected?\nAnswer: E{i} leads G{i}. G{i} is located "
            f"in F{i}. So E{i} is connected to F{i} through G{i}." for i in range(n)]


def test_form_collapsed_class_is_refused():
    """The instrument the reviewers used, promoted to a fail-closed gate. Measured on kanto at 906
    records: the fixed-skeleton renderer scores 19 effective forms (ratio 0.021) against 552-596 for
    an equal-mass relations_llm sample and 356 for the deterministic per-triple floor."""
    texts = _skeleton_texts()
    metrics = form_metrics(texts)
    assert metrics["form_ratio"] < 0.10, metrics
    try:
        assert_not_form_collapsed(texts, 0.20, "fixture")
    except SystemExit as exit_error:
        assert "form-collapsed" in str(exit_error), str(exit_error)
    else:
        raise AssertionError("a form-collapsed class was accepted")


def test_varied_class_passes_the_form_gate():
    """The pass direction, and a calibration of how demanding this ruler is: an EARLIER version of
    this fixture varied only three tokens inside a fixed sentence and scored 1 form over 120 records.
    Lexical jaccard at 0.55 collapses a shared skeleton no matter what is slotted into it — which is
    precisely why the three-template renderer scored 19 on 906 real records."""
    bank = [f"lexeme{i}" for i in range(120 * 6)]
    texts = [" ".join(bank[i * 6:(i + 1) * 6]) + " and so the record stands." for i in range(120)]
    metrics = assert_not_form_collapsed(texts, 0.20, "fixture")
    assert metrics["verdict"] == "pass" and metrics["form_ratio"] >= 0.20, metrics


def test_a_shared_skeleton_with_swapped_names_does_not_pass_the_form_gate():
    """Sibling of the cited defect: swapping the NAMES in one skeleton is what the removed renderer
    did, and it must not buy a passing form ratio however many names are swapped."""
    texts = [f"The archivist noted a link between {chr(65 + i % 26)}ville and the coastal works, "
             f"recorded in volume {i % 7} of the register." for i in range(120)]
    assert form_metrics(texts)["form_ratio"] < 0.20, form_metrics(texts)


def test_form_metrics_names_its_ruler_and_reports_both():
    metrics = form_metrics(["a b c d", "e f g h"])
    assert "effective_forms" in metrics["ruler"], metrics
    assert "distinct_2" in metrics and "also_reported" in metrics, metrics


# --------------------------------------------------------------------- pool shape


def test_shared_object_and_shared_subject_pairs_are_walks():
    """The shapes a literal 'graph walk' reading discards. On kanto they are the majority of the
    renderable pool, and dropping them costs 42% of train-graph coverage."""
    for triples, shape in ((SHARED_OBJECT, "shared_object"), (SHARED_SUBJECT, "shared_subject")):
        uniq = _index(triples)
        frames = _frames(triples)
        walks, _ = enumerate_walks(uniq, frames["classes"], max_walk=2)
        assert len(walks) == 1, (shape, walks)
        assert walk_shape(walks[0], uniq, frames["classes"]) == shape


def test_directed_chain_is_still_a_walk():
    uniq = _index(CHAIN)
    frames = _frames(CHAIN)
    walks, _ = enumerate_walks(uniq, frames["classes"], max_walk=2)
    assert len(walks) == 1 and walk_shape(walks[0], uniq, frames["classes"]) == "chain", walks


def test_inverse_pair_sharing_both_entities_is_not_a_walk():
    triples = [_t("x:c:giovanni", "leads", "x:o:rocket", "Giovanni", "Team Rocket"),
               _t("x:o:rocket", "member_of", "x:c:giovanni", "Team Rocket", "Giovanni")]
    uniq = _index(triples)
    frames = _frames(triples)
    walks, _ = enumerate_walks(uniq, frames["classes"], max_walk=2)
    assert walks == [], walks


def test_three_edge_star_is_not_a_walk_but_its_pairs_are():
    hub = [_t("x:h:hub", "leads", "x:a:a", "Hub", "A"),
           _t("x:h:hub", "leads", "x:b:b", "Hub", "B"),
           _t("x:h:hub", "leads", "x:c:c", "Hub", "C")]
    uniq = _index(hub)
    frames = _frames(hub)
    walks, _ = enumerate_walks(uniq, frames["classes"], max_walk=3)
    assert walks and all(len(w) == 2 for w in walks), walks


def test_hub_guard_bounds_enumeration_and_reports_what_it_skipped():
    hub = [_t("x:h:hub", "leads", f"x:e:{i}", "Hub", f"Node {i}") for i in range(12)]
    uniq = _index(hub)
    frames = _frames(hub)
    wide, stats_wide = enumerate_walks(uniq, frames["classes"], max_walk=2, max_hub_degree=64)
    narrow, stats_narrow = enumerate_walks(uniq, frames["classes"], max_walk=2, max_hub_degree=8)
    assert len(wide) == 66 and stats_wide["hub_skipped"] == [], (len(wide), stats_wide)
    assert narrow == [], narrow
    assert stats_narrow["hub_skipped"][0]["degree"] == 12, stats_narrow


def test_hub_guard_is_inert_below_its_threshold():
    """Correctly-unmoved control: kanto's maximum entity degree is 31, so the default 64 changes
    nothing there. Proven on a fixture rather than assumed."""
    uniq = _index(CHAIN + SHARED_OBJECT + SHARED_SUBJECT)
    frames = _frames(list(uniq.values()))
    a, sa = enumerate_walks(uniq, frames["classes"], max_walk=3, max_hub_degree=64)
    b, sb = enumerate_walks(uniq, frames["classes"], max_walk=3, max_hub_degree=1000)
    assert a == b and sa["hub_skipped"] == sb["hub_skipped"] == [], (sa, sb)


# --------------------------------------------------------------------- budget


def test_budget_arithmetic_is_a_share_of_the_final_corpus():
    limit, info = budget_for_fraction(8159, 0.10)
    assert limit == 906, limit
    assert abs(info["share_of_final"] - 906 / (8159 + 906)) < 1e-4, info
    assert budget_for_fraction(0, 0.10)[0] == 0
    assert budget_for_fraction(100, 1.5)[0] == 0


def test_budget_base_excludes_this_class_so_reruns_do_not_compound():
    with tempfile.TemporaryDirectory() as tmp:
        hyd = Path(tmp) / "25_hydrated"
        hyd.mkdir(parents=True)
        (hyd / "relations.jsonl").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        (hyd / OUT_NAME).write_text("x\n" * 100, encoding="utf-8")
        total, per = base_surface_lines(SimpleNamespace(dir=Path(tmp), slug="u"))
    assert total == 5, (total, per)
    assert OUT_NAME not in per, per


def test_budget_truncates_and_reports_the_shortfall():
    records, stats = _render(CHAIN + SHARED_OBJECT, _stub_from_context(), variations=4, budget=3)
    assert len(records) == 3, len(records)
    assert stats["budget_shortfall"] == 0, stats


# --------------------------------------------------------------------- domain availability


def test_taxonomy_domain_is_unavailable_and_never_enumerates():
    """EDC pins relation_type to semantic labels, so its raw predicates are inadmissible. That is a
    reported state — 0 records, no crash — and it must cost nothing: a 792-triple unit with a
    268-degree hub must not enumerate a pool before discovering it cannot label it."""
    triples = [_t("e:a:x", "belongs_to_class", "e:b:y", "X", "Y"),
               _t("e:b:y", "belongs_to_class", "e:c:z", "Y", "Z")]
    uniq = _index(triples)
    frames = _frames(triples)
    selected, stats = plan_walks(uniq, uniq, REL_Q, "u", "edc", classes=frames["classes"],
                                 names_of=frames["names_of"], aliases_of=frames["aliases_of"])
    assert selected == [] and stats["class_unavailable"], stats
    assert stats["walks_enumerated"] == 0, stats


def test_a_single_unlabelable_predicate_drops_only_the_walks_it_heads():
    """Not blanket suppression, on a real domain enum."""
    triples = [_t("a:c:alpha", "unadmitted_pred", "m:o:mid", "Alpha", "Mid"),
               _t("m:o:mid", "leads", "z:o:zeta", "Mid", "Zeta"),
               _t("m:o:mid", "located_in", "x:r:kanto", "Mid", "Kanto")]
    uniq = _index(triples)
    frames = _frames(triples)
    selected, stats = plan_walks(uniq, uniq, REL_Q, "u", "pokemon-lore", classes=frames["classes"],
                                 names_of=frames["names_of"], aliases_of=frames["aliases_of"])
    assert stats["drops"]["unlabelable"] >= 1, stats["drops"]
    assert selected, "the admissible walk must survive"
    assert all(ctx["relation_type"] in ("leads", "located_in") for ctx in selected), selected


def test_unauthorable_template_is_a_counted_drop_not_a_partial_sentence():
    triples = [_t("x:a:a", "no_declarative", "x:b:b", "A", "B"),
               _t("x:b:b", "leads", "x:c:c", "B", "C")]
    uniq = _index(triples)
    frames = _frames(triples)
    selected, stats = plan_walks(uniq, uniq, REL_Q, "u", None, classes=frames["classes"],
                                 names_of=frames["names_of"], aliases_of=frames["aliases_of"])
    assert selected == [] and stats["drops"]["unauthorable"] == 1, stats["drops"]


def test_unit_name_placeholder_is_filled_from_the_unit_not_left_raw():
    triples = [_t("x:a:a", "needs_n", "x:b:b", "A", "B"),
               _t("x:b:b", "leads", "x:c:c", "B", "C")]
    seen = {}

    def stub(system, user, **kw):
        seen["system"] = system
        return "A and B met at Silver Conference, and B leads C, linking A to C."

    _render(triples, stub, variations=1)
    assert "met at the unit." in seen["system"], seen["system"]


# --------------------------------------------------------------------- determinism


def test_output_is_byte_identical_across_processes_with_different_hash_seeds():
    """hash() is salted per process; a walk id or a form rotation keyed on it is reproducible in one
    interpreter and wrong across two. This runs the real code twice, in two interpreters."""
    script = (
        "import sys, hashlib, json\n"
        "sys.path.insert(0, %r)\n"
        "from pipeline.test_generic_reasoning import CHAIN, SHARED_OBJECT, SHARED_SUBJECT, "
        "_render, _stub_from_context\n"
        "recs, _ = _render(CHAIN + SHARED_OBJECT + SHARED_SUBJECT, _stub_from_context(), "
        "variations=4)\n"
        "print(hashlib.sha256(json.dumps(recs, sort_keys=True).encode()).hexdigest())\n"
    ) % str(Path(__file__).resolve().parents[1])
    digests = []
    for seed in ("0", "12345"):
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                             env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
                                  "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
        assert out.returncode == 0, out.stderr[-800:]
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1] and digests[0], digests


def test_module_uses_no_salted_builtin_hash():
    source = Path(__file__).with_name("generic_reasoning.py").read_text(encoding="utf-8")
    assert " hash(" not in source and "=hash(" not in source, "builtin hash() is salted per process"


if __name__ == "__main__":
    # Auto-discover so an appended case can never be defined-but-unrun.
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} REASONING-SURFACE TESTS PASS")
