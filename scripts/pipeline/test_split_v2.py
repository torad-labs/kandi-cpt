#!/usr/bin/env python3
"""Red-green proofs for alias+inverse-closed split_v2."""
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


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} SPLIT-V2 TESTS PASS")
