"""The new-domain proof — 'swap the templates' is REAL.

A NON-festival domain (pokemon-lore: character/region/organization, no venue/stage/day) ships its own
configs/templates/pokemon-lore.yaml, and the SAME engine code (build_sft's rel_q rendering, fact_expand's
ctx-driven generators) emits ITS sentences — never EDC's. This is what makes the manifest's documented
"copy this file, swap the templates" a live operation instead of a fiction. Deterministic — no LLM/GPU.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_new_domain_swap.py
"""
from __future__ import annotations

from pipeline.domain import Domain
from pipeline import fact_expand as FE

FESTIVAL_PREDS = {"located_at", "belongs_to_class", "serves", "requires", "practiced_by", "prohibits"}


def test_pokemon_templates_load_and_are_not_festival():
    """Domain.templates loads a non-festival domain's own templates (mechanism is domain-general)."""
    rel_q = ((Domain("pokemon-lore").templates or {}).get("sft_templates") or {}).get("rel_q") or {}
    assert rel_q, "pokemon-lore sft_templates.rel_q did not load — Domain.templates is not domain-general"
    assert "originates_from" in rel_q and "member_of" in rel_q, "pokemon predicates missing"
    assert not (set(rel_q) & FESTIVAL_PREDS), "festival predicates bled into the pokemon templates"


def test_build_sft_rel_q_renders_pokemon_qa():
    """The SAME build_sft rel_q render path (q.format(s,n) / a.format(s,o,n)) emits a POKEMON sentence."""
    rel_q = Domain("pokemon-lore").templates["sft_templates"]["rel_q"]
    qt, at = rel_q["originates_from"]
    q = qt.format(s="Ash Ketchum", n="Pokémon")
    a = at.format(s="Ash Ketchum", o="Kanto", n="Pokémon")
    assert q == "Which region is Ash Ketchum from?", q
    assert a == "Ash Ketchum originates from Kanto.", a
    assert "stage" not in a and "EDC" not in a, "festival vocabulary leaked into the pokemon answer"


def test_fact_expand_generator_is_template_driven():
    """The SAME fact_templates function, given a pokemon Ctx + templates, emits pokemon prose — proving the
    generator renders the domain's templates against its context, not hardcoded EDC strings."""
    ctx = FE.Ctx(festival="the Pokémon world",
                 templates={"fact_no_day": ["In {festival}, {artist} hails from the {c} region."],
                            "attractors": []})
    out = FE.fact_templates("Ash Ketchum", "kanto", "", ctx)
    assert out == ["In the Pokémon world, Ash Ketchum hails from the Kanto region."], out
    assert "EDC" not in out[0] and "stage" not in out[0], "festival vocabulary leaked into pokemon hydration"


if __name__ == "__main__":
    test_pokemon_templates_load_and_are_not_festival()
    test_build_sft_rel_q_renders_pokemon_qa()
    test_fact_expand_generator_is_template_driven()
    print("OK — new-domain proof: pokemon-lore swaps the templates; the same engine renders its own "
          "(non-festival) sentences. 'swap the templates' is a live operation.")
