from pipeline import span_match as SM


def test_generic_span_match_uses_selected_domain_entities_and_facts():
    SM.configure("millbrook-hollow", "booster_1")
    text = "Elowen Dask teaches at Hearthmere Hall."

    spans = SM.char_spans(text)
    assert [text[start:end] for start, end in spans] == ["Elowen Dask", "Hearthmere Hall"]
    fact_ids = SM.fact_ids_in(text)
    assert len(fact_ids) == 1
    assert fact_ids[0] != "Hol|edc_las_vegas|na"


def test_generic_span_match_does_not_attribute_fact_on_subject_only():
    SM.configure("millbrook-hollow", "booster_1")
    assert SM.fact_ids_in("Elowen Dask waved from the doorway.") == []
