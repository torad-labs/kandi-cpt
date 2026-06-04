"""qgre-cpt — the domain-agnostic CPT data-quality layer.

`spec`  reads a domain manifest's quality sections (entity_types / predicates / relations / quality).
`gates` runs the deterministic quality-gate suite that turns the corpus-audit failure modes into
        machine-checkable contracts (balance · exposure-cap · faithfulness · id-integrity · source-vs-gap).

See docs/qgre_data_pipeline_audit.md for the defects these gates prevent from recurring.
"""
