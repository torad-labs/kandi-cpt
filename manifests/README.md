# manifests — exposure ledger, corpus manifests, lineage

Bookkeeping that turns the README's "each fact seen ~5×" blocker into a measured, balanced
quantity.

- **Exposure ledger:** per source-doc/fact → how many synthetic surfaces reference it → the
  resulting token-exposure count. The thing you balance to hit ~1000 exposures/fact on the
  facts that must become parametric, and to surface the long tail that's under-exposed.
- **Corpus manifests:** which `10_clean` + `20_synthetic` + `replay` inputs (and config
  version) produced a given `30_tokenized` shard — so any corpus build is reproducible.
- **Lineage:** synthetic-record → source-record links for invalidation when source changes.

This folder is the **home** for those artifacts; the tooling that populates them is its own
piece of work, driven by the fact-density rebuild. **Git:** committed.
