"""EDC raw-data scrapers — fills data/<festival>/<city>/<year>/00_raw/<source>/.

Two fetchers (HTTP + patched browser) behind one interface; source modules pick
the right one. See scripts/scrape.py for the CLI and docs/ORGANIZATION.md for the
stage this feeds (00_raw -> 10_clean). Acquisition only — extraction is the next stage.
"""
