# scripts/dev — kandi-cpt developer tools

Ad-hoc dev/verification tooling promoted from `/tmp` scratch.

| script | what it does |
|---|---|
| `verify_all.py` / `verify_all.sh` | Adversarial build-completeness check: Python syntax, the config contract, the ACTION_CATALOG, and serve.py routes — run before declaring the platform build green. |
| `rm_fns.py` | String-aware Rust function/test remover: deletes named items (with attrs/docs) without breaking surrounding code. Used during festival-factory Rust refactors. |
