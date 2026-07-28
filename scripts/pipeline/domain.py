"""Domain manifest loader — the config-driven abstraction that makes the scrape→CPT→SFT pipeline
domain-general. A domain (configs/domains/<name>.yaml) declares its hierarchy, unit registry, ontology
refs, model, and the ordered pipeline stages (argv command templates). run_pipeline.py executes them.

The festival-specific code is just the EDC *instance* of this manifest; pointing the pipeline at a new
domain is "write a new <name>.yaml (+ its ontology schema/taxonomy/templates) and run one command."
"""
from __future__ import annotations

import yaml

import config as C

PY = str(C.ROOT / ".venv/bin/python")


class Stage:
    def __init__(self, spec: dict):
        self.name = spec["name"]
        self.desc = spec.get("desc", "")
        self.cmd = spec["cmd"]
        self.gpu = bool(spec.get("gpu", False))


# What each profile expects to be able to validate. An explicit `supports:` block overrides
# individual entries — the profile is the default, not a straitjacket.
RECIPE_PROFILES = {
    "generic_facts": {"generic_hydrate": True, "contrast": True, "reasoning": True, "sft": True},
    "bespoke_facts": {"generic_hydrate": False, "contrast": False, "reasoning": False, "sft": True},
    "style": {"generic_hydrate": False, "contrast": False, "reasoning": False, "sft": False},
}


class Domain:
    """Parsed configs/domains/<name>.yaml."""

    def __init__(self, name: str):
        self.name = name
        self.path = C.CONFIGS / "domains" / f"{name}.yaml"
        if not self.path.exists():
            raise FileNotFoundError(f"no domain manifest: {self.path}")
        self.spec = yaml.safe_load(self.path.read_text())
        # Path fork (docs/cpt-overhaul/05): facts | style | mixed. Default facts.
        raw_kind = str(self.spec.get("kind") or "facts").lower().strip()
        self.kind = raw_kind if raw_kind in {"facts", "style", "mixed"} else "facts"
        self.hierarchy = self.spec.get("hierarchy", ["festival", "city", "year"])
        self.registry = self.spec.get("registry", "")
        self.model_path = self.spec.get("model_path", str(C.MODEL_PATH))
        self.configs = self.spec.get("configs", {})
        self.ontology = self.spec.get("ontology", {})
        self.vars = self.spec.get("vars", {})
        # Recipe profile: what capability model this domain actually follows. EDC generates its
        # surfaces with a bespoke script; requiring it to expose generic hydrate/contrast templates
        # it never reads produced two permanent false gaps. Unset = generic_facts, so every existing
        # manifest keeps its current meaning.
        self.recipe_profile = str(self.spec.get("recipe_profile") or "generic_facts")
        self.supports = dict(RECIPE_PROFILES.get(self.recipe_profile, RECIPE_PROFILES["generic_facts"]))
        self.supports.update(self.spec.get("supports") or {})
        self.templates = self._load_templates()
        self.stages = [Stage(s) for s in self.spec.get("stages", [])]

    def _load_templates(self) -> dict:
        """Domain-authored content (Q/A templates + scaffold data) the engine seams read so a NEW
        domain 'swaps the templates' (manifest `:6`) without touching code. Resolved from the
        manifest's `templates:` path if set, else the convention configs/templates/<name>.yaml.
        Absent → {} — the engine seams then fall back to neutral, NON-festival defaults, so a domain
        that declares no templates gets no festival bleed."""
        rel = self.spec.get("templates") or f"configs/templates/{self.name}.yaml"
        path = C.ROOT / rel
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text()) or {}

    def stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def resolve(self, cmd: list, **extra) -> list[str]:
        """Substitute {placeholders} in an argv template from vars + configs + builtins + call-site extras."""
        sub = {"py": PY, "root": str(C.ROOT), "model_path": self.model_path,
               "cpt": self.configs.get("cpt", ""), "sft": self.configs.get("sft", ""),
               **self.vars, **{k: v for k, v in extra.items() if v is not None}}
        out = []
        for a in cmd:
            try:
                out.append(str(a).format(**sub))
            except KeyError as e:
                raise KeyError(f"stage cmd references unknown placeholder {e} (have: {sorted(sub)})")
        return out

    def select(self, only=None, frm=None, to=None) -> list[Stage]:
        """Pick stages by --only CSV, or a --from/--to window over the declared order."""
        names = self.stage_names()
        if only:
            want = [s.strip() for s in only.split(",") if s.strip()]
            bad = [w for w in want if w not in names]
            if bad:
                raise ValueError(f"unknown stage(s) {bad}; have {names}")
            return [st for w in want if (st := self.stage(w)) is not None]
        for w in (frm, to):
            if w and w not in names:
                raise ValueError(f"unknown stage '{w}'; have {names}")
        i = names.index(frm) if frm else 0
        j = names.index(to) + 1 if to else len(names)
        return self.stages[i:j]
