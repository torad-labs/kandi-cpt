# kandi-cpt

**CPT / knowledge-install training engine** for Torad. Turns documents about a subject into a
relationally-complete corpus and continues-pretrains a small model (Gemma 4 E2B) so that subject's
knowledge lives in the weights.

This is **not** the orchestration agent. Conductor (`conductor-app`) drives this engine over SSH /
`run_pipeline.py` / bash. Training competence lives **here** (and in conductor's knowledge playbooks),
not in the driver's weights.

| | |
|---|---|
| **Precision** | **fp4 (Quartet-II NVFP4) + int8/fp8 master** — never raise (see below) |
| **Reference domain** | EDC (`configs/domains/edc.yaml`) — byte-identity goldens gate every generalization |
| **Base model (runtime)** | `~/Documents/dev/infra/models/gemma-4-E2B-it-int8-v2` (int8_rowwise_v2) |
| **Cloud image** | `ghcr.io/torad-labs/conductor-trainer` (private) |
| **Paid smoke** | **PROVEN 2026-08-02** — receipt `logs/phase_r18_cloud_smoke_rehearsal.json` |

---

## Read this first (agents)

1. **`CLAUDE.md`** — non-negotiables, precision law, domain-general architecture, GPU safety. **Source of
   truth for "do not break".** This README is the map; CLAUDE.md is the law.
2. **`docs/ORGANIZATION.md`** — data layout, lifecycle, what is safe to delete.
3. **`docs/cloud-trainer-image.md`** — build/publish/point-conductor-at the cloud image (operator page).
4. **`docs/DATA_PIPELINE.md`** — stage semantics when editing generators.
5. **Receipts under `logs/phase_*.json`** — what was actually measured (prefer over prose status).

If CLAUDE.md and this file disagree on methodology or precision, **believe CLAUDE.md**.

---

## Non-negotiables (summary)

### 1. Never raise training precision

Training is **NVFP4 fp4 weights + int8/fp8 masters + stochastic rounding + 4-bit optimizer states**.
Proven and bit-exact. Do **not** silently introduce `bf16`/`fp16`/`fp32` residency, `autocast`, or
"safer" full-precision optimizer states. Raising precision is human-only, evidence-backed.

- Gate: `scripts/qgre/check_no_bf16.sh` (via `scripts/run_tests.sh`)
- Override (operator only): `QGRE_PRECISION_OK=1`

### 2. EDC output is the hard gate

EDC is the reference domain. Generator changes must keep per-generator **sha256 byte-identity**
goldens green (`scripts/pipeline/test_*_general.py`, `_golden/`). A diverging EDC byte is a regression.

### 3. GPU / host safety

- `nvidia-smi` before heavy runs.
- **Never two FP4 trainers at once** (1200W PSU) — `scripts/torad_actions.py` holds `data/_build/gpu.lock`.
- Permanent artifacts stay **in-project** (`checkpoints/`, `data/`), never `/tmp` or `~/.cache`.
- Heavy jobs: detached `systemd-run --user` with memory caps (`MemoryHigh=13G` / `MemoryMax=14G` pattern).

---

## What the engine does

```
scrape → extract(10_clean) → curate → structured(15_structured)
  → meta → hydrate(25_hydrated) → synth → probes → audit
  → tokenize(30_tokenized) → cpt(fp4) → build_sft → sft(+gate)
```

Driven by:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_pipeline.py --domain <d> [--check] [--from …] [--to …]
```

`--check` is R3 preflight (validate refs, abort). Domain config: `configs/domains/<d>.yaml` +
`configs/templates/<d>.yaml`.

### A domain is data, not code

| You author | Engine reads via |
|---|---|
| `configs/domains/<d>.yaml` — ontology, stages, quality | `pipeline/domain.py`, `ontology.py`, `schema_validate.py` |
| `configs/templates/<d>.yaml` — meta/SFT/hydrate/synth templates | `Domain.templates` |
| `scripts/scrape/sources/<d>.py` — **only real new code** (acquire) | `scrape/__main__.py` |

Skeletons: `configs/domains/_TEMPLATE.yaml`, `configs/templates/_TEMPLATE.yaml`,
`scripts/scrape/sources/_TEMPLATE.py`.

**Generic path (new domains):** `generic_expand.py`, `generic_synth.py`, `generic_probes.py`.  
**Festival-specific (EDC only):** `fact_expand.py`, `synth_corpus.py` specialist paths, Rust
`festival-factory/` — see `festival-factory/DE-FESTIVALIZATION.md`.

Self-test that a festival assumption does not leak onto the generic path:
`scripts/pipeline/test_selftest_domain_e2e.py` (fixture domain `_selftest`).

---

## Layout

| Path | Tracked | Role |
|---|---|---|
| `data/<domain>/<unit>/` | mixed | One unit (EDC: `data/edc/<city>/<year>/` — **always two levels**) |
| `  00_raw/` | ✗ | Scraped sources |
| `  10_clean/` | ✓ | Cleaned prose — source of truth for text |
| `  15_structured/` | ✓ | entities / triples / discourse |
| `  25_hydrated/` | ✓ | Predicate-expanded facts |
| `  holdout/` + probes | ✓ | Eval carve-out |
| `data/_build/` | ✗ | Tokenized shards, train status, instruments |
| `data/_shared/` | ✓ | Cross-unit catalogs / replay |
| `configs/domains/`, `configs/templates/` | ✓ | Domain content |
| `scripts/` | ✓ | Pipeline + train + serve |
| `docker/` | ✓ | `conductor-trainer` image |
| `checkpoints/` | ✗ | CPT/SFT adapters |
| `.venv/` | ✗ | **The real dependency set** (see Setup) |
| `logs/phase_*.json` | ✓ | Campaign receipts |
| `vendor/venv-freeze.txt` | ✓ | Frozen venv package list (rebuild aid) |

**EDC path trap:** units are nested `city/year`. One-level globs
(`data/edc/*/splits`) match nothing and exit 0. Always use two levels:
`data/edc/*/*/…`.

Full lifecycle table: [`docs/ORGANIZATION.md`](docs/ORGANIZATION.md).

---

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
# Prefer restoring the proven venv over requirements.txt alone — see below.
pip install -r requirements.txt   # install pin only; NOT sufficient to train
```

### `requirements.txt` vs the baked venv

| | `requirements.txt` | Host `.venv` (truth) |
|---|---|---|
| Purpose | Reviewable install pin + Dependabot surface | What actually trains |
| torch | `2.13.0+cu130` (GHSA-rrmf-rvhw-rf47) | Must match or be deliberately rebuilt |
| quartet2 | **not on any index** | scikit-build CUDA extension from quartet-ii tree |
| torchao / flashinfer | may be incomplete | required by `train_cpt.py` |

**The cloud image bakes `.venv` whole** for this reason (`docs/cloud-trainer-image.md`).
`requirements.txt` alone **cannot** reproduce a trainable environment. Treat a published
`conductor-trainer` image as the artifact of record for cloud runs.

Extra index for CUDA 13 wheels:

```text
--extra-index-url https://download.pytorch.org/whl/cu130
```

`CUDA_HOME` must be a **cu13** toolkit (e.g. `/usr/local/cuda-13.3`) whenever flashinfer/Quartet JIT.

---

## Day-to-day commands

```bash
# Always
export PYTHONPATH=scripts
. .venv/bin/activate

# Full engine suite (no pytest — plain assert + __main__)
bash scripts/run_tests.sh

# Domain preflight
python scripts/run_pipeline.py --domain edc --check

# Local training console (stdlib HTTP, no GPU)
python scripts/serve.py --port 8800   # http://127.0.0.1:8800

# CPT train (example — use real configs under configs/)
# Prefer containment: systemd-run --user -p MemoryHigh=13G -p MemoryMax=14G …
```

**Tests are not pytest.** New tests mirror `scripts/pipeline/test_*_general.py`.

LLM provider resolution (plan A Google AI Studio, plan B OpenRouter):
`$GOOGLE_AI_API_KEY` / `$GEMINI_API_KEY` / libsecret `service=googleai`, else OpenRouter.
Force: `$LLM_PROVIDER=googleai|openrouter`.

---

## Cloud training (`conductor-trainer`)

Conductor's cloud path SSHes into a rented GPU, uploads a bundle zip, runs:

```text
/opt/qgre/bin/run-cloud-training <bundlePath> <runDir>
```

Telemetry: `<runDir>/cpt_stages.json` + `<runDir>/metrics.jsonl`.  
Stages vocabulary: `PENDING | RUNNING | DONE | FAILED` (exactly **one** non-terminal stage at a
time — `stages_write` collapses prior stages to `DONE`; RunTimeline rejects two `FAILED`s).

### Image variants

| Tag / form | Contents |
|---|---|
| Base `conductor-trainer` | Engine tree + baked `.venv` + toolchain (gcc/g++/c++, python3.13-dev, ninja, openssh) |
| Smoke `conductor-trainer:smoke` | Base + **baked int8 model** + **baked SSH host keys** (fingerprint in `/opt/qgre/SMOKE_HOST_KEY_FPR`) |

Smoke host keys are a **deliberate MITM tradeoff** for paid smoke only. Production needs a real
host-key channel (TOFU or provider API). See `docker/Dockerfile.smoke` header.

### Build

```bash
bash docker/build-cloud-trainer.sh                    # → conductor-trainer:dev
# smoke overlay (model + host keys):
docker build -f docker/Dockerfile.smoke \
  --build-arg BASE=conductor-trainer:dev \
  -t conductor-trainer:smoke \
  /home/marcos/Documents/dev/infra/models
```

Context uses `git ls-files` + full `.venv` copy. Script forces disk TMPDIR (not tmpfs `/tmp`).

### Toolchain gaps closed on paid smoke (do not regress)

| Gap | Failure mode | Fix |
|---|---|---|
| no `gcc`/`cc` | Triton JIT on Ampere | `gcc-14` + symlinks |
| no `Python.h` | Triton `cuda_utils.c` | `python3.13-dev` |
| no `ninja` | flashinfer SM120 cutlass JIT | `ninja-build` late layer |
| no `c++`/`g++` names | ninja link after nvcc | symlink `g++-14` → `g++`/`c++` |
| `post_install` stayed `PENDING` on train fail | incomplete stage doc | write `post_install FAILED` |

Harness (must stay green before pushing image logic):

```bash
QGRE_ENGINE_DIR="$PWD" PAYLOAD_MB="50" bash docker/tests/verdict-time-bound.sh
# LAW 1 verdict latency · LAW 2 evidence · LAW 3 train-fail closes post_install
```

### Publish (private registry)

```bash
docker tag conductor-trainer:smoke ghcr.io/torad-labs/conductor-trainer:smoke
docker push ghcr.io/torad-labs/conductor-trainer:smoke
# Prefer digest pin:
#   ghcr.io/torad-labs/conductor-trainer@sha256:<digest>
```

Vast authenticates the pull via `image_login` (conductor): env `QGRE_CLOUD_REGISTRY_LOGIN` or
libsecret `secret-tool lookup service vastai-registry`  
Shape: `-u <user> -p <read-only-token> ghcr.io`.

### Paid smoke (proven recipe)

Receipt: **`logs/phase_r18_cloud_smoke_rehearsal.json`** (attempt 25 PROVEN).

From **conductor-app** (not this repo), detached:

```bash
# Host key from the smoke image you pushed:
FPR=$(docker run --rm --entrypoint cat conductor-trainer:smoke /opt/qgre/SMOKE_HOST_KEY_FPR)

systemd-run --user --unit=qgre-paid-smoke --collect \
  -p WorkingDirectory=/path/to/conductor-app \
  -p MemoryHigh=13G -p MemoryMax=14G \
  -E QGRE_PAID_SMOKE=1 \
  -E QGRE_CLOUD_TRAINING_IMAGE=ghcr.io/torad-labs/conductor-trainer@sha256:<DIGEST> \
  -E QGRE_CLOUD_SSH_USER=root \
  -E QGRE_CLOUD_SSH_KEY_PATH=$HOME/.ssh/qgre_cloud_smoke \
  -E QGRE_CLOUD_SSH_HOST_KEY="$FPR" \
  -E QGRE_CLOUD_BUNDLE_FILES=/path/to/kandi-cpt/configs/train_smoke_cloud.yaml \
  -E QGRE_CLOUD_MIN_COMPUTE_CAP=1000 \
  bash -c './gradlew --no-daemon :conductor-agent:liveTest --tests "ai.torad.conductor.agent.provider.CloudSmokeProbe"; echo UNIT_EXIT=$?'
```

| Env | Role |
|---|---|
| `QGRE_PAID_SMOKE=1` | Arms the paid probe (else SKIP) |
| `QGRE_CLOUD_TRAINING_IMAGE` | **Digest-pinned** image ref |
| `QGRE_CLOUD_SSH_*` | root + key path + **baked smoke host-key fingerprint** |
| `QGRE_CLOUD_BUNDLE_FILES` | Smoke yaml (and any extra bundle files) |
| `QGRE_CLOUD_MIN_COMPUTE_CAP=1000` | Blackwell / real Quartet floor (Ampere hits Triton float8e4nv gap) |
| `QGRE_CLOUD_FORCE_OFFER_ID` | Optional pin to one marketplace offer |
| `PUBLIC_KEY` | Injected by conductor `createInstance` from `$QGRE_CLOUD_SSH_KEY_PATH.pub` |
| Vast API | `$VAST_API_KEY` or libsecret `service=vastai` |

**Success criteria:** `bundle/config/train/post_install` all `DONE`, `metrics.jsonl` ≥1 row
(smoke uses 20 steps), teardown `Stopped`, zero `conductor`-labeled instances left billing,
probe prints `PAID SMOKE PROVEN` / `UNIT_EXIT=0`.

**Ready window:** 30 minutes (cold pull of ~13GB image). Probe deadline 45 minutes.

**Local rehearsal** (no money): smoke container on loopback + `QGRE_SMOKE_LOCAL_PORT` — honest
no-GPU shape is `post_install FAILED` (not `DONE`).

Config: `configs/train_smoke_cloud.yaml` — `smoke_steps=20`, fp4, int8_master storage, init_from
int8 base path baked into the smoke image.

---

## Conductor integration (who owns what)

| Concern | Owner |
|---|---|
| Rent / SSH / bundle / poll / teardown | **conductor-app** (`StartCloudRun`, `CloudSmokeProbe`, sshj) |
| Train entrypoint + stages + metrics bridge | **this repo** (`docker/bin/run-cloud-training`) |
| Image contents + venv bake | **this repo** (`docker/`) |
| Offer filters (verified, directPort, VRAM, compute_cap, cuda_max_good) | conductor |
| Stage playbooks / agent loop | conductor `conductor-agent/knowledge/` |

Do not re-implement vast HTTP in this repo. Do not move training kernels into conductor.

---

## Security / CodeQL

| Topic | Policy |
|---|---|
| **`data/**`** | Scraped third-party HTML/JS — **not product code**. Excluded from CodeQL via `.github/codeql/codeql-config.yml` `paths-ignore` + advanced workflow `.github/workflows/codeql.yml`. |
| Default CodeQL setup | Disabled (no paths-ignore API). Advanced setup owns analysis. |
| `scripts/serve.py` | URL path/query → filesystem must go through `_safe_run_id` / `_under` / `_contained_file` (path-injection). |
| Registry | **Private** ghcr; read-only pull token in `vastai-registry` keyring. |
| Smoke host keys | Baked = MITM-able by anyone with the image; smoke-only. |

If CodeQL floods again on `data/`, check that default setup was not re-enabled and that
`paths-ignore` still lists `data/**`.

---

## Tests & gates

```bash
bash scripts/run_tests.sh                          # pipeline tests + bf16 gate + …
QGRE_ENGINE_DIR="$PWD" bash docker/tests/verdict-time-bound.sh   # cloud entrypoint laws
bash scripts/qgre/check_no_bf16.sh                 # precision floor
```

No pytest in the venv. Goldens under `scripts/pipeline/_golden/`.

---

## Key receipts (measured truth)

| Receipt | Meaning |
|---|---|
| `logs/phase_r18_cloud_smoke_rehearsal.json` | Local cloud rehearsal + **paid vast smoke PROVEN** (attempt ledger through 25) |
| `logs/phase_r17_gaps_batch_landing.json` | Gaps batch / image birth |
| `logs/OVERNIGHT_FINDINGS.md` | Historical corpus-scale findings (methodology since superseded by fp4 full-param) |

Prefer receipts over memory when resuming a campaign.

---

## Doc map

| Doc | Use when |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Agent law: precision, domains, GPU, conventions |
| [`docs/ORGANIZATION.md`](docs/ORGANIZATION.md) | Data layout / delete safety |
| [`docs/cloud-trainer-image.md`](docs/cloud-trainer-image.md) | Build/push/wire the cloud image |
| [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) | Stage semantics |
| [`docs/cpt_installation_best_practices.md`](docs/cpt_installation_best_practices.md) | Install-strength methodology |
| [`docs/KERNEL-ENGINEERING-PLAYBOOK.md`](docs/KERNEL-ENGINEERING-PLAYBOOK.md) | CUDA/kernel work |
| [`festival-factory/DE-FESTIVALIZATION.md`](festival-factory/DE-FESTIVALIZATION.md) | Festival vs generic boundary |
| [`configs/README.md`](configs/README.md) | Config surface |
| Conductor repo `CLAUDE.md` | Orchestrator + paid-smoke status from the app side |

---

## Commits & branches

- Engine commits are often **local on `main`** until the operator pushes; `main` is branch-protected
  (required checks) — feature PRs are normal.
- Do not force-push shared history. Do not leave orphan feature branches after squash-merge
  (delete remote branch when the PR lands).

---

## Known open follow-ups

- Ampere Triton **float8e4nv** pseudoquant path (smoke pins Blackwell `compute_cap≥1000` instead).
- Production **host-key channel** (smoke still bakes keys).
- Runtime venv may lag `requirements.txt` torch pin until a deliberate rebuild + fp4 smoke.
- `build_sft.py` stage display names vs slugs (locked by byte-identity golden — operator decision).
