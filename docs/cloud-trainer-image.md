# conductor-trainer — the cloud training image

One page. Build it, publish it, point conductor at it.

The conductor cloud path has been fully implemented since E12-F3 (2026-07-03) and has never been
runnable, because the image its remote half assumes was never authored. Slice G was deliberately
deferred as "manual/gated, never part of the deterministic suite". This is that deferred slice.
**Nothing here is wired into any gate** — not `scripts/run_tests.sh`, not `plan_install.py`, not
`./gradlew`. Building this image is an operator action, on purpose.

## Build

```sh
bash docker/build-cloud-trainer.sh                    # tag defaults to conductor-trainer:dev
bash docker/build-cloud-trainer.sh myreg.io/qgre/conductor-trainer:2026-08-01
```

The script assembles a build context in a temp dir from two things and nothing else:

| what | how it is selected | size |
|---|---|---|
| the engine tree | `git ls-files` — tracked files only | 977 MB / 16,058 files |
| the engine venv | `.venv` copied whole | 6.8 GB |

Result: an ~8.5 GB image. **Untracked `data/` and all of `checkpoints/` are excluded** — the
context is `git ls-files`, so whatever is TRACKED under `data/` does ship: measured in the built
image, 14,686 files / 777 MB under `data/`. `checkpoints/` and `.git` are absent entirely.

The context lands under `$TMPDIR`, which the script defaults to `/var/tmp` (disk) rather than
inheriting `/tmp` (tmpfs on this host). ~7.8 GB of context in RAM on a box that runs at 41/59 GB
used with swap 45/47 full is a freeze, not a slow build. Set `TMPDIR` yourself to override; the
script prints the chosen path and its filesystem, and warns if it is tmpfs.

## Why the venv is baked instead of pip-installed

`requirements.txt` **cannot build a trainable image** and must not be used as the dependency spec:

* it declares 7 packages and pins `torch==2.10.0+cu130`; the venv the engine actually runs on has
  **153** distributions and `torch 2.12.0+cu130`;
* it names neither `torchao` nor `flashinfer`, both imported directly by `scripts/train_cpt.py`;
* decisively, the FP4 moat is **`quartet2`** (`train_cpt.py:257/:304/:325`, taken whenever
  `precision: fp4`). It is on no package index. Its installed metadata records
  `quartet2 @ file:///home/marcos/Documents/dev/Quartet-II/kernels`, **a path that no longer
  exists** — the source moved to `/home/marcos/Documents/dev/projects/quartet-ii/repo/kernels` —
  and it is a scikit-build-core/CMake CUDA extension, not a wheel.

So the working set is only reproducible by copying it. The cost of that choice, stated plainly:
**the image's dependencies are pinned by one machine's directory, not by a reviewable file.** If
that host venv is ever lost, this image is not rebuildable from either repo. Rebuilding it would
require the quartet-ii tree and a CUDA toolchain. Treat a published image as the artifact of record.

The one coupling that *is* now enforced is glibc: the baked extensions were compiled on the build
host, and the build refuses to produce an image whose base is older than they need. It measures the
requirement from the artifacts (the highest `GLIBC_2.x` symbol across the 555 baked shared objects
— currently 2.34) against the container's own glibc (currently 2.42), so a future host upgrade
fails the *build* instead of failing at `import torch` on a rented GPU.

No CUDA toolkit is installed via apt: the venv already carries one, including a compiler at
`.venv/lib/.../nvidia/cu13/bin/nvcc`, which is candidate #6 of `scripts/cuda_home.py`'s own
preference list and resolves in-container.

## Publish, then point conductor at it

```sh
docker tag conductor-trainer:dev myreg.io/qgre/conductor-trainer:2026-08-01
docker push myreg.io/qgre/conductor-trainer:2026-08-01
export QGRE_CLOUD_TRAINING_IMAGE=myreg.io/qgre/conductor-trainer:2026-08-01
```

`QGRE_CLOUD_TRAINING_IMAGE` is validated before use (`CloudLaunchSpec.kt:25-30`): it **must** carry
an explicit tag or a `sha256:` digest. A bare `conductor-trainer` is rejected. Prefer the digest —
the launcher accepts `name@sha256:<64 hex>` and that is the only form that pins what actually runs.

**Keep the registry PRIVATE.** The image carries the engine tree, the compiled `quartet2` fp4
kernels (the moat), and — in the smoke variant — the base model weights; a public registry hands
all of that to anyone. Vast pulls the image itself, and conductor authenticates that pull:
`createInstance` carries vast's `image_login` pass-through (the CLI's documented
`-u <user> -p <token> <server>` string) whenever a login is configured — env
`QGRE_CLOUD_REGISTRY_LOGIN`, else libsecret `secret-tool lookup service vastai-registry`. Use a
READ-ONLY registry token there (for ghcr: a token with only `read:packages`), never a
broad-scoped account token — the string lands in the rented machine's docker config. With no
login configured the body reverts to the five-field shape — `image`, `runtype: "ssh_direct"`,
`target_state`, `cancel_unavail`, `label` — which requires a public image and is only acceptable
for images carrying nothing proprietary. Either way there is **no `disk` field**, so the
provider's default disk allocation has to be big enough for the pull (8.54 GB base / 13.2 GB
smoke). Check it before the first paid launch. The canonical home is
`ghcr.io/torad-labs/conductor-trainer` (org-owned, private).

## What the bundle must contain

The host bundle is a **flat zip**. Entry names are full-matched against `[A-Za-z0-9._-]+`
(`CloudBundleTransfer.kt:21,:272-278`) and taken from each file's basename (`:258`), so **no
directory can travel in a bundle**. Host-side its contents are undefined —
`QGRE_CLOUD_BUNDLE_FILES` has no default anywhere. This image therefore defines the contract:

> The bundle **must contain exactly one `.yaml`/`.yml` entry**; that is the CPT config.
> Zero entries → the run is rejected. More than one → rejected. Ambiguity is never resolved by
> guessing, and there is no environment-variable override: `CloudRemoteTransport.kt:285` opens the
> session with no `setEnvVar` and this image's `sshd_config` allowlists only
> `LANG LC_* COLORTERM NO_COLOR`, so the host cannot set one. An error naming a variable the
> operator cannot set would be worse than no message.

Everything else in the bundle is unpacked beside the config, into `<runDir>/bundle/`, untouched.

## The two payloads that are NOT in the image — bake them, or do not launch

Both are directories, so neither can ride in the flat zip. Measured:

| payload | size | referenced by |
|---|---|---|
| base model `gemma-4-E2B-it-int8-v2` | **4.4 GB**, absolute path `/home/marcos/Documents/dev/infra/models/…` | `configs/train_edc_lv_2026.yaml:18` |
| tokenized corpus `corpus_all97_v2.arrow` | **175 MB** directory | `configs/train_edc_lv_2026.yaml:26` |

**There is exactly one reachable route today: bake them into the image** (add a `COPY` to the
Dockerfile; the image grows to ~13 GB, against a `createInstance` that sends no `disk` field).

Two routes that read like options and are not:

* *"tar them into the bundle and untar on the instance"* — the entrypoint never untars, so the
  config's declared paths are still missing when the inputs check runs, and it refuses in ~90 ms — measured,
  with a 64 MB `model.tar` in the bundle. It does not even land on disk: the refusal now precedes
  the extract, so `<runDir>/bundle/` holds the streamed config and nothing else. Making this real means staging and
  unpacking declared payload archives BEFORE the inputs check, which is a design change with a
  disk-consumption consequence (bundle + unpacked + untarred), not an operator option.
* *"fetch them on the instance before launching"* — there is no window.
  `CloudRunStart.launchOnTrackedMachine` (`CloudRunStart.kt:97-120`) is atomic: connect → upload →
  launch, with no operator step in between, and every non-`Started` branch calls
  `stopTrackedMachine`. The cloud tool surface is exactly eight tools (`start_cloud_run`,
  `stop_cloud_run`, `preview_cloud_cost`, `check_transfer`, `show_cloud_machine`,
  `show_cloud_cleanup`, `offer_cloud_sign_in`, `offer_own_cloud_account`) — no upload-only tool and
  no remote-exec tool.

Whichever you choose, the config in the bundle must name paths that exist on the instance. The
entrypoint stats them and refuses in well under a tenth of a second if they do not.

## The entrypoint contract

`/opt/qgre/bin/run-cloud-training <bundlePath> <runDir>` — the launcher backgrounds it, records
`$!`, and decides Started-vs-Failed on a `sleep 1 && kill -0` probe (`CloudRemoteLaunch.kt:163-170`).

Consequences the operator can rely on, each measured in this image:

* **The verdict is O(1) in bundle size.** Only three things happen before it: three `stat`s, a
  central-directory read (`unzip -Z1`, 1–3 ms at 3.2 GB), and streaming the single config entry
  (`unzip -p`, 2–3 ms at 3.2 GB). Measured verdict time over five reps on bundles whose config is
  the LAST entry: **400 MB 132/78/68/70/67 ms, 1600 MB 81/66/64/66/63 ms, 3200 MB 63/63/70/71/68 ms**
  — the 3.2 GB median (68 ms) is at or below the 400 MB median (70 ms). The full extract and the CRC test happen
  *after* the verdict, so a bad bundle cannot outlive the launcher's probe no matter how large its
  payload is. (This is a fix, not a feature: the first revision of this file ran `unzip -t` and
  `unzip -j` first, and a 300 MB bundle with an invalid config was reported **Started** — 1606 ms
  to the verdict — leaving a rented machine running.)
* **The admit/reject decision is the ENGINE's, not a copy of it.** preflight calls
  `CptConfig.from_yaml()` then `CptConfig.validate_runtime(world=1)` — the same two calls
  `train_cpt.py` makes at `:1248` and `:1256`. So a smoke run with no corpus is admitted (the
  engine gates `data.tokenized` behind `smoke_steps > 0`), and a resume config whose
  `data.init_from` checkpoint is absent, or a config with `run.save_dir` unset, is refused. The
  earlier revision hand-stat'd two paths and got both of those backwards.
* **Two checks are the entrypoint's own**, because the engine does not perform them:
  `model.model_path` existence, and `run.save_dir` resolving to the runDir itself (the launcher
  redirects our stdout into `<runDir>/train.log` while `train_cpt.py`'s `_start_status` appends to
  `<save_dir>/train.log`; the same file with two writers corrupts the stream the metrics bridge
  parses).
* **A bad run is rejected in well under a second** — measured 11–99 ms across an empty file, a
  non-zip, a bundle with no config, an ambiguous two-config bundle, an unparseable config, a
  config the engine's validator rejects, a config whose model is absent, a `save_dir` collision,
  and 300 MB / 3200 MB bundles carrying an invalid config. Each exits non-zero with a diagnostic on
  stderr (which the launcher has redirected into `<runDir>/train.log`) and leaves **no pid file**.
  You do not get an idle box billing by the hour.
* **A do-nothing trainer never leaves a success document — on evidence, not on timing.**
  `post_install DONE` requires exit 0 **and at least one metric row**; a validate-only run, a
  stubbed trainer, or an idle-and-exit records `post_install FAILED "… no install evidence"`.
  Separately, a supervisor whose whole lifetime fits inside the host's 1 s liveness probe records
  `train FAILED "… the launch was DISCARDED, no install ran"`, because the host removed the pid
  file and reported the launch Failed — but do NOT read the probe as the safety mechanism: after
  the O(1)-verdict reorder, pre-verdict bundle work can hold the supervisor alive for seconds a
  do-nothing trainer never lived, which is exactly why success is gated on evidence. (Measured:
  a stub trainer over a 150–300 MB stored payload cleared the probe on supervisor lifetime alone.)
* **Telemetry never blinks.** `<runDir>/metrics.jsonl` is regenerated at end of run into a temp
  file and `rename()`d into place, so a poll can never observe a zero-row file for a run that has
  already produced rows. (Measured with a tight reader across the whole run: 233–342 such
  observations per run before the fix, 0 after, over ~2.9 M polls.) `cpt_stages.json` already used
  that shape.
* **Stopping a run stops the trainer.** `cleanup` TERMs the recorded pid; that is the supervisor,
  which forwards TERM to the trainer's whole process group and escalates to KILL after 10 s.
  Measured on the real `supervisor → bash → .venv/bin/python` chain: zero orphaned trainer
  processes, zero orphaned bridge processes.
* **Telemetry exists because this image creates it.** The engine writes neither `metrics.jsonl` nor
  `cpt_stages.json` — grep the whole engine tree and you get zero hits — while
  `CloudRemoteTelemetry.kt:134-148` reads exactly those two files. The entrypoint maintains
  `<runDir>/cpt_stages.json` (stages `bundle → config → train → post_install`) and bridges the
  engine's own step lines into `<runDir>/metrics.jsonl`. That bridge parses a *human log line*
  (`train_cpt.py:1527`), which is why `scripts/pipeline/test_metrics_line_format.py` now pins its
  shape in the engine's deterministic suite: a cosmetic edit there used to empty the operator's
  loss chart with nothing going red anywhere.

`QGRE_CLOUD_TRAINER_CMD` overrides the trainer command. It is a **test hook** for proving
supervision without a GPU, and it is not reachable from the host (the ssh transport carries no
environment); leave it unset and the trainer is `scripts/train_fp4.sh`, the engine's canonical
launcher.

## What this image was NOT able to prove locally — read before the first paid run

* **sshd is installed and serves port 22, but nothing here proves vast's `ssh_direct` plumbing is
  satisfied by it.** `CloudProvisioner.kt:121` sends `runtype: "ssh_direct"` and every later
  operation rides sshj. If the first smoke test fails, this is the first place to look. The image
  generates host keys at first boot and appends `$PUBLIC_KEY` to `root`'s `authorized_keys` —
  **and that variable name is INVENTED.** It came from general knowledge of how vast injects keys,
  from no code in either repo; nothing read here specifies the injection mechanism. Vast also
  commonly overrides a container's command, in which case this image's `CMD` never runs and sshd
  never starts. Both are for the operator's paid smoke test to arbitrate.
* **No training step has ever executed from this image.** Without a GPU the run reaches
  `train_cpt.py:1264 torch.cuda.set_device` and stops at *"Found no NVIDIA driver on your system"* —
  after the engine has printed its full RUN CONTRACT (`precision: fp4`, `int8_master: storage`,
  `optimizer_lowbit: 4`) and frozen run provenance. Everything up to device acquisition is proven;
  the first real step is the operator's smoke test.
* **`quartet2` cannot be imported in a CPU-only container.** Its `.so` needs `libcuda.so.1`, the
  NVIDIA *driver* library, which no image can ship — the container runtime injects it. Verified
  instead: the only unresolved links are `libcuda.so.1` (driver) and `libcudart.so.13` (present in
  the venv, resolved at runtime). On the host, with a driver, the same chain imports 8/8.

## Four engine tests are red inside this image, and should be

`scripts/run_tests.sh` reads **62 PASS / 0 FAIL on the host** and **58 PASS / 4 FAIL in the
container** — 62 entries either way. All four are absent-artifact failures, not image defects:

| red in container | why | flips green if you supply… |
|---|---|---|
| `check_artifact_precision.sh` | no weights in the image at all, so it reports *"resolved ZERO artifacts — refusing to pass vacuously"*. This gate is fail-closed by design; a green here would be the bug. | **the base model alone** — measured rc=0, *"1 artifact(s) scanned"*. It does not need `checkpoints/`; it needs any artifact. |
| `test_selftest_domain_e2e.py` | asserts `--check` is green on a filled fixture; the only failing check is `model_path`. | **the base model** — measured rc=0. |
| `test_plan_install.py` | **two** required inputs are missing, not one. With the base model bind-mounted it is STILL red, on `configs.cpt data.tokenized does not exist: …/pokemon_lore_kanto_v1.arrow`. | the base model **and** the tokenized corpus. |
| `test_execution_contract.py` | asserts a non-empty `engine.commit`; `.git` is not in the image. The commit is recorded at `/opt/qgre/ENGINE_COMMIT` instead. | nothing — `.git` is deliberately excluded. |

Measured by bind-mounting the 4.4 GB model read-only at its config-declared absolute path, no
`--gpus`. `test_cpt_config.py` (11/11) and `scripts/qgre/test_gates.py` (9/9) pass in the container
as they do on the host.

## Provenance

`/opt/qgre/ENGINE_COMMIT` records the engine commit and whether the tree was dirty at build time;
the same values are OCI labels (`org.opencontainers.image.revision`, `ai.torad.qgre.engine.dirty`).
`build-cloud-trainer.sh` refuses to tag a dirty tree as a release unless you pass `--allow-dirty`.
