#!/usr/bin/env bash
# Build the conductor-trainer cloud image. OPERATOR-INVOKED ONLY.
#
# This is E12-F3 slice G, recorded in the conductor backlog as "deliberately deferred as
# manual/gated per its own investigation - optional packaging, never part of the deterministic
# suite". Do NOT call this from scripts/run_tests.sh, plan_install.py, any gradle task, or any
# gate. Nothing in the deterministic suite may depend on docker.
#
#   bash docker/build-cloud-trainer.sh [TAG] [--allow-dirty]
#
# The context is assembled in a temp dir from exactly two inputs, which is the whole reason this
# script exists rather than a bare `docker build` at the repo root: a root-context build would
# stream data/ (18G) and checkpoints/ (96G) to the daemon.
#
#   tracked engine tree   git ls-files      977M / 16,058 files
#   the working venv      .venv             6.8G   (see docs/cloud-trainer-image.md for WHY it is
#                                                   copied rather than pip-installed: quartet2,
#                                                   the fp4 moat, is on no index and its recorded
#                                                   source path no longer exists)
#
# The glibc pairing between that copied venv and the Dockerfile's base image is asserted INSIDE
# the build (see "THE glibc WALL" in docker/Dockerfile), so it holds for a bare `docker build` too.
set -euo pipefail

TAG="conductor-trainer:dev"
ALLOW_DIRTY=0
for arg in "$@"; do
    case "$arg" in
        --allow-dirty) ALLOW_DIRTY=1 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) TAG="$arg" ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v docker >/dev/null 2>&1 || { echo "docker is not installed" >&2; exit 127; }
[ -d .venv ] || { echo "no .venv at $REPO_ROOT — the image bakes it; nothing to build" >&2; exit 1; }

COMMIT="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
    DIRTY=true
    if [ "$ALLOW_DIRTY" -eq 0 ]; then
        echo "refusing to build from a dirty tree; commit first or pass --allow-dirty" >&2
        exit 1
    fi
else
    DIRTY=false
fi

# THE CONTEXT MUST NOT LAND IN tmpfs. `mktemp -d` inherits TMPDIR, which is /tmp on this host — a
# 30G tmpfs, i.e. RAM. The context is ~7.8GB (a 6.8G venv copy + the 978M tracked tree), and this
# box runs at 41/59GB used with swap 45/47 full; assembling it in tmpfs is a freeze, not a slow
# build. /var/tmp is on the 1.8T disk. An operator who has deliberately set TMPDIR keeps it.
: "${TMPDIR:=/var/tmp}"
export TMPDIR

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

CTX_FSTYPE="$(df --output=fstype "$CTX" 2>/dev/null | tail -1 || echo unknown)"
echo "assembling context in $CTX  (TMPDIR=$TMPDIR, filesystem=$CTX_FSTYPE, $(df -h --output=avail "$CTX" 2>/dev/null | tail -1 | tr -d ' ') free)"
if [ "$CTX_FSTYPE" = tmpfs ]; then
    echo "  WARNING: the build context is on tmpfs (RAM). ~7.8GB of context will be held in memory." >&2
    echo "           Set TMPDIR to a disk-backed path (e.g. TMPDIR=/var/tmp) before rebuilding." >&2
fi

mkdir -p "$CTX/engine"
git ls-files -z | tar --null -T - -cf - | (cd "$CTX/engine" && tar -xf -)
echo "  tracked files: $(find "$CTX/engine" -type f | wc -l)"
cp -a .venv "$CTX/engine/.venv"
cp "$REPO_ROOT/docker/Dockerfile" "$CTX/Dockerfile"
cp -a "$REPO_ROOT/docker/bin" "$CTX/bin"
echo "  context size: $(du -sh "$CTX" | cut -f1)"

docker build \
    --build-arg ENGINE_COMMIT="$COMMIT" \
    --build-arg ENGINE_DIRTY="$DIRTY" \
    -t "$TAG" \
    "$CTX"

echo
echo "built $TAG  (engine $COMMIT, dirty=$DIRTY)"
echo "next: docker push, then export QGRE_CLOUD_TRAINING_IMAGE=<repo>@sha256:<digest>"
echo "      the launcher REQUIRES an explicit tag or digest (CloudLaunchSpec.kt:25-30)"
