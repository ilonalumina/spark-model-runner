#!/usr/bin/env bash
# Adapter: DeepSeek-V4.1-Flash EXL3 2.0 bpw (MCG, Viterbi pack), vLLM TP=2, 2x DGX Spark (sfxnz).
#
# Named for the recipe it drives: adapters/<model>/<recipe>.sh mirrors
# recipes/<model>/<recipe>/, and the CLI sets RECIPE_DIR from that name.
#
# CONTRACT (upstream run.sh @ 3de9146; first written against 45d3303): no .env at all. Every knob is
# `${VAR:-default}` read from the environment, so caller exports win for every
# key — the GLM shape. Everything below is passed in from outside; the recipe's
# files are never written by this layer.
#
# One interface and one HCA name serve both ranks: run.sh forwards the head's
# IFACE and HCA to the worker verbatim. adapter_start refuses when the derived
# facts differ between the nodes, because this launcher cannot express that.
#
# GID index: run.sh passes no NCCL_IB_GID_INDEX to either rank and forwards no
# such key to the worker, so NCCL selects the GID itself. The derived GID
# indices are reported in the journal by the CLI but cannot be pinned here.
#
# Weights: each node reads its own Hub cache (no NFS). SKIP_DOWNLOAD=1 keeps the
# pack download out of the unit; adapter_prepare / adapter_start refuse when
# the pinned snapshot is missing on either node instead.

# Enforced on the launcher, not merely reported.
ADAPTER_PORT=8888
ADAPTER_MODEL_ID="deepseek-ai/DeepSeek-V4.1-Flash"   # the recipe's SERVED_NAME default
ADAPTER_HEAD_CONTAINER="dsv41-flash-exl3"            # run.sh uses one CONTAINER_NAME on both nodes
ADAPTER_WORKER_CONTAINER="dsv41-flash-exl3"
# The serve image run.sh defaults to at 3de9146: canonical-e14 (docker/Dockerfile; the round-35 default
# DSV41_P2B_COOP=2 needs its coop kernels, and the strict audit fails on e13 or e12).
ADAPTER_IMAGE="dsv41-flash-exl3-sm121:canonical-e14"
# The pack revision run.sh defaults to at 3de9146 (SNAPSHOT_SHA).
ADAPTER_REVISION="2.0bpw-mcg-viterbi-lmhead-mxfp8"
# The checkpoint run.sh defaults to (MODEL).
ADAPTER_CHECKPOINT="sfxnz/DeepSeek-V4.1-Flash-EXL3"

_contract() {
    # Node identity. run.sh reads HEAD_IP for --master-addr and WORKER_HOST as
    # the ssh destination for the worker rank and for stop.sh.
    export HEAD_IP
    export WORKER_HOST="$WORKER_SSH"
    export PORT="$ADAPTER_PORT"
    export SERVED_NAME="$ADAPTER_MODEL_ID"
    export CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER"
    export IMAGE="$ADAPTER_IMAGE"
    export ORCHESTRATE=auto
    export SKIP_DOWNLOAD=1
    export SNAPSHOT_SHA="$ADAPTER_REVISION"
    # The model and the cache: run.sh forwards MODEL and HF_CACHE to the worker verbatim (worker_config), so the
    # head's absolute cache path is the path on both nodes, and the worker check below uses exactly that.
    export MODEL="$ADAPTER_CHECKPOINT"
    export HF_CACHE="$HOME/.cache/huggingface"
    # Launcher-only early exits (RESOLVE_SNAPSHOT_ONLY / VALIDATE_ONLY / PREFLIGHT_ONLY) never reach a real start.
    unset RESOLVE_SNAPSHOT_ONLY VALIDATE_ONLY PREFLIGHT_ONLY
    unset ROLE
}

# The snapshot directory run.sh would resolve on this node, or non-zero.
# RESOLVE_SNAPSHOT_ONLY=1 exits before any guard, container or ssh.
_head_snapshot() {
    ( cd "$RECIPE_DIR" && _contract && RESOLVE_SNAPSHOT_ONLY=1 ./run.sh 2>/dev/null )
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/run.sh" ]  || { echo "adapter: $RECIPE_DIR/run.sh is missing or not executable" >&2; return 1; }
    [ -x "$RECIPE_DIR/stop.sh" ] || { echo "adapter: $RECIPE_DIR/stop.sh is missing or not executable" >&2; return 1; }
    local snap img="$ADAPTER_IMAGE"
    snap="$(_head_snapshot)" || { echo "adapter: pinned snapshot missing on the head — hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision $ADAPTER_REVISION" >&2; return 1; }
    docker image inspect "$img" >/dev/null 2>&1 \
        || { echo "adapter: image $img missing on the head — docker build -f docker/Dockerfile -t $img docker (see its README)" >&2; return 1; }
    printf 'adapter: head snapshot %s, image %s\n' "$snap" "$img"
}

# The worker's checks need the ssh path, so they run at start (after the CLI
# has derived the fabric), bounded like every other worker probe.
_worker_ready() {
    local img="$ADAPTER_IMAGE" rev hub
    rev="$ADAPTER_REVISION"
    hub="$HF_CACHE/hub/models--${MODEL//\//--}"
    ssh_bounded 20 "docker image inspect '$img' >/dev/null 2>&1" \
        || { echo "adapter: image $img missing on the worker ($WORKER_SSH)" >&2; return 1; }
    ssh_bounded 20 "h=$hub; { [ -f \$h/refs/$rev ] && [ -d \$h/snapshots/\$(tr -d '[:space:]' < \$h/refs/$rev) ]; } || [ -d \$h/snapshots/$rev ]" \
        || { echo "adapter: pinned snapshot $rev missing on the worker ($WORKER_SSH)" >&2; return 1; }
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    if [ "$HEAD_IF" != "$WORKER_IF" ] || [ "$HEAD_IB" != "$WORKER_IB" ]; then
        echo "adapter: head $HEAD_IF/$HEAD_IB and worker $WORKER_IF/$WORKER_IB differ — run.sh forwards one IFACE/HCA to both ranks and cannot express that" >&2
        return 1
    fi
    _contract
    _worker_ready || return 1
    export IFACE="$HEAD_IF"
    export HCA="$HEAD_IB"
    ./run.sh
}

# Stop by RECORDED identity (ADAPTER_* and WORKER_SSH loaded from the launched
# record by the CLI). run.sh takes caller exports for every key, so the
# upstream stop receives that same identity.
adapter_stop() {
    local rc=0
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh_bounded 30 "docker rm -f '$ADAPTER_WORKER_CONTAINER' >/dev/null 2>&1 || true" || rc=1
    ( cd "$RECIPE_DIR" && _contract && ./stop.sh ) \
        || echo "adapter: upstream stop.sh returned non-zero; containers were removed by recorded identity above" >&2
    return "$rc"
}

adapter_health() {
    curl -fsS -m 10 -o /dev/null "http://127.0.0.1:${ADAPTER_PORT}/health"
}

adapter_describe() {
    printf 'dir    : %s\n' "$RECIPE_DIR"
    printf 'model  : %s\n' "$ADAPTER_MODEL_ID"
    printf 'api    : http://127.0.0.1:%s/v1\n' "$ADAPTER_PORT"
    printf 'head   : %s\n' "$ADAPTER_HEAD_CONTAINER"
    printf 'worker : %s on %s\n' "$ADAPTER_WORKER_CONTAINER" "$WORKER_SSH"
}
