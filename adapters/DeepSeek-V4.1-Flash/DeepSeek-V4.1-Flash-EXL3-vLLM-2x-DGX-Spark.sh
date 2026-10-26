#!/usr/bin/env bash
# Adapter: DeepSeek-V4.1-Flash EXL3 2.0 bpw, vLLM TP=2, 2x DGX Spark (sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark).
# run.sh takes every setting from the environment. It uses one interface and HCA name on both nodes, so they must match.
# Download the pinned snapshot and build the image on both nodes before the first launch.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID="deepseek-ai/DeepSeek-V4.1-Flash"
ADAPTER_HEAD_CONTAINER="dsv41-flash-exl3"
ADAPTER_WORKER_CONTAINER="dsv41-flash-exl3"
ADAPTER_IMAGE="dsv41-flash-exl3-sm121:canonical-e14"
ADAPTER_REVISION="2.0bpw-mcg-viterbi-lmhead-mxfp8"
ADAPTER_CHECKPOINT="sfxnz/DeepSeek-V4.1-Flash-EXL3"

_contract() {
    export HEAD_IP
    export WORKER_HOST="$WORKER_SSH"
    export PORT="$ADAPTER_PORT"
    export SERVED_NAME="$ADAPTER_MODEL_ID"
    export CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER"
    export IMAGE="$ADAPTER_IMAGE"
    export ORCHESTRATE=auto
    export SKIP_DOWNLOAD=1
    export SNAPSHOT_SHA="$ADAPTER_REVISION"
    export MODEL="$ADAPTER_CHECKPOINT"
    # run.sh passes this path unchanged to the worker.
    export HF_CACHE="$HOME/.cache/huggingface"
    unset RESOLVE_SNAPSHOT_ONLY VALIDATE_ONLY PREFLIGHT_ONLY
    unset ROLE
}

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
