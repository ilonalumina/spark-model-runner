#!/usr/bin/env bash
# Adapter: GLM-5.3-Flash EXL3, 2x DGX Spark (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).
# The launcher takes the environment over .env for every key, so node identity and fabric settings are exported.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID="GLM-5.3-Flash"
ADAPTER_HEAD_CONTAINER="glm53-exl3-head"
ADAPTER_WORKER_CONTAINER="glm53-exl3-worker"

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/.env" ]     || { echo "adapter: $RECIPE_DIR/.env is missing — cp .env.example .env first" >&2; return 1; }
}

_contract() {
    export HEAD_IP WORKER_IP WORKER_SSH
    export PORT="$ADAPTER_PORT"
    export SERVED_MODEL_NAME="$ADAPTER_MODEL_ID"
    export CONTAINER_HEAD="$ADAPTER_HEAD_CONTAINER"
    export CONTAINER_WORKER="$ADAPTER_WORKER_CONTAINER"
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _contract
    export HEAD_CX7_IF="$HEAD_IF"
    export HEAD_CX7_IB="$HEAD_IB"
    export HEAD_GID="$HEAD_GIDX"
    export WORKER_CX7_IF="$WORKER_IF"
    export WORKER_CX7_IB="$WORKER_IB"
    export WORKER_GID="$WORKER_GIDX"

    # Without BUILD=1 the launcher pulls the published image, which lags the checkout, over the local build.
    export BUILD=1
    export SKIP_DOWNLOAD=1
    export SKIP_SYNC=1

    ./start.sh
}

adapter_stop() {
    cd "$RECIPE_DIR" || return 1
    _contract
    ./start.sh stop
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
