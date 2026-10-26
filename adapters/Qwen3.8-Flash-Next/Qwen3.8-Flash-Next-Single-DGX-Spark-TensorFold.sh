#!/usr/bin/env bash
# Adapter: Qwen3.8-Flash-Next on TensorFold, one DGX Spark (MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold).
# Run scripts/prepare.sh by hand before the first launch.

ADAPTER_TOPOLOGY=single
ADAPTER_PORT=8888
ADAPTER_MODEL_ID="Qwen3.8-Flash-Next"
ADAPTER_HEAD_CONTAINER="qwen38-flash-next-tf"

# The recipe reads the environment before .env, so these exports take precedence.
_identity() {
    export CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER" PORT="$ADAPTER_PORT" HOST=0.0.0.0 SERVED_NAME="$ADAPTER_MODEL_ID"
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing" >&2; return 1; }
    [ -x "$RECIPE_DIR/stop.sh" ] || { echo "adapter: $RECIPE_DIR/stop.sh is missing" >&2; return 1; }
    [ -f "$RECIPE_DIR/scripts/config.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/config.sh is missing" >&2; return 1; }
    local now was snap
    now="$( cd "$RECIPE_DIR" && _identity && source ./scripts/config.sh && prepared_state )" \
        || { echo "adapter: could not read the recipe's prepared state" >&2; return 1; }
    was="$( cd "$RECIPE_DIR" && _identity && source ./scripts/config.sh && cat "$PREPARED_MARKER" 2>/dev/null )" || was=""
    if [ "$now" != "$was" ]; then
        echo "adapter: the image or checkpoint is not prepared for this configuration; run scripts/prepare.sh" >&2
        echo "adapter:   now: $now" >&2
        echo "adapter:   was: ${was:-(never prepared)}" >&2
        return 1
    fi
    snap="$( cd "$RECIPE_DIR" && _identity && source ./scripts/config.sh \
             && printf '%s/hub/models--%s/snapshots/%s' "$HF_CACHE" "${MODEL_ID//\//--}" "$(snapshot_rev)" )"
    [ -f "$snap/config.json" ] || { echo "adapter: no checkpoint at $snap; run scripts/prepare.sh" >&2; return 1; }
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _identity
    export PREPARE=0
    ./start.sh
}

adapter_stop() {
    ( cd "$RECIPE_DIR" && export CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER" PORT="$ADAPTER_PORT" && ./stop.sh ) \
        || echo "adapter: stop.sh returned non-zero; removing $ADAPTER_HEAD_CONTAINER" >&2
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
}

adapter_health() {
    curl -fsS -m 10 -o /dev/null "http://127.0.0.1:${ADAPTER_PORT}/health"
}

adapter_describe() {
    printf 'dir    : %s\n' "$RECIPE_DIR"
    printf 'model  : %s\n' "$ADAPTER_MODEL_ID"
    printf 'api    : http://127.0.0.1:%s/v1\n' "$ADAPTER_PORT"
    printf 'head   : %s\n' "$ADAPTER_HEAD_CONTAINER"
}
