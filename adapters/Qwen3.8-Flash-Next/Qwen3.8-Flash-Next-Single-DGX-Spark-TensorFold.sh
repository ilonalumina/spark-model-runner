#!/usr/bin/env bash
# Adapter: Qwen3.8-Flash-Next on TensorFold, one DGX Spark (MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold).
#
# Single node: the head serves alone. The runner derives no fabric facts and queries no worker for this recipe, and
# the worker settings in the config are not required.
#
# The launcher is ./start.sh. Its settings come from the environment first, then ./.env (scripts/config.sh), so the
# identity is exported here and wins over the file. start.sh returns once /health answers and a smoke request has
# text, and exits non-zero on a failed start. scripts/prepare.sh (image build and checkpoint download) is never run
# from the unit: start.sh gets PREPARE=0, and adapter_prepare refuses a recipe whose prepared state differs from what
# is on the node now or whose checkpoint is missing. Run prepare.sh by hand first.

ADAPTER_TOPOLOGY=single
ADAPTER_PORT=8888
ADAPTER_MODEL_ID="Qwen3.8-Flash-Next"
ADAPTER_HEAD_CONTAINER="qwen38-flash-next-tf"

_identity() { # the identity this launch uses, for the recipe's scripts
    export CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER" PORT="$ADAPTER_PORT" HOST=0.0.0.0 SERVED_NAME="$ADAPTER_MODEL_ID"
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing" >&2; return 1; }
    [ -x "$RECIPE_DIR/stop.sh" ] || { echo "adapter: $RECIPE_DIR/stop.sh is missing" >&2; return 1; }
    [ -f "$RECIPE_DIR/scripts/config.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/config.sh is missing" >&2; return 1; }
    # what prepare.sh last left ready against what is there now (scripts/config.sh: prepared_state); read-only
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

# The recipe's stop for the recorded container: up to STOP_TIMEOUT (30 s) to shut down, then removed. The export wins
# over today's .env, so the recorded name is the one stopped.
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
