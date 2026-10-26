#!/usr/bin/env bash
# Adapter: Qwen3.8-Flash-Next NVFP4, 2x DGX Spark (MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks).
# start.sh lets .env override the environment, so .env must already state the derived fabric settings; the adapter
# checks them and prints the lines to set.

ADAPTER_HEAD_CONTAINER="vllm-fn"
ADAPTER_WORKER_CONTAINER="vllm-fn"

# Reads one key from .env without sourcing it.
_env_get() {
    sed -n "s/^[[:space:]]*$1=[\"']\{0,1\}\([^\"'#]*\)[\"']\{0,1\}.*/\1/p" "$RECIPE_DIR/.env" | head -1 | sed 's/[[:space:]]*$//'
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/.env" ]     || { echo "adapter: $RECIPE_DIR/.env is missing — cp .env.sample .env and set the cluster block" >&2; return 1; }
    ADAPTER_PORT="$(_env_get PORT)"
    ADAPTER_MODEL_ID="$(_env_get SERVED_MODEL_NAME)"
    [ -n "$ADAPTER_PORT" ]     || { echo "adapter: PORT is not set in $RECIPE_DIR/.env" >&2; return 1; }
    [ -n "$ADAPTER_MODEL_ID" ] || ADAPTER_MODEL_ID="qwen3.8-flash-next"
    local v
    v="$(_env_get HEAD_IP)";   [ "$v" = "$HEAD_IP" ]   || { echo "adapter: .env HEAD_IP='$v' but the runner config says $HEAD_IP" >&2; return 1; }
    v="$(_env_get WORKER_IP)"; [ "$v" = "$WORKER_IP" ] || { echo "adapter: .env WORKER_IP='$v' but the runner config says $WORKER_IP" >&2; return 1; }
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_IB:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _verify_env_pins
}

_verify_env_pins() {
    local bad=0 v
    v="$(_env_get IFACE)";        [ "$v" = "$HEAD_IF" ]   || { echo "  IFACE=\"$HEAD_IF\"          (.env has '$v')" >&2; bad=1; }
    # A leading '=' makes NCCL match this device exactly.
    v="$(_env_get IB_HCA)";       [ "$v" = "=$HEAD_IB" ]  || { echo "  IB_HCA=\"=$HEAD_IB\"         (.env has '$v')" >&2; bad=1; }
    v="$(_env_get IB_GID_INDEX)"; [ "$v" = "$HEAD_GIDX" ] || { echo "  IB_GID_INDEX=$HEAD_GIDX         (.env has '$v')" >&2; bad=1; }
    [ "$HEAD_GIDX" = "$WORKER_GIDX" ] || { echo "  head GID $HEAD_GIDX != worker GID $WORKER_GIDX — this recipe cannot express that" >&2; bad=1; }
    if [ "$bad" -ne 0 ]; then
        echo "adapter: $RECIPE_DIR/.env does not match the derived fabric facts; set the lines above and relaunch" >&2
        return 1
    fi
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _verify_env_pins || return 1
    # The only fabric settings start.sh reads from the environment.
    export WORKER_IFACE="$WORKER_IF"
    export WORKER_IB_HCA="=$WORKER_IB"
    # --no-download still copies the weights to the worker on first launch; --launch would not.
    ./start.sh --no-download
}

# stop.sh reads the worker from today's .env, so the recorded containers are removed first.
adapter_stop() {
    local rc=0
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh_bounded 30 "docker rm -f '$ADAPTER_WORKER_CONTAINER' >/dev/null 2>&1 || true" || rc=1
    if [ -f "$RECIPE_DIR/.env" ] && [ "$(_env_get WORKER_IP)" = "${WORKER_SSH##*@}" ]; then
        ( cd "$RECIPE_DIR" && ./stop.sh ) || true
    else
        echo "adapter: today's .env names a different worker than the launched record; upstream stop.sh skipped, containers removed by recorded identity" >&2
    fi
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
