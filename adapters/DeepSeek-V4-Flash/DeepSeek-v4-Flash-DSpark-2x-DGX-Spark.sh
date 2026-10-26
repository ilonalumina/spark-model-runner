#!/usr/bin/env bash
# Adapter: DeepSeek-V4-Flash Vision-Exp with DSpark speculative decoding, 2x DGX Spark (MiaAI-Lab recipe, Docker Compose).
# .env.dspark overrides the environment and is copied to the worker, so its node and fabric lines are checked, not exported.
# The worker needs the recipe at the same path, or WORKER_DIR set in .env.dspark.
# Set DSPARK_RESTART_POLICY=no so Docker does not restart the ranks on its own.

ENV_FILE_NAME=".env.dspark"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
ADAPTER_HEAD_CONTAINER="${PROJECT_NAME}-vllm-dspark-1"
ADAPTER_WORKER_CONTAINER="${PROJECT_NAME}-vllm-dspark-1"

_env_get() {
    sed -n "s/^[[:space:]]*$1=[\"']\{0,1\}\([^\"'#]*\)[\"']\{0,1\}.*/\1/p" "$RECIPE_DIR/$ENV_FILE_NAME" | head -1 | sed 's/[[:space:]]*$//'
}
_host_part() { printf '%s' "${1##*@}"; }
_dev_part()  { printf '%s' "${1#=}"; }

adapter_prepare() {
    [ -x "$RECIPE_DIR/start-deepseek-v4-flash-dspark.sh" ] || { echo "adapter: $RECIPE_DIR/start-deepseek-v4-flash-dspark.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/$ENV_FILE_NAME" ] || { echo "adapter: $RECIPE_DIR/$ENV_FILE_NAME is missing — cp .env.dspark.example .env.dspark and set the node block" >&2; return 1; }
    ADAPTER_PORT="$(_env_get VLLM_PORT)"
    ADAPTER_MODEL_ID="$(_env_get SERVED_MODEL_NAME | awk '{print $1}')"
    : "${ADAPTER_PORT:=8888}"
    : "${ADAPTER_MODEL_ID:=deepseek-v4-flash-vision-exp}"
    local bad=0 v
    v="$(_env_get MASTER_ADDR)";         [ "$v" = "$HEAD_IP" ]                   || { echo "  MASTER_ADDR=$HEAD_IP            ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    v="$(_env_get VLLM_HOST_IP)";        [ "$v" = "$HEAD_IP" ]                   || { echo "  VLLM_HOST_IP=$HEAD_IP           ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    v="$(_env_get WORKER_VLLM_HOST_IP)"; [ "$v" = "$WORKER_IP" ]                 || { echo "  WORKER_VLLM_HOST_IP=$WORKER_IP  ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    v="$(_env_get WORKER_HOST)";         [ "$(_host_part "$v")" = "$WORKER_IP" ] || { echo "  WORKER_HOST=$WORKER_IP          ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    if [ "$bad" -ne 0 ]; then
        echo "adapter: $RECIPE_DIR/$ENV_FILE_NAME node identity does not match the runner config; set the lines above" >&2
        return 1
    fi
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_IB:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _verify_env_pins
}

_verify_env_pins() {
    local bad=0 v
    v="$(_env_get NCCL_SOCKET_IFNAME)"; [ "$v" = "$HEAD_IF" ]                || { echo "  NCCL_SOCKET_IFNAME=$HEAD_IF     ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    v="$(_env_get TP_SOCKET_IFNAME)";   [ "$v" = "$HEAD_IF" ]                || { echo "  TP_SOCKET_IFNAME=$HEAD_IF       ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    v="$(_env_get GLOO_SOCKET_IFNAME)"; [ "$v" = "$HEAD_IF" ]                || { echo "  GLOO_SOCKET_IFNAME=$HEAD_IF     ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    v="$(_env_get NCCL_IB_HCA)";        [ "$(_dev_part "$v")" = "$HEAD_IB" ] || { echo "  NCCL_IB_HCA=$HEAD_IB            ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    v="$(_env_get WORKER_NCCL_SOCKET_IFNAME)"; [ -z "$v" ] || [ "$v" = "$WORKER_IF" ]               || { echo "  WORKER_NCCL_SOCKET_IFNAME=$WORKER_IF ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    v="$(_env_get WORKER_NCCL_IB_HCA)";        [ -z "$v" ] || [ "$(_dev_part "$v")" = "$WORKER_IB" ] || { echo "  WORKER_NCCL_IB_HCA=$WORKER_IB ($ENV_FILE_NAME has '$v')" >&2; bad=1; }
    [ "$HEAD_IF" = "$WORKER_IF" ] || [ -n "$(_env_get WORKER_NCCL_SOCKET_IFNAME)" ] || { echo "  WORKER_NCCL_SOCKET_IFNAME=$WORKER_IF  (worker interface differs from head; set it)" >&2; bad=1; }
    if [ "$bad" -ne 0 ]; then
        echo "adapter: $RECIPE_DIR/$ENV_FILE_NAME does not match the derived fabric facts; set the lines above and relaunch" >&2
        return 1
    fi
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _verify_env_pins || return 1
    local v
    v="$(_env_get DSPARK_RESTART_POLICY)"
    if [ "$v" != "no" ]; then
        echo "adapter: note — DSPARK_RESTART_POLICY is '${v:-unless-stopped}'; dockerd will restore the ranks on reboot on its own, alongside the runner. Set DSPARK_RESTART_POLICY=no in $ENV_FILE_NAME if the runner is to be the only thing that starts this serve." >&2
    fi
    local rc=0
    ./start-deepseek-v4-flash-dspark.sh || rc=$?
    # Exit 3: the ranks are already running.
    if [ "$rc" -eq 3 ]; then
        if adapter_health; then
            echo "adapter: launcher reported ranks already running and /health answers — treating as up" >&2
            return 0
        fi
        echo "adapter: launcher reported ranks already running but /health does not answer" >&2
        return 3
    fi
    return "$rc"
}

# The recipe's stop script reads the worker from today's .env.dspark, so containers are removed by recorded name first.
adapter_stop() {
    local rc=0 project="${ADAPTER_HEAD_CONTAINER%-vllm-dspark-1}"
    docker rm -f "$ADAPTER_HEAD_CONTAINER" "${project}-vl-sidecar-1" >/dev/null 2>&1 || true
    ssh_bounded 30 "docker rm -f '$ADAPTER_WORKER_CONTAINER' '${project}-vl-sidecar-1' >/dev/null 2>&1 || true" || rc=1
    if [ -f "$RECIPE_DIR/$ENV_FILE_NAME" ] && [ "$(_host_part "$(_env_get WORKER_HOST)")" = "${WORKER_SSH##*@}" ]; then
        ( cd "$RECIPE_DIR" && PROJECT_NAME="$project" ./stop-deepseek-v4-flash-dspark.sh ) || true
    else
        echo "adapter: today's $ENV_FILE_NAME names a different worker than the launched record; upstream stop skipped, containers removed by recorded identity" >&2
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
