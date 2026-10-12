#!/usr/bin/env bash
# Adapter: DeepSeek-V4-Flash Vision-Exp, DSpark speculative decoding, 2x DGX
# Spark (MiaAI-Lab recipe). Compose-based launcher.
#
# Named for the recipe it drives: adapters/<model>/<recipe>.sh mirrors
# recipes/<model>/<recipe>/, and the CLI sets RECIPE_DIR from that name.
#
# Contract (read from the launcher, not assumed):
# - Config file is .env.dspark, sourced with `set -a`: the file wins over the
#   environment, and the launcher pushes that same file to the worker's
#   checkout. Nothing exported here reaches rank 1. adapter_prepare and
#   adapter_start verify the identity and interface/HCA lines against the
#   derived facts and refuse on mismatch. The file is never written by this layer.
# - GID index: the launcher validates RoCEv2 GIDs from sysfs on both ranks
#   itself (NCCL_IB_GID_AUTO=1 default) and leaves NCCL_IB_GID_INDEX unset. No
#   pin to verify here.
# - Containers are compose-named: ${PROJECT_NAME}-vllm-dspark-1 on both nodes
#   (plus an optional ${PROJECT_NAME}-vl-sidecar-1), PROJECT_NAME default
#   deepseek-v4-flash. Ranks carry restart: ${DSPARK_RESTART_POLICY:-unless-stopped}.
# - Exit 3 from the launcher = ranks already running (dockerd restored them
#   after a reboot); accepted here only if /health answers.
# - Its stop script re-sources .env.dspark for WORKER_HOST, so it stops what
#   today's file names. adapter_stop therefore removes the containers the
#   launched record names, on the recorded worker, directly, and runs the
#   upstream stop afterwards only when today's file still names that worker.
# - Worker checkout: REMOTE_WORKER_DIR defaults to this directory's path, so
#   the same recipes/... path has to exist on the worker, owned by the
#   operator account, or WORKER_DIR is set in .env.dspark.

ENV_FILE_NAME=".env.dspark"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
ADAPTER_HEAD_CONTAINER="${PROJECT_NAME}-vllm-dspark-1"
ADAPTER_WORKER_CONTAINER="${PROJECT_NAME}-vllm-dspark-1"

_env_get() {
    sed -n "s/^[[:space:]]*$1=[\"']\{0,1\}\([^\"'#]*\)[\"']\{0,1\}.*/\1/p" "$RECIPE_DIR/$ENV_FILE_NAME" | head -1 | sed 's/[[:space:]]*$//'
}
_host_part() { printf '%s' "${1##*@}"; }
_dev_part()  { printf '%s' "${1#=}"; }

# Side-effect free: identity + config presence + node identity and fabric lines.
adapter_prepare() {
    [ -x "$RECIPE_DIR/start-deepseek-v4-flash-dspark.sh" ] || { echo "adapter: $RECIPE_DIR/start-deepseek-v4-flash-dspark.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/$ENV_FILE_NAME" ] || { echo "adapter: $RECIPE_DIR/$ENV_FILE_NAME is missing — cp .env.dspark.example .env.dspark and set the node block" >&2; return 1; }
    ADAPTER_PORT="$(_env_get VLLM_PORT)"
    ADAPTER_MODEL_ID="$(_env_get SERVED_MODEL_NAME | awk '{print $1}')"   # first alias
    : "${ADAPTER_PORT:=8888}"                       # the launcher's own default chain
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
    # Fabric pins too, before the current serve is stopped (derived read-only when the caller has not yet).
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_IB:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _verify_env_pins
}

# Fabric-dependent verification against the derived facts, in preflight and again at launch.
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

# Stop by recorded identity (ADAPTER_* and WORKER_SSH as loaded by the CLI).
# The sidecar shares the project prefix of the recorded container name.
adapter_stop() {
    local rc=0 project="${ADAPTER_HEAD_CONTAINER%-vllm-dspark-1}"
    docker rm -f "$ADAPTER_HEAD_CONTAINER" "${project}-vl-sidecar-1" >/dev/null 2>&1 || true
    ssh_bounded 30 "docker rm -f '$ADAPTER_WORKER_CONTAINER' '${project}-vl-sidecar-1' >/dev/null 2>&1 || true" || rc=1
    # Upstream stop for compose project cleanup (networks, orphans, optional
    # NFS), only if today's file still names the recorded worker.
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
