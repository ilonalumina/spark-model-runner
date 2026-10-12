#!/usr/bin/env bash
# Adapter: Qwen3.8-Flash-Next NVFP4, dual DGX Spark (MiaAI-Lab recipe).
#
# Named for the recipe it drives: adapters/<model>/<recipe>.sh mirrors
# recipes/<model>/<recipe>/, and the CLI sets RECIPE_DIR from that name.
#
# This recipe's contract is the reverse of GLM's. Its start.sh does
# `source .env` after reading the environment (".env beats the environment"
# in its own words). Caller exports cannot carry the fabric pins in; only
# WORKER_IFACE / WORKER_IB_HCA (and ABLIT / HF_TOKEN) are read afterwards.
# So adapter_prepare and adapter_start verify that .env already states the
# derived pins and refuse on mismatch, printing the exact lines. The recipe's files are never
# written by this layer.
#
# Its stop.sh likewise sources .env for WORKER_IP: it stops whatever today's
# file names, not what was launched. So adapter_stop removes the containers
# the launched record names, on the recorded worker, directly; the upstream
# stop.sh is run afterwards only when today's file still names the same
# worker (for parity of side effects), never as the thing that guarantees
# the stop. The CLI's verify_absent is the proof either way.
#
# Container name is hardcoded to vllm-fn on both nodes by start.sh/stop.sh.

ADAPTER_HEAD_CONTAINER="vllm-fn"
ADAPTER_WORKER_CONTAINER="vllm-fn"

# Read a KEY="value" / KEY=value line without sourcing the file (sourcing
# would let it clobber this shell the way it clobbers the launcher's).
_env_get() {
    sed -n "s/^[[:space:]]*$1=[\"']\{0,1\}\([^\"'#]*\)[\"']\{0,1\}.*/\1/p" "$RECIPE_DIR/.env" | head -1 | sed 's/[[:space:]]*$//'
}

# Side-effect free: identity, config presence and the fabric pins. Sets
# ADAPTER_PORT and ADAPTER_MODEL_ID from the file the launcher will read.
adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/.env" ]     || { echo "adapter: $RECIPE_DIR/.env is missing — cp .env.sample .env and set the cluster block" >&2; return 1; }
    ADAPTER_PORT="$(_env_get PORT)"
    ADAPTER_MODEL_ID="$(_env_get SERVED_MODEL_NAME)"
    [ -n "$ADAPTER_PORT" ]     || { echo "adapter: PORT is not set in $RECIPE_DIR/.env" >&2; return 1; }
    [ -n "$ADAPTER_MODEL_ID" ] || ADAPTER_MODEL_ID="qwen3.8-flash-next"   # the launcher's own default
    local v
    v="$(_env_get HEAD_IP)";   [ "$v" = "$HEAD_IP" ]   || { echo "adapter: .env HEAD_IP='$v' but the runner config says $HEAD_IP" >&2; return 1; }
    v="$(_env_get WORKER_IP)"; [ "$v" = "$WORKER_IP" ] || { echo "adapter: .env WORKER_IP='$v' but the runner config says $WORKER_IP" >&2; return 1; }
    # Fabric pins too, before the current serve is stopped (derived read-only when the caller has not yet).
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_IB:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _verify_env_pins
}

# Fabric-dependent verification against the derived facts, in preflight and again at launch.
_verify_env_pins() {
    local bad=0 v
    v="$(_env_get IFACE)";        [ "$v" = "$HEAD_IF" ]   || { echo "  IFACE=\"$HEAD_IF\"          (.env has '$v')" >&2; bad=1; }
    # NCCL_IB_HCA exact-match syntax: a leading '=' means this device only.
    v="$(_env_get IB_HCA)";       [ "$v" = "=$HEAD_IB" ]  || { echo "  IB_HCA=\"=$HEAD_IB\"         (.env has '$v')" >&2; bad=1; }
    v="$(_env_get IB_GID_INDEX)"; [ "$v" = "$HEAD_GIDX" ] || { echo "  IB_GID_INDEX=$HEAD_GIDX         (.env has '$v')" >&2; bad=1; }
    # One IB_GID_INDEX serves both ranks; refuse if the nodes disagree.
    [ "$HEAD_GIDX" = "$WORKER_GIDX" ] || { echo "  head GID $HEAD_GIDX != worker GID $WORKER_GIDX — this recipe cannot express that" >&2; bad=1; }
    if [ "$bad" -ne 0 ]; then
        echo "adapter: $RECIPE_DIR/.env does not match the derived fabric facts; set the lines above and relaunch" >&2
        return 1
    fi
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _verify_env_pins || return 1

    # The two per-node overrides the launcher does honour from the environment.
    export WORKER_IFACE="$WORKER_IF"
    export WORKER_IB_HCA="=$WORKER_IB"

    # --no-download: weights are already in the head's HF cache. The launcher
    # still rsyncs them to the worker on first launch and skips once present.
    # (--launch would skip that sync, which is wrong for a first launch.)
    ./start.sh --no-download
}

# Stop by recorded identity. ADAPTER_* and WORKER_SSH were loaded from the
# launched record by the CLI; ssh_bounded comes from fabric-facts.sh.
adapter_stop() {
    local rc=0
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh_bounded 30 "docker rm -f '$ADAPTER_WORKER_CONTAINER' >/dev/null 2>&1 || true" || rc=1
    # Upstream stop for parity of side effects, only if today's .env still
    # names the recorded worker; otherwise it would act on a different host.
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
