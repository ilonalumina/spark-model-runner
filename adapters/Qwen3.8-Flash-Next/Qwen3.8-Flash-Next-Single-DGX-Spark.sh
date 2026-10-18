#!/usr/bin/env bash
# Adapter: Qwen3.8-Flash-Next NVFP4, ONE DGX Spark (TP=1), head only (MiaAI-Lab).
#
# Named for the recipe it drives: adapters/<model>/<recipe>.sh mirrors
# recipes/<model>/<recipe>/, and the CLI sets RECIPE_DIR from that name.
#
# Single node: ADAPTER_TOPOLOGY=single — no fabric derivation, no worker query
# in the stop proof. The worker node stays idle while this recipe is selected.
#
# CONTRACT (upstream start.sh @ b7c5da1): environment > .env > default, but
# only for the keys it captures before `source .env` (the _CLI_* list and
# _ENV_SNAPSHOT_VARS). PORT is on that list, so the export below wins.
# TP1_CONTAINER_NAME and TP1_MODEL_ID are not: .env wins for them in start.sh,
# while stop.sh restores an inherited TP1_CONTAINER_NAME after .env. The
# container name is verified at prepare and passed explicitly at start and
# stop; the checkpoint is .env's TP1_MODEL_ID (the
# default Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 answers the Hub with an auth
# error), and start.sh's Step 1 refuses a checkpoint missing from the cache.
# V030 (the vLLM 0.30 lane) stays at its default, false.
#
# Not used: the recipe's own supervisor, maintenance and heartbeat user units
# (scripts/supervise.sh, systemd/). The model-runner unit is the only lifecycle
# authority; a second supervisor would relaunch what the unit stopped. The
# recipe's memory watchdog (files/memwatch.sh) is used as shipped: start.sh
# starts it beside the container and stop.sh ends it. When the watchdog stops
# the container, `model-runner status` reads DOWN and nothing relaunches it.

ADAPTER_TOPOLOGY=single
ADAPTER_PORT=8888
ADAPTER_HEAD_CONTAINER="vllm-fn-tp1"     # start.sh / stop.sh default for TP1_CONTAINER_NAME

# Effective values of the keys this adapter depends on, computed the way
# start.sh / stop.sh compute them: PORT captured from the environment before
# `source .env`, everything else from .env then the default.
_effective() {
    ( cd "$RECIPE_DIR" && env "$@" bash --noprofile --norc -c '
        set -euo pipefail
        _CLI_PORT="${PORT:-}"
        _SNAP_NAME="${SERVED_MODEL_NAME-}"; _SNAPSET_NAME="${SERVED_MODEL_NAME+set}"
        _SNAP_IMAGE="${IMAGE-}";           _SNAPSET_IMAGE="${IMAGE+set}"
        source ./.env >/dev/null 2>&1
        [ -n "$_SNAPSET_NAME" ]  && SERVED_MODEL_NAME="$_SNAP_NAME"
        [ -n "$_SNAPSET_IMAGE" ] && IMAGE="$_SNAP_IMAGE"
        set +u
        printf "PORT\t%s\n" "${_CLI_PORT:-${PORT:-8888}}"
        printf "CONTAINER\t%s\n" "${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
        printf "MODEL\t%s\n" "${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
        printf "IMAGE\t%s\n" "${IMAGE-}"' 2>/dev/null )
}

_load_effective() {
    local out k v rc=0
    out="$(_effective "$@")" || rc=$?
    [ "$rc" -eq 0 ] || return 3
    while IFS=$'\t' read -r k v; do
        [ -n "$k" ] && printf -v "_EFF_$k" '%s' "$v"
    done <<< "$out"
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing or not executable" >&2; return 1; }
    [ -x "$RECIPE_DIR/stop.sh" ]  || { echo "adapter: $RECIPE_DIR/stop.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/.env" ]     || { echo "adapter: $RECIPE_DIR/.env is missing — cp .env.sample .env" >&2; return 1; }
    _load_effective "PORT=$ADAPTER_PORT" || { echo "adapter: $RECIPE_DIR/.env does not source under the launcher's 'set -euo pipefail'" >&2; return 1; }
    [ "$_EFF_CONTAINER" = "$ADAPTER_HEAD_CONTAINER" ] \
        || { printf '  TP1_CONTAINER_NAME must be %s\t(launcher would use %s)\n' "'$ADAPTER_HEAD_CONTAINER'" "'$_EFF_CONTAINER'" >&2
             echo "adapter: $RECIPE_DIR/.env names another container; fix or delete that line — nothing was stopped" >&2; return 1; }
    [ -n "$_EFF_IMAGE" ] || { echo "adapter: IMAGE is not set in $RECIPE_DIR/.env (start.sh requires it)" >&2; return 1; }
    ADAPTER_MODEL_ID="$_EFF_MODEL"
}

# start.sh refuses while nvidia-smi lists a compute app, and sizes the GPU
# budget from MemAvailable. After a switch the previous recipe's containers are
# gone but the driver may still be releasing memory: wait for no compute apps,
# then MemAvailable stable within 256 MiB across two reads.
_settle_head() {
    local deadline=$(( $(date +%s) + ${1:-180} )) last=0 now
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
            sleep 2; continue
        fi
        now="$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)"
        if [ "$last" -gt 0 ] && [ "$now" -ge "$last" ] && [ $(( now - last )) -lt 256 ]; then
            return 0
        fi
        last="$now"; sleep 2
    done
    echo "adapter: GPU did not settle within ${1:-180}s" >&2
    return 1
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _settle_head 180 || return 1
    # the recorded name too: adapter_prepare checked that .env resolves to it
    export PORT="$ADAPTER_PORT" TP1_CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER"
    # start.sh resolves the checkpoint from the local HF cache and never
    # downloads; a missing one fails at its Step 1.
    ./start.sh
}

# Stop by RECORDED identity. stop.sh sources .env and then restores a non-empty
# inherited TP1_CONTAINER_NAME, so the recorded name is passed in explicitly:
# stop.sh then ends the watchdog for that name and stops it gracefully
# (SIGTERM, STOP_TIMEOUT) so vLLM can unlink its /dev/shm segments, whatever
# .env or the caller's environment says today.
adapter_stop() {
    ( cd "$RECIPE_DIR" && TP1_CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER" ./stop.sh ) || {
        echo "adapter: upstream stop.sh returned non-zero; removing $ADAPTER_HEAD_CONTAINER by recorded identity" >&2
        pkill -f "memwatch.sh $ADAPTER_HEAD_CONTAINER" 2>/dev/null || true
    }
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
}

adapter_health() {
    curl -fsS -m 10 -o /dev/null "http://127.0.0.1:${ADAPTER_PORT}/health"
}

adapter_describe() {
    printf 'dir    : %s\n' "$RECIPE_DIR"
    printf 'model  : %s\n' "${ADAPTER_MODEL_ID:-qwen3.8-flash-next}"
    printf 'api    : http://127.0.0.1:%s/v1\n' "$ADAPTER_PORT"
    printf 'head   : %s\n' "$ADAPTER_HEAD_CONTAINER"
}
