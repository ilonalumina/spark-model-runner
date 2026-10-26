#!/usr/bin/env bash
# Adapter: DeepSeek-V4.1-Flash EXL3 2.9 bpw, 2x DGX Spark (MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks).
# start.sh lets .env override the environment, so the adapter computes the values the launcher will actually use
# and refuses when node identity or fabric settings disagree with the derived facts.

ADAPTER_HEAD_CONTAINER="dsv41-exl3-head"
ADAPTER_WORKER_CONTAINER="dsv41-exl3-worker"

_KEYS="PORT SERVED_MODEL_NAME HEAD_IP WORKER_IP WORKER_USER WORKER_SSH CONTAINER_HEAD CONTAINER_WORKER HEAD_CX7_IF HEAD_CX7_IB WORKER_CX7_IF WORKER_CX7_IB HEAD_GID WORKER_GID"

# _effective SEED=VALUE ...: prints "KEY<TAB>value" for each key in _KEYS as start.sh would resolve it.
_effective() {
    ( cd "$RECIPE_DIR" && env "$@" bash --noprofile --norc -c '
        set -euo pipefail
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
        set -a
        source "$SCRIPT_DIR/.env" >/dev/null 2>&1
        set +a
        set +u
        # Defaults from MiaAI-Lab start.sh, AGPL-3.0.
        PORT="${PORT:-8888}"
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-v4.1-Flash-EXL3}"
        HEAD_IP="${HEAD_IP:-10.0.0.1}"
        WORKER_IP="${WORKER_IP:-10.0.0.2}"
        WORKER_USER="${WORKER_USER:-$USER}"
        WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"
        CONTAINER_HEAD="${CONTAINER_HEAD:-dsv41-exl3-head}"
        CONTAINER_WORKER="${CONTAINER_WORKER:-dsv41-exl3-worker}"
        HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
        HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
        WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
        WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
        HEAD_GID="${HEAD_GID:-${NCCL_IB_GID_INDEX-}}"
        WORKER_GID="${WORKER_GID:-${NCCL_IB_GID_INDEX-}}"
        for k in '"$_KEYS"'; do printf "%s\t%s\n" "$k" "${!k}"; done' ./start.sh 2>/dev/null )
}

# Sets _EFF_<KEY> for each key; returns 3 if .env fails under start.sh's shell options.
_load_effective() {
    local out k v rc=0
    out="$(_effective "$@")" || rc=$?
    [ "$rc" -eq 0 ] || return 3
    while IFS=$'\t' read -r k v; do
        [ -n "$k" ] && printf -v "_EFF_$k" '%s' "$v"
    done <<< "$out"
}

_identity_seeds() {
    printf '%s\n' "HEAD_IP=$HEAD_IP" "WORKER_IP=$WORKER_IP" "WORKER_SSH=$WORKER_SSH" \
        "WORKER_USER=${WORKER_SSH%@*}" "CONTAINER_HEAD=$ADAPTER_HEAD_CONTAINER" "CONTAINER_WORKER=$ADAPTER_WORKER_CONTAINER"
}

# _require KEY=VALUE ...: prints each key whose effective value differs; returns 1 if any.
_require() {
    local bad=0 kv k want have
    for kv in "$@"; do
        k="${kv%%=*}"; want="${kv#*=}"
        eval "have=\"\${_EFF_$k-}\""
        if [ "$have" != "$want" ]; then
            printf '  %s must be %s\t(launcher would use %s)\n' "$k" "'$want'" "'$have'" >&2
            bad=1
        fi
    done
    return "$bad"
}

_unsourceable_msg() {
    echo "adapter: $RECIPE_DIR/.env does not source under the launcher's 'set -euo pipefail' with SCRIPT_DIR set (the launcher would fail the same way)" >&2
}

_fabric_seeds() {
    printf '%s\n' "HEAD_CX7_IF=$HEAD_IF" "HEAD_CX7_IB=$HEAD_IB" "WORKER_CX7_IF=$WORKER_IF" "WORKER_CX7_IB=$WORKER_IB" \
        "HEAD_GID=$HEAD_GIDX" "WORKER_GID=$WORKER_GIDX"
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/.env" ]     || { echo "adapter: $RECIPE_DIR/.env is missing — cp .env.example .env and set the worker block" >&2; return 1; }
    local seeds; mapfile -t seeds < <(_identity_seeds)
    _load_effective "${seeds[@]}" || { _unsourceable_msg; return 1; }
    if ! _require "${seeds[@]}"; then
        echo "adapter: $RECIPE_DIR/.env redirects identity keys away from the runner config / this adapter (lines above); fix or delete those lines — nothing was stopped" >&2
        return 1
    fi
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_IB:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    mapfile -t -O "${#seeds[@]}" seeds < <(_fabric_seeds)
    _load_effective "${seeds[@]}" || { _unsourceable_msg; return 1; }
    if ! _require "${seeds[@]}"; then
        echo "adapter: $RECIPE_DIR/.env disagrees with the derived fabric facts (lines above); set the lines to these values or delete them so the exports apply — nothing was stopped" >&2
        return 1
    fi
    ADAPTER_PORT="$_EFF_PORT"
    ADAPTER_MODEL_ID="$_EFF_SERVED_MODEL_NAME"
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    local seeds; mapfile -t seeds < <(_identity_seeds)
    mapfile -t -O "${#seeds[@]}" seeds < <(_fabric_seeds)
    _load_effective "${seeds[@]}" || { _unsourceable_msg; return 1; }
    if ! _require "${seeds[@]}"; then
        echo "adapter: $RECIPE_DIR/.env disagrees with the derived fabric facts or the identity (lines above); set the lines to these values or delete them so the exports apply, then relaunch" >&2
        return 1
    fi
    local kv; for kv in "${seeds[@]}"; do export "$kv"; done
    ./start.sh
}

# Removes the recorded containers, then runs `start.sh stop` only if .env still resolves to the same identity.
adapter_stop() {
    local rc=0
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh_bounded 30 "docker rm -f '$ADAPTER_WORKER_CONTAINER' >/dev/null 2>&1 || true" || rc=1
    if [ -f "$RECIPE_DIR/.env" ] && [ -x "$RECIPE_DIR/start.sh" ]; then
        local seeds; mapfile -t seeds < <(_identity_seeds)
        if _load_effective "${seeds[@]}" && _require "${seeds[@]}"; then
            ( cd "$RECIPE_DIR" && for kv in "${seeds[@]}"; do export "$kv"; done && ./start.sh stop ) \
                || echo "adapter: upstream stop returned non-zero; containers were removed by recorded identity above" >&2
        else
            echo "adapter: upstream stop skipped — .env no longer resolves to the launched identity; containers were removed by recorded identity above" >&2
        fi
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
