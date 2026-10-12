#!/usr/bin/env bash
# Adapter: DeepSeek v4.1 Flash EXL3 2.9 bpw, 2x DGX Spark (MiaAI-Lab recipe).
#
# Named for the recipe it drives: adapters/<model>/<recipe>.sh mirrors
# recipes/<model>/<recipe>/, and the CLI sets RECIPE_DIR from that name.
#
# Contract (upstream start.sh, read through commit 8404ac7): `set -euo
# pipefail`, SCRIPT_DIR from its own path, a fixed list of tuning knobs
# captured, then `set -a; source "$SCRIPT_DIR/.env"; set +a`, then every key
# defaulted with ${VAR:-default}. A key assigned by .env therefore wins over
# the environment; a key it leaves alone falls to the environment, then to
# upstream's default. None of the identity or fabric keys are on the
# captured-knob list.
#
# This adapter never infers what .env "assigns". It computes the effective
# configuration the launcher will run with: a child bash in the recipe
# directory, seeded with the exports this adapter intends to make, with the
# same shell options and SCRIPT_DIR, sourcing .env and applying upstream's
# defaults, then printing the final values. Identity and fabric keys have to
# come out equal to the runner's facts or the adapter refuses (in
# adapter_prepare, before `use` stops anything); port and model id are
# recorded as they come out. A .env that fails under the launcher's options
# fails here too, distinctly.
#
# The recipe's files are never written by this layer.

ADAPTER_HEAD_CONTAINER="dsv41-exl3-head"
ADAPTER_WORKER_CONTAINER="dsv41-exl3-worker"

_KEYS="PORT SERVED_MODEL_NAME HEAD_IP WORKER_IP WORKER_USER WORKER_SSH CONTAINER_HEAD CONTAINER_WORKER HEAD_CX7_IF HEAD_CX7_IB WORKER_CX7_IF WORKER_CX7_IB HEAD_GID WORKER_GID"

# _effective SEED=VALUE ... — print "KEY<TAB>value" for every key in _KEYS as
# the launcher would see it. Exit 3 when .env does not source under the
# launcher's options. The child inherits this shell's environment plus the
# seeds, so `$USER`, runner exports and .env references resolve as at launch.
_effective() {
    ( cd "$RECIPE_DIR" && env "$@" bash --noprofile --norc -c '
        set -euo pipefail
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
        set -a
        # a plain statement: under errexit a failing command inside .env aborts
        # this child exactly as it aborts the launcher (a `|| exit` here would
        # suppress errexit inside the sourced file)
        source "$SCRIPT_DIR/.env" >/dev/null 2>&1
        set +a
        set +u
        # upstream defaults, copied from the recipe start.sh, MiaAI-Lab, AGPL-3.0,
        # so the computed configuration matches what the launcher runs with
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

# Any non-zero exit of the child (a failing command or an unbound reference in
# .env under the launcher's options) is "unsourceable" here.
_load_effective() {
    # fills the _EFF_<KEY> variables; returns 3 on an unsourceable .env
    local out k v rc=0
    out="$(_effective "$@")" || rc=$?
    [ "$rc" -eq 0 ] || return 3
    while IFS=$'\t' read -r k v; do
        [ -n "$k" ] && printf -v "_EFF_$k" '%s' "$v"
    done <<< "$out"
}

# The identity this adapter requires, as seed arguments. Recorded identity at
# stop (ADAPTER_* and WORKER_SSH loaded from the launched record), the runner
# config and this adapter's names at prepare/start.
_identity_seeds() {
    printf '%s\n' "HEAD_IP=$HEAD_IP" "WORKER_IP=$WORKER_IP" "WORKER_SSH=$WORKER_SSH" \
        "WORKER_USER=${WORKER_SSH%@*}" "CONTAINER_HEAD=$ADAPTER_HEAD_CONTAINER" "CONTAINER_WORKER=$ADAPTER_WORKER_CONTAINER"
}

# Compare effective values against requirements given as KEY=VALUE args.
# Prints every mismatch; returns 1 if any.
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

# Side-effect free. Checks the full identity contract before `use` stops the
# running serve, and records the effective port and model id.
# Fabric pins, derived this launch: interface, HCA and per-node GID (one KEY=value a word; the values hold no spaces).
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
    # The fabric pins too, so a .env line that disagrees with this launch refuses here, before the current serve
    # is stopped (derived read-only when the caller has not yet).
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
    # The launcher sees exactly what the child saw: the same seeds, exported.
    local kv; for kv in "${seeds[@]}"; do export "$kv"; done
    # Weights: download.sh has run (or AUTO_DOWNLOAD=1 fetches what is missing).
    ./start.sh
}

# Stop by recorded identity first (ADAPTER_HEAD/WORKER_CONTAINER and WORKER_SSH
# were loaded from the launched record by the CLI; ssh_bounded comes from
# fabric-facts.sh). The upstream stop runs only when today's .env still resolves
# to that recorded identity; otherwise it would act on foreign names or a
# foreign worker. Depends on nothing but the recorded cleanup identity.
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
