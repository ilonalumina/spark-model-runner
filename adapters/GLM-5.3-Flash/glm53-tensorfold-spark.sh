#!/usr/bin/env bash
# Adapter: GLM-5.3-Flash on TensorFold, 2x DGX Spark (jayleaton/glm53-tensorfold-spark).
#
# The launcher is scripts/serve.sh. It reads config/prod.env, and a caller export wins for PORT, HOST,
# SERVED_NAME (and the other keys it lists); NAME is not in the file, so an export sets the container
# names ($NAME-r0 on the head, $NAME-r1 on the worker). The node identity (WORKER_SSH, HEAD_IP) and the
# CX7 names (NCCL_SOCKET_IFNAME, NCCL_IB_HCA) come from the file only, so they are verified against this
# launch's derived facts rather than exported. `serve.sh start` returns once /v1/models answers and the
# canary has run, and exits non-zero on a failed start.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID="GLM-5.3-Flash-EXL3"
ADAPTER_HEAD_CONTAINER="glm53-tf-r0"
ADAPTER_WORKER_CONTAINER="glm53-tf-r1"

_cfg() { # value of $1 in config/prod.env, read in a subshell
    ( set -a; . "$RECIPE_DIR/config/prod.env"; printf '%s' "${!1:-}" )
}

_in_list() { # is $1 one of the comma-separated entries of $2
    case ",$2," in *",$1,"*) return 0 ;; esac
    return 1
}

_identity_ok() { # config/prod.env names the node identity and fabric this launch uses (HEAD_IF ... WORKER_IB derived)
    local bad=0 k want got hcas
    for k in WORKER_SSH HEAD_IP; do
        want="${!k}"; got="$(_cfg "$k")"
        [ "$got" = "$want" ] || { echo "adapter: config/prod.env $k=$got, this launch uses $want" >&2; bad=1; }
    done
    got="$(_cfg NCCL_SOCKET_IFNAME)"
    [ "$got" = "$HEAD_IF" ] && [ "$got" = "$WORKER_IF" ] \
        || { echo "adapter: NCCL_SOCKET_IFNAME=$got; derived head $HEAD_IF, worker $WORKER_IF" >&2; bad=1; }
    # NCCL_IB_HCA may list several devices, comma-separated, each optionally with :port (the file's
    # default lists both CX7 functions); the derived device of each node must be one of them
    got="$(_cfg NCCL_IB_HCA)"; got="${got#=}"; hcas="$(sed 's/:[0-9]*//g' <<<"$got")"
    _in_list "$HEAD_IB" "$hcas" && _in_list "$WORKER_IB" "$hcas" \
        || { echo "adapter: NCCL_IB_HCA=$got; must list derived head $HEAD_IB, worker $WORKER_IB" >&2; bad=1; }
    [ "$bad" = 0 ]
}

# Side-effect free, and the gate `use` runs before it stops the current serve: the file is present and filled
# in, and names the identity and fabric this launch uses (derived here when the caller has not yet, read-only).
adapter_prepare() {
    [ -x "$RECIPE_DIR/scripts/serve.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/serve.sh is missing" >&2; return 1; }
    [ -f "$RECIPE_DIR/config/prod.env" ] || { echo "adapter: cp config/prod.env.example config/prod.env first" >&2; return 1; }
    if grep -qE '^[A-Z_]+="?<' "$RECIPE_DIR/config/prod.env"; then
        echo "adapter: config/prod.env still has <placeholders> (WORKER_SSH, HEAD_IP, HEAD_HF, WORKER_HF)" >&2; return 1
    fi
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_IB:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _identity_ok
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _identity_ok || return 1

    export NAME="${ADAPTER_HEAD_CONTAINER%-r0}"
    export PORT="$ADAPTER_PORT"
    export HOST=0.0.0.0          # the file's 127.0.0.1 would hide the API from other hosts
    export SERVED_NAME="$ADAPTER_MODEL_ID"
    ./scripts/serve.sh start
}

# Removes the recorded containers on the recorded worker, which is what `serve.sh stop` does, without
# re-reading today's config file for the worker address.
adapter_stop() {
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" docker rm -f "$ADAPTER_WORKER_CONTAINER" >/dev/null 2>&1 || true
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
