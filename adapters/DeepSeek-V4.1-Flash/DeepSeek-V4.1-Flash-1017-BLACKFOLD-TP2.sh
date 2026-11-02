#!/usr/bin/env bash
# Adapter: BLACKFOLD DeepSeek-V4.1-Flash-1017 on TensorFold native, 2x DGX Spark (Blackwellboy/DeepSeek-V4.1-Flash-1017-BLACKFOLD-TP2); deploy/blackfold.env wins, so it is verified.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID=""
ADAPTER_HEAD_CONTAINER="ds41-native"
ADAPTER_WORKER_CONTAINER="ds41-native"

_config() { echo "$RECIPE_DIR/deploy/blackfold.env"; }

_cfg() {
    ( . "$(_config)" >/dev/null 2>&1; printf '%s' "${!1:-}" )
}

_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }

_node_addr() {
    local real ip
    real="$(ssh -G "$1" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }')"
    real="${real:-$1}"
    ip="$(getent ahosts "$real" 2>/dev/null | awk '{ print $1; exit }')"
    printf '%s' "${ip:-$real}"
}

# The index the launcher picks on a node: the first IPv4 RoCE v2 entry of the primary device.
_GID_PICK='d=$1; for i in $(seq 0 15); do g=$(cat /sys/class/infiniband/$d/ports/1/gids/$i 2>/dev/null) || continue; t=$(cat /sys/class/infiniband/$d/ports/1/gid_attrs/types/$i 2>/dev/null); case "$g" in 0000:0000:0000:0000:0000:ffff:*) [ "$t" = "RoCE v2" ] && { echo $i; exit 0; } ;; esac; done; echo -1'

_identity_ok() {
    local bad=0 got hca
    got="$(_node_addr "$(_cfg WORKER_SSH)")"
    [ "$got" = "$WORKER_IP" ] \
        || { echo "adapter: WORKER_SSH=$(_cfg WORKER_SSH) ($got); this launch's WORKER_IP is $WORKER_IP" >&2; bad=1; }
    got="$(_cfg HEAD_ADDR)"
    [ "$got" = "$HEAD_IP" ] || { echo "adapter: HEAD_ADDR=$got; this launch's HEAD_IP is $HEAD_IP" >&2; bad=1; }
    got="$(_cfg PRIMARY_NETDEV)"
    [ "$got" = "$HEAD_IF" ] && [ "$got" = "$WORKER_IF" ] \
        || { echo "adapter: PRIMARY_NETDEV=$got; derived head $HEAD_IF, worker $WORKER_IF" >&2; bad=1; }
    hca="$(_cfg PRIMARY_HCA)"
    [ "$hca" = "$HEAD_IB" ] && [ "$hca" = "$WORKER_IB" ] \
        || { echo "adapter: PRIMARY_HCA=$hca; derived head $HEAD_IB, worker $WORKER_IB" >&2; bad=1; }
    got="$(bash -c "$_GID_PICK" _ "$hca")"
    [ "$got" = "$HEAD_GIDX" ] \
        || { echo "adapter: the launcher would use GID index $got on the head; derived $HEAD_GIDX" >&2; bad=1; }
    got="$(_ssh "bash -c $(printf '%q' "$_GID_PICK") _ '$hca'")"
    [ "$got" = "$WORKER_GIDX" ] \
        || { echo "adapter: the launcher would use GID index $got on the worker; derived $WORKER_GIDX" >&2; bad=1; }
    got="$(_cfg PORT)"
    [ "$got" = "$ADAPTER_PORT" ] || { echo "adapter: PORT=$got; set PORT=$ADAPTER_PORT in $(_config)" >&2; bad=1; }
    [ -n "$(_cfg SERVED_NAME)" ] || { echo "adapter: SERVED_NAME is empty in $(_config)" >&2; bad=1; }
    [ "$bad" = 0 ]
}

_image_ready() {
    local image h w
    image="$(_cfg IMAGE)"
    h="$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null)" \
        || { echo "adapter: image $image is not on the head" >&2; return 1; }
    w="$(_ssh "docker image inspect '$image' --format '{{.Id}}'" 2>/dev/null)" \
        || { echo "adapter: image $image is not on the worker" >&2; return 1; }
    [ "$h" = "$w" ] || { echo "adapter: image $image differs: head $h, worker $w" >&2; return 1; }
}

# Args: build kit template MODEL_DIR ENGRAM_DIR CACHE_DIR engine-binary.
_NODE_CHECK='
b=$1 k=$2 t=$3 m=$4 r=$5 c=$6 e=$7
[ -x "$b/engine/zig-out/$e" ] || echo "$b/engine/zig-out/$e"
[ -s "$k/aot/aot.json" ] || echo "$k/aot/aot.json"
[ -s "$k/vision/gate_bias_vl.safetensors" ] || echo "$k/vision/gate_bias_vl.safetensors"
[ -s "$t" ] || echo "$t"
for f in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
  [ -s "$m/$f" ] || echo "$m/$f"
done
ls "$r"/*.safetensors >/dev/null 2>&1 || echo "$r/*.safetensors"
[ -d "$c/tf-ds-rank" ] && [ -n "$(ls -A "$c/tf-ds-rank" 2>/dev/null)" ] || echo "$c/tf-ds-rank/"
[ -s "$c/dsv41_token_map.json" ] || echo "$c/dsv41_token_map.json"
command -v python3 >/dev/null || echo "python3"
'

_files_ready() {
    local m r c miss
    m="$(_cfg MODEL_DIR)"; r="$(_cfg ENGRAM_DIR)"; c="$(_cfg CACHE_DIR)"
    [ -f "$RECIPE_DIR/tools/engram_slice.py" ] || { echo "adapter: $RECIPE_DIR/tools/engram_slice.py is missing" >&2; return 1; }
    miss="$(bash -c "$_NODE_CHECK" _ "$(_cfg HEAD_BUILD)" "$(_cfg HEAD_KIT)" "$(_cfg HEAD_TEMPLATE)" "$m" "$r" "$c" native/bin/tensorfold-native)"
    [ -z "$miss" ] || { printf 'adapter: the head is missing:\n%s\n' "$(sed 's/^/  /' <<<"$miss")" >&2; return 1; }
    miss="$(_ssh "bash -c $(printf '%q' "$_NODE_CHECK") _ '$(_cfg WORKER_BUILD)' '$(_cfg WORKER_KIT)' '$(_cfg WORKER_TEMPLATE)' '$m' '$r' '$c' bin/tf-dsv41-lanes")" \
        || { echo "adapter: could not check the worker's files over $WORKER_SSH" >&2; return 1; }
    [ -z "$miss" ] || { printf 'adapter: the worker is missing:\n%s\n' "$(sed 's/^/  /' <<<"$miss")" >&2; return 1; }
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/deploy/blackfold-native" ] || { echo "adapter: $RECIPE_DIR/deploy/blackfold-native is missing" >&2; return 1; }
    [ -f "$(_config)" ] || { echo "adapter: $(_config) is missing (cp deploy/blackfold.env.example deploy/blackfold.env)" >&2; return 1; }
    if grep -v '^[[:space:]]*#' "$(_config)" | grep -q 'CHANGE_ME'; then
        echo "adapter: $(_config) still holds a CHANGE_ME" >&2; return 1
    fi
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_GIDX:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _identity_ok || return 1
    ADAPTER_MODEL_ID="$(_cfg SERVED_NAME)"
    _image_ready || return 1
    _files_ready || return 1
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _identity_ok || return 1
    deploy/blackfold-native start
}

adapter_stop() {
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    _ssh "docker rm -f '$ADAPTER_WORKER_CONTAINER'" >/dev/null 2>&1 || true
}

adapter_health() {
    curl -fsS -m 10 -o /dev/null "http://127.0.0.1:${ADAPTER_PORT}/health"
}

adapter_describe() {
    printf 'dir    : %s\n' "$RECIPE_DIR"
    printf 'model  : %s\n' "${ADAPTER_MODEL_ID:-$(_cfg SERVED_NAME)}"
    printf 'api    : http://127.0.0.1:%s/v1\n' "$ADAPTER_PORT"
    printf 'head   : %s\n' "$ADAPTER_HEAD_CONTAINER"
    printf 'worker : %s on %s\n' "$ADAPTER_WORKER_CONTAINER" "$WORKER_SSH"
}
