#!/usr/bin/env bash
# Adapter: DeepSeek-V4.1-Flash-1017 on TensorFold TP2, 2x DGX Spark (bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10); config.env wins, so it is verified.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID=""
ADAPTER_HEAD_CONTAINER="tp2-lane"
ADAPTER_WORKER_CONTAINER="tp2-lane"

_config() { echo "$RECIPE_DIR/config.env"; }

_cfg() {
    ( . "$(_config)" >/dev/null 2>&1; printf '%s' "${!1:-}" )
}

_in_list() {
    case ",$2," in *",$1,"*) return 0 ;; esac
    return 1
}

_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }

_node_addr() {
    local real ip
    real="$(ssh -G "$1" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }')"
    real="${real:-$1}"
    ip="$(getent ahosts "$real" 2>/dev/null | awk '{ print $1; exit }')"
    printf '%s' "${ip:-$real}"
}

_is_local() {
    local self addr
    self=" localhost 127.0.0.1 ::1 $(hostname 2>/dev/null) $(hostname -s 2>/dev/null) $(hostname -f 2>/dev/null) $(hostname -I 2>/dev/null) "
    addr="$(_node_addr "$1")"
    case "$self" in *" $1 "*|*" $addr "*) return 0 ;; esac
    return 1
}

_identity_ok() {
    local bad=0 got nodes hcas
    read -ra nodes <<<"$(_cfg NODES)"
    [ "${#nodes[@]}" = 2 ] || { echo "adapter: NODES=\"$(_cfg NODES)\"; it names two ssh aliases, rank 0 first" >&2; return 1; }
    _is_local "${nodes[0]}" \
        || { echo "adapter: NODES names ${nodes[0]} as rank 0; this host runs rank 0 and serves the API" >&2; bad=1; }
    got="$(_node_addr "${nodes[1]}")"
    [ "$got" = "$WORKER_IP" ] \
        || { echo "adapter: NODES names ${nodes[1]} ($got) as rank 1; this launch's WORKER_IP is $WORKER_IP" >&2; bad=1; }
    got="$(_cfg MASTER_IP)"
    [ "$got" = "$HEAD_IP" ] || { echo "adapter: MASTER_IP=$got; this launch's HEAD_IP is $HEAD_IP" >&2; bad=1; }
    got="$(_cfg NET_IF)"
    [ "$got" = "$HEAD_IF" ] && [ "$got" = "$WORKER_IF" ] \
        || { echo "adapter: NET_IF=$got; derived head $HEAD_IF, worker $WORKER_IF" >&2; bad=1; }
    got="$(_cfg RDMA_HCA)"; hcas="$(sed 's/:[0-9]*//g' <<<"$got")"
    _in_list "$HEAD_IB" "$hcas" && _in_list "$WORKER_IB" "$hcas" \
        || { echo "adapter: RDMA_HCA=$got; must list derived head $HEAD_IB, worker $WORKER_IB" >&2; bad=1; }
    [ "$HEAD_GIDX" = 5 ] && [ "$WORKER_GIDX" = 5 ] \
        || { echo "adapter: the link's RoCE v2 address is at GID index $HEAD_GIDX on the head and $WORKER_GIDX on the worker; this engine opens index 5 only (RUNNING.md section 2)" >&2; bad=1; }
    got="$(_cfg PORT)"
    [ "$got" = "$ADAPTER_PORT" ] || { echo "adapter: PORT=$got; set PORT=$ADAPTER_PORT in $(_config)" >&2; bad=1; }
    got="$(_cfg HOST)"
    [ "$got" = 0.0.0.0 ] || { echo "adapter: HOST=$got; set HOST=0.0.0.0 in $(_config) so the serve answers off this host" >&2; bad=1; }
    [ -n "$(_cfg SERVED_NAME)" ] || { echo "adapter: SERVED_NAME is empty in $(_config)" >&2; bad=1; }
    [ "$bad" = 0 ]
}

_image_ready() {
    local image h w
    image="$(_cfg IMAGE)"
    h="$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null)" \
        || { echo "adapter: image $image is not on the head (docker pull $image)" >&2; return 1; }
    w="$(_ssh "docker image inspect '$image' --format '{{.Id}}'" 2>/dev/null)" \
        || { echo "adapter: image $image is not on the worker (docker pull $image there)" >&2; return 1; }
    [ "$h" = "$w" ] || { echo "adapter: image $image differs: head $h, worker $w" >&2; return 1; }
}

# Args: rank MODEL_DIR CACHE_DIR KIT_DIR ENGINE_DIR ENGRAM_DIR.
_NODE_CHECK='
g=$1 m=$2 c=$3 k=$4 e=$5 r=$6
if [ "$g" = 1 ]; then
  for f in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
    [ -s "$m/$f" ] || echo "MODEL_DIR/$f"
  done
  [ -s "$c/dsv41_token_map.json" ] || echo "CACHE_DIR/dsv41_token_map.json (scripts/serve.sh inputs)"
  [ -s "$k/aot/aot.json" ] || echo "KIT_DIR/aot/aot.json"
  for f in tf-dsv41-lanes rdma_gather.fatbin; do [ -s "$e/$f" ] || echo "ENGINE_DIR/$f"; done
  [ -d "$r" ] || echo "ENGRAM_DIR $r"
fi
n=$(ls "$c"/rank-cache/rank${g}of2-*.bin 2>/dev/null | wc -l)
[ "$n" = 1 ] || echo "CACHE_DIR/rank-cache holds $n rank${g}of2-*.bin files; one is needed (scripts/serve.sh inputs)"
[ -s "$k/grammar/libtfgrammar.so" ] || echo "KIT_DIR/grammar/libtfgrammar.so"
[ -s "$k/vision/gate_bias_vl.safetensors" ] || echo "KIT_DIR/vision/gate_bias_vl.safetensors (kit/verify_kit.sh)"
ls "$k"/rope-*.f32 >/dev/null 2>&1 || echo "KIT_DIR/rope-*.f32 (kit/verify_kit.sh)"
'

_files_ready() {
    local m c k e r miss
    m="$(_cfg MODEL_DIR)"; c="$(_cfg CACHE_DIR)"; k="$(_cfg KIT_DIR)"; e="$(_cfg ENGINE_DIR)"; r="$(_cfg ENGRAM_DIR)"
    miss="$(bash -c "$_NODE_CHECK" _ 0 "$m" "$c" "$k" "$e" "$r")"
    [ -z "$miss" ] || { printf 'adapter: the head is missing:\n%s\n' "$(sed 's/^/  /' <<<"$miss")" >&2; return 1; }
    miss="$(_ssh "bash -c $(printf '%q' "$_NODE_CHECK") _ 1 '$m' '$c' '$k' '$e' '$r'")" \
        || { echo "adapter: could not check the worker's files over $WORKER_SSH" >&2; return 1; }
    [ -z "$miss" ] || { printf 'adapter: the worker is missing:\n%s\n' "$(sed 's/^/  /' <<<"$miss")" >&2; return 1; }
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/scripts/serve.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/serve.sh is missing" >&2; return 1; }
    [ -f "$(_config)" ] || { echo "adapter: $(_config) is missing (cp config.env.example config.env)" >&2; return 1; }
    if grep -v '^[[:space:]]*#' "$(_config)" | grep -q '<[A-Z0-9_]*>'; then
        echo "adapter: $(_config) still holds a <PLACEHOLDER>" >&2; return 1
    fi
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_GIDX:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _identity_ok || return 1
    ADAPTER_MODEL_ID="$(_cfg SERVED_NAME)"
    _image_ready || return 1
    _files_ready || return 1
    ( cd "$RECIPE_DIR" && DSV41_CONFIG="$(_config)" ./scripts/serve.sh preflight ) >/dev/null \
        || { echo "adapter: the recipe's preflight failed (above; RUNNING.md section 2)" >&2; return 1; }
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _identity_ok || return 1
    DSV41_CONFIG="$(_config)" ./scripts/serve.sh start
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
