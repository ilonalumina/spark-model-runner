#!/usr/bin/env bash
# Adapter: DeepSeek-V4.1-Flash-1017 on TensorFold's Zig engine, 2x DGX Spark (MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks).
# Before the first launch: set NFS_SERVER and NFS_PATH in scripts/local.sh, then run scripts/prepare.sh with no serve running.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID="DeepSeek-V4.1-Flash-1017"
ADAPTER_HEAD_CONTAINER="dsv41-1017-exl3-tf"
ADAPTER_WORKER_CONTAINER="dsv41-1017-exl3-tf"

_identity() {
    export WORKER="$WORKER_SSH"
    export CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER" PORT="$ADAPTER_PORT" HOST=0.0.0.0 SERVED_NAME="$ADAPTER_MODEL_ID"
}

_fabric() {
    export FABRIC_PEER="$WORKER_IP"
}

_prepared_files_ready() {
    local ident snap_c want wwant f esnap need have
    ident="$(image_ident "$IMAGE")"
    [ "$ident" != missing ] || { echo "adapter: image $IMAGE is not on the head" >&2; return 1; }
    [ "$(worker_image_ident 1 "$IMAGE")" = "$ident" ] \
        || { echo "adapter: $(worker_host 1)'s $IMAGE is missing or differs from the head's" >&2; return 1; }
    snap_c="$(model_in_container)"
    want="$(docker run --rm --pull=never --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HF_CACHE:/hf:ro" \
        --entrypoint bash "$IMAGE" -o pipefail -c "tf-dsv41-rank-cache $snap_c /tmp/x 0 2 --index-only | sha256sum | cut -c1-20")" \
        && [ -n "$want" ] || { echo "adapter: could not name rank 0's weight file from the checkpoint index" >&2; return 1; }
    [ -s "$DATA_DIR/rank-cache/rank0of2-$want.bin" ] \
        || { echo "adapter: rank 0's weight file $DATA_DIR/rank-cache/rank0of2-$want.bin is missing" >&2; return 1; }
    wwant="$(worker 1 "docker run --rm --pull=never --network none --user \$(id -u):\$(id -g) -e HOME=/tmp -v '$NFS_VOLUME:/hf:ro' --entrypoint bash '$IMAGE' -o pipefail -c 'tf-dsv41-rank-cache $snap_c /tmp/x 1 2 --index-only | sha256sum | cut -c1-20'")" \
        && [ -n "$wwant" ] || { echo "adapter: could not name rank 1's weight file on $(worker_host 1)" >&2; return 1; }
    worker 1 "[ -s $WORKER_CACHE_DIR/rank1of2-$wwant.bin ]" \
        || { echo "adapter: rank 1's weight file rank1of2-$wwant.bin is missing in $WORKER_CACHE_DIR on $(worker_host 1)" >&2; return 1; }
    esnap="$(engram_cache_dir)/snapshots/$(engram_rev)"
    need=0
    for f in $ENGRAM_SHARDS; do
        [ -f "$esnap/$f" ] || { echo "adapter: Engram shard $esnap/$f is missing on the head" >&2; return 1; }
        if [ -n "${ENGRAM_SHARD_SIZE[$f]:-}" ] && [ "$(stat -L -c %s "$esnap/$f")" != "${ENGRAM_SHARD_SIZE[$f]}" ]; then
            echo "adapter: Engram shard $esnap/$f is not the release's ${ENGRAM_SHARD_SIZE[$f]} bytes" >&2; return 1
        fi
        need=$(( need + $(stat -L -c %s "$esnap/$f") ))
    done
    have="$(worker 1 "d=$WORKER_ENGRAM_DIR; s=0; for f in $ENGRAM_SHARDS; do [ -f \$d/\$f ] && s=\$(( s + \$(stat -L -c %s \$d/\$f) )); done; echo \$s")" || have=""
    [ "$have" = "$need" ] \
        || { echo "adapter: Engram's shards in $WORKER_ENGRAM_DIR on $(worker_host 1) are ${have:-unreadable} bytes, not the head's $need" >&2; return 1; }
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing" >&2; return 1; }
    [ -f "$RECIPE_DIR/scripts/config.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/config.sh is missing" >&2; return 1; }
    [ -f "$RECIPE_DIR/scripts/local.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/local.sh is missing (NFS_SERVER, NFS_PATH)" >&2; return 1; }
    local now was
    now="$( cd "$RECIPE_DIR" && _identity && _fabric && source ./scripts/config.sh && source ./scripts/nodes.sh && prepared_state )" \
        || { echo "adapter: could not read the recipe's prepared state (worker $WORKER_SSH reachable?)" >&2; return 1; }
    was="$( cd "$RECIPE_DIR" && _identity && _fabric && source ./scripts/config.sh && cat "$PREPARED_MARKER" 2>/dev/null )" || was=""
    if [ "$now" != "$was" ]; then
        echo "adapter: scripts/prepare.sh has not prepared this setup (settings, revisions or the head's image changed); stop the running serve, then run it" >&2
        echo "adapter:   now: $now" >&2
        echo "adapter:   was: ${was:-(never prepared)}" >&2
        return 1
    fi
    ( cd "$RECIPE_DIR" && _identity && _fabric && source ./scripts/config.sh && source ./scripts/nodes.sh && _prepared_files_ready ) \
        || { echo "adapter: stop the running serve, then run scripts/prepare.sh to make the missing pieces" >&2; return 1; }
    ( cd "$RECIPE_DIR" && _identity && _fabric && source ./scripts/config.sh && source ./scripts/nodes.sh && worker_files_check 1 ) \
        || { echo "adapter: the worker cannot read the model and kit over NFS (above)" >&2; return 1; }
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _identity
    _fabric
    export PREPARE=0
    ./start.sh
}

adapter_stop() {
    ( cd "$RECIPE_DIR" && _identity && ./stop.sh )
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
