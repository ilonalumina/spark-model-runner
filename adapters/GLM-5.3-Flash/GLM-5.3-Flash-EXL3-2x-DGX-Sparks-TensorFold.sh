#!/usr/bin/env bash
# Adapter: GLM-5.3-Flash on TensorFold v0.6.0 with MiaAI-Lab's patches, 2x DGX Spark (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
# Run scripts/prepare.sh by hand before the first launch; serving settings belong in the recipe's .env.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID="GLM-5.3-Flash-EXL3"
ADAPTER_HEAD_CONTAINER="glm53-flash-tf"
ADAPTER_WORKER_CONTAINER="glm53-flash-tf"

# The recipe reads the environment before its files, so these exports take precedence.
_identity() {
    export WORKER="$WORKER_SSH" FABRIC_PEER="$WORKER_IP"
    export CONTAINER_NAME="$ADAPTER_HEAD_CONTAINER" PORT="$ADAPTER_PORT" HOST=0.0.0.0 SERVED_NAME="$ADAPTER_MODEL_ID"
}

# Fails on a missing or truncated shard: each .safetensors file must match the size its header declares.
_SNAPSHOT_CHECK='
import json, os, struct, sys
d = sys.argv[1]
bad = []
if not os.path.isfile(os.path.join(d, "config.json")):
    bad.append("config.json missing")
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.isfile(idx):
    for shard in sorted(set(json.load(open(idx))["weight_map"].values())):
        if not os.path.isfile(os.path.join(d, shard)):
            bad.append(shard + " missing (named by the index)")
files = total = 0
for root, _, names in os.walk(d, followlinks=True):
    for name in names:
        if not name.endswith(".safetensors"):
            continue
        p = os.path.join(root, name)
        size = 0
        try:
            size = os.path.getsize(p)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                head = json.loads(f.read(n))
            end = max((v["data_offsets"][1] for k, v in head.items() if k != "__metadata__"), default=0)
            if size != 8 + n + end:
                bad.append(f"{os.path.relpath(p, d)}: {size} bytes, its header says {8 + n + end}")
        except Exception as e:
            bad.append(f"{os.path.relpath(p, d)}: unreadable ({type(e).__name__}: {e})")
        files += 1
        total += size
if not files:
    bad.append("no .safetensors files")
if bad:
    print("\n".join(bad[:20]), file=sys.stderr)
    sys.exit(1)
print(files, total)
'

_checkpoints_ready() {
    local ids=("$MODEL_ID") id rev sub here there wdir head_files worker_files
    [ "$DRAFTER" = dflash2 ] && ids+=("$DFLASH2_ID")
    wdir="$(worker_hf_cache 1)" || { echo "adapter: cannot reach the worker ($WORKER)" >&2; return 1; }
    for id in "${ids[@]}"; do
        rev="$(snapshot_rev "$id")"
        [ -n "$rev" ] || { echo "adapter: $id: no pinned or cached revision" >&2; return 1; }
        sub="hub/models--${id//\//--}/snapshots/$rev"
        here="$(python3 -c "$_SNAPSHOT_CHECK" "$HF_CACHE/$sub")" \
            || { echo "adapter: $id @ ${rev:0:8} is not complete on the head ($HF_CACHE/$sub)" >&2; return 1; }
        there="$(worker 1 "python3 -c $(printf '%q' "$_SNAPSHOT_CHECK") '$wdir/$sub'")" \
            || { echo "adapter: $id @ ${rev:0:8} is not complete on the worker ($wdir/$sub)" >&2; return 1; }
        head_files="$(cd "$HF_CACHE/$sub" && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort)"
        worker_files="$(worker 1 "cd '$wdir/$sub' && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort")" || worker_files=""
        [ "$head_files" = "$worker_files" ] \
            || { echo "adapter: $id @ ${rev:0:8} differs between the head and the worker" >&2; return 1; }
    done
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing" >&2; return 1; }
    [ -f "$RECIPE_DIR/scripts/config.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/config.sh is missing" >&2; return 1; }
    local shape
    shape="$( cd "$RECIPE_DIR" && _identity && source ./scripts/config.sh && echo "${TP:-2} ${WORKER_WEIGHTS:-copy}" )"
    [ "$shape" = "2 copy" ] \
        || { echo "adapter: supported with TP=2 and WORKER_WEIGHTS=copy; this configuration has TP=${shape% *} WORKER_WEIGHTS=${shape#* }" >&2; return 1; }
    local now was
    now="$( cd "$RECIPE_DIR" && _identity && source ./scripts/config.sh && source ./scripts/nodes.sh && prepared_state )" \
        || { echo "adapter: could not read the recipe's prepared state (worker $WORKER_SSH reachable?)" >&2; return 1; }
    was="$( cd "$RECIPE_DIR" && _identity && source ./scripts/config.sh && cat "$PREPARED_MARKER" 2>/dev/null )" || was=""
    if [ "$now" != "$was" ]; then
        echo "adapter: the image or checkpoint is not prepared on both Sparks for this configuration; run scripts/prepare.sh" >&2
        echo "adapter:   now: $now" >&2
        echo "adapter:   was: ${was:-(never prepared)}" >&2
        return 1
    fi
    # The prepared marker records revisions, not files, so the files are checked separately.
    ( cd "$RECIPE_DIR" && _identity && source ./scripts/config.sh && source ./scripts/nodes.sh && _checkpoints_ready ) \
        || { echo "adapter: run scripts/prepare.sh to download or copy the missing files" >&2; return 1; }
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _identity
    export PREPARE=0
    ./start.sh
}

# Same as stop.sh (logs saved to ~/.cache/tensorfold-glm53/logs), but on the recorded worker.
adapter_stop() {
    local sl
    sl="$(cd "$RECIPE_DIR" 2>/dev/null && source ./scripts/nodes.sh >/dev/null 2>&1 && declare -f save_log)" || sl=""
    if [ -n "$sl" ]; then
        ( eval "$sl"; save_log "$HOME/.cache/tensorfold-glm53/logs" 0 "$ADAPTER_HEAD_CONTAINER" 10 ) >/dev/null 2>&1 || true
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
            "$sl; save_log \"\$HOME/.cache/tensorfold-glm53/logs\" 1 '$ADAPTER_WORKER_CONTAINER' 10" >/dev/null 2>&1 || true
    fi
    docker stop -t 30 "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
        "docker stop -t 30 '$ADAPTER_WORKER_CONTAINER' >/dev/null 2>&1; docker rm -f '$ADAPTER_WORKER_CONTAINER'" \
        >/dev/null 2>&1 || true
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
