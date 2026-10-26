#!/usr/bin/env bash
# Adapter: DeepSeek-V4.1-Flash on TensorFold, 2x DGX Spark (jayleaton/deepseek-v41-tensorfold-spark).
# Node identity and CX7 names are read only from config/prod.env, so they are checked against the derived fabric, not exported.
# Do not enable the recipe's own watchdog or boot units; model-runner owns start, stop and boot.
# The example config enables lossy expert pruning (TF_DSV41_EXPERT_TOPP, _MIN_K, _RENORM); remove those lines for the unpruned model.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID="DeepSeek-V4.1-Flash-TF"
ADAPTER_HEAD_CONTAINER="dsv41-tf-r0"
ADAPTER_WORKER_CONTAINER="dsv41-tf-r1"

_cfg() {
    ( set -a; . "$RECIPE_DIR/config/prod.env"; printf '%s' "${!1:-}" )
}

_in_list() {
    case ",$2," in *",$1,"*) return 0 ;; esac
    return 1
}

_exports() {
    export NAME="${ADAPTER_HEAD_CONTAINER%-r0}"
    export PORT="$ADAPTER_PORT"
    export HOST=0.0.0.0
    export SERVED_NAME="$ADAPTER_MODEL_ID"
}

_identity_ok() {
    local bad=0 k want got hcas
    for k in WORKER_SSH HEAD_IP; do
        want="${!k}"; got="$(_cfg "$k")"
        [ "$got" = "$want" ] || { echo "adapter: config/prod.env $k=$got, this launch uses $want" >&2; bad=1; }
    done
    got="$(_cfg NCCL_SOCKET_IFNAME)"
    [ "$got" = "$HEAD_IF" ] && [ "$got" = "$WORKER_IF" ] \
        || { echo "adapter: NCCL_SOCKET_IFNAME=$got; derived head $HEAD_IF, worker $WORKER_IF" >&2; bad=1; }
    # NCCL_IB_HCA may list several devices, each with an optional :port.
    got="$(_cfg NCCL_IB_HCA)"; got="${got#=}"; hcas="$(sed 's/:[0-9]*//g' <<<"$got")"
    _in_list "$HEAD_IB" "$hcas" && _in_list "$WORKER_IB" "$hcas" \
        || { echo "adapter: NCCL_IB_HCA=$got; must list derived head $HEAD_IB, worker $WORKER_IB" >&2; bad=1; }
    [ "$bad" = 0 ]
}

# Fails on missing files or a truncated shard; prints the file list with sizes for comparing the two nodes.
_MODEL_CHECK='
import json, os, struct, sys
d = sys.argv[1]
bad = []
for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    if not os.path.isfile(os.path.join(d, f)):
        bad.append(f + " missing")
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.isfile(idx):
    for shard in sorted(set(json.load(open(idx))["weight_map"].values())):
        if not os.path.isfile(os.path.join(d, shard)):
            bad.append(shard + " missing (named by the index)")
files = 0
listing = []
for root, dirs, names in os.walk(d, followlinks=True):
    dirs[:] = [x for x in dirs if not x.startswith(".")]
    for name in names:
        p = os.path.join(root, name)
        listing.append(f"{os.path.relpath(p, d)} {os.path.getsize(p)}")
        if not name.endswith(".safetensors"):
            continue
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
if not files:
    bad.append("no .safetensors files")
if bad:
    print("\n".join(bad[:20]), file=sys.stderr)
    sys.exit(1)
print("\n".join(sorted(listing)))
'

# Validates one rank's Engram shards in scripts/pack_engram.py's format; prints "<layer> <lo> <hi> <total>" per shard.
_ENGRAM_CHECK='
import os, struct, sys
d, rank = sys.argv[1], int(sys.argv[2])
bad = []
for layer in (1, 14):
    p = os.path.join(d, f"engram-l{layer}-r{rank}of2.bin")
    try:
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            magic, ly, lo, hi, total, row = struct.unpack("<6Q", f.read(48))
    except Exception as e:
        bad.append(f"{p}: unreadable ({type(e).__name__}: {e})")
        continue
    if (magic, ly, row) != (0x31344E4531565344, layer, 264) or not lo < hi <= total:
        bad.append(f"{p}: header magic {magic:#x} layer {ly} rows {lo}..{hi} of {total} row {row}")
    elif size != 4096 + (hi - lo) * row:
        bad.append(f"{p}: {size} bytes, its header says {4096 + (hi - lo) * row}")
    else:
        print(layer, lo, hi, total)
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
'

_files_ready() {
    local hm wm he we h_list w_list e0 e1
    hm="$(_cfg HEAD_MODEL)"; wm="$(_cfg WORKER_MODEL)"; he="$(_cfg HEAD_ENGRAM)"; we="$(_cfg WORKER_ENGRAM)"
    h_list="$(python3 -c "$_MODEL_CHECK" "$hm")" \
        || { echo "adapter: the model at $hm on the head is not complete (docs/INSTALL.md)" >&2; return 1; }
    w_list="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "python3 -c $(printf '%q' "$_MODEL_CHECK") '$wm'")" \
        || { echo "adapter: the model at $wm on the worker is not complete (docs/INSTALL.md)" >&2; return 1; }
    [ "$h_list" = "$w_list" ] \
        || { echo "adapter: the model differs between the head ($hm) and the worker ($wm): files or sizes" >&2; return 1; }
    e0="$(python3 -c "$_ENGRAM_CHECK" "$he" 0)" \
        || { echo "adapter: the rank 0 Engram shards at $he on the head are not complete (docs/INSTALL.md: pack_engram.py --rank 0)" >&2; return 1; }
    e1="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "python3 -c $(printf '%q' "$_ENGRAM_CHECK") '$we' 1")" \
        || { echo "adapter: the rank 1 Engram shards at $we on the worker are not complete (docs/INSTALL.md: pack_engram.py --rank 1)" >&2; return 1; }
    # Rank 0 holds rows from 0, rank 1 from where rank 0 ends to the last row.
    local layer lo0 hi0 t0 lo1 hi1 t1
    while read -r layer lo0 hi0 t0 && read -r _ lo1 hi1 t1 <&3; do
        [ "$lo0" = 0 ] && [ "$lo1" = "$hi0" ] && [ "$hi1" = "$t0" ] && [ "$t1" = "$t0" ] \
            || { echo "adapter: Engram layer $layer: rank 0 rows $lo0..$hi0 of $t0, rank 1 rows $lo1..$hi1 of $t1" >&2; return 1; }
    done <<<"$e0" 3<<<"$e1"
}

adapter_prepare() {
    [ -x "$RECIPE_DIR/scripts/serve.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/serve.sh is missing" >&2; return 1; }
    [ -f "$RECIPE_DIR/config/prod.env" ] || { echo "adapter: cp config/prod.env.example config/prod.env first" >&2; return 1; }
    if grep -qE '^[A-Z_]+="?[^"#]*<' "$RECIPE_DIR/config/prod.env"; then
        echo "adapter: config/prod.env still has <placeholders> (WORKER_SSH, HEAD_IP and the node paths)" >&2; return 1
    fi
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_IB:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _identity_ok || return 1
    _files_ready || return 1
    ( cd "$RECIPE_DIR" && _exports && ./scripts/serve.sh preflight static ) \
        || { echo "adapter: the recipe's preflight failed (above); scripts/serve.sh build" >&2; return 1; }
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _identity_ok || return 1
    _exports
    ./scripts/serve.sh start
}

# Same effect as `serve.sh stop`, but on the recorded worker rather than the one in today's config.
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
