#!/usr/bin/env bash
# Adapter: DeepSeek-V4.1-Flash on TensorFold, 2x DGX Spark (jayleaton/deepseek-v41-tensorfold-spark).
#
# The launcher is scripts/serve.sh. It reads config/prod.env, and a non-empty caller export wins for the keys it
# lists (PORT, HOST, NAME, SERVED_NAME and others); NAME sets the container names ($NAME-r0 on the head, $NAME-r1
# on the worker). The node identity (WORKER_SSH, HEAD_IP), the CX7 names (NCCL_SOCKET_IFNAME, NCCL_IB_HCA) and the
# node-local paths come from the file only, so the identity is verified against this launch's derived facts rather
# than exported. `serve.sh start` returns once /v1/models answers, rank 0 shows PARALLEL slots and the canary has
# passed, and exits non-zero on a failed start (a failed canary removes both ranks first).
#
# The recipe's watchdog and boot units (scripts/systemd) are not used: model-runner owns start, stop and boot, and the
# watchdog's own restart would run without this adapter's exports.
#
# Routed-expert pruning is on in the recipe's config/prod.env.example, its measured config: in decode each
# token keeps the highest-weighted of its 6 routed experts until their router weights reach 0.85, at least 3, and the
# kept weights are renormalized (TF_DSV41_EXPERT_TOPP=0.85, TF_DSV41_EXPERT_MIN_K=3, TF_DSV41_EXPERT_RENORM=kept).
# Lossy: replies change (top-1 agreement 0.9944, MMLU-200 88.5% against 87.5% unpruned), +5% decode. The shared
# expert and prefill are not pruned. To serve the unpruned model, delete those three lines from config/prod.env and
# run `model-runner restart`; nothing is rebuilt.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID="DeepSeek-V4.1-Flash-TF"
ADAPTER_HEAD_CONTAINER="dsv41-tf-r0"
ADAPTER_WORKER_CONTAINER="dsv41-tf-r1"

_cfg() { # value of $1 in config/prod.env, read in a subshell
    ( set -a; . "$RECIPE_DIR/config/prod.env"; printf '%s' "${!1:-}" )
}

_in_list() { # is $1 one of the comma-separated entries of $2
    case ",$2," in *",$1,"*) return 0 ;; esac
    return 1
}

_exports() {
    export NAME="${ADAPTER_HEAD_CONTAINER%-r0}"
    export PORT="$ADAPTER_PORT"
    export HOST=0.0.0.0          # the file's 127.0.0.1 would hide the API from other hosts
    export SERVED_NAME="$ADAPTER_MODEL_ID"
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
    # NCCL_IB_HCA may list several devices, comma-separated, each optionally with :port (the file's default lists
    # both CX7 functions); the derived device of each node must be one of them
    got="$(_cfg NCCL_IB_HCA)"; got="${got#=}"; hcas="$(sed 's/:[0-9]*//g' <<<"$got")"
    _in_list "$HEAD_IB" "$hcas" && _in_list "$WORKER_IB" "$hcas" \
        || { echo "adapter: NCCL_IB_HCA=$got; must list derived head $HEAD_IB, worker $WORKER_IB" >&2; bad=1; }
    [ "$bad" = 0 ]
}

# A model folder as the engine reads it, checked without loading anything: config.json and the tokenizer files
# present, every shard model.safetensors.index.json names present, and every .safetensors file exactly the size its
# own header declares (8 + header + the largest tensor end), so a missing or truncated shard fails here. Prints the
# folder's files with their sizes, for the head / worker comparison.
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
    dirs[:] = [x for x in dirs if not x.startswith(".")]   # the .cache bookkeeping of hf download is not the model
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

# One rank's packed Engram shards (scripts/pack_engram.py's format): engram-l{1,14}-r<rank>of2.bin each with the
# DSV41EN1 magic, its own layer, a non-empty row range and 264-byte rows, and exactly 4096 + rows x 264 bytes, so an
# empty or truncated shard fails here. Prints "<layer> <lo> <hi> <total>" a shard, for the rank 0 / rank 1 comparison.
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

_files_ready() { # model folder and Engram shards on both nodes, as the file names them; reads only
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
    # the two ranks split each layer's table between them: rank 0 from row 0, rank 1 from where rank 0 ends to the end
    local layer lo0 hi0 t0 lo1 hi1 t1
    while read -r layer lo0 hi0 t0 && read -r _ lo1 hi1 t1 <&3; do
        [ "$lo0" = 0 ] && [ "$lo1" = "$hi0" ] && [ "$hi1" = "$t0" ] && [ "$t1" = "$t0" ] \
            || { echo "adapter: Engram layer $layer: rank 0 rows $lo0..$hi0 of $t0, rank 1 rows $lo1..$hi1 of $t1" >&2; return 1; }
    done <<<"$e0" 3<<<"$e1"
}

# Side-effect free (model-runner's contract), and the same gate `use` runs before it stops the current serve: the
# file is present and filled in, it names the identity and fabric this launch uses (derived here when the caller has
# not yet, read-only), the model and Engram shards are complete on both nodes, then the recipe's own static preflight,
# which only reads both nodes (image on both and identical, RoCE port states).
adapter_prepare() {
    [ -x "$RECIPE_DIR/scripts/serve.sh" ] || { echo "adapter: $RECIPE_DIR/scripts/serve.sh is missing" >&2; return 1; }
    [ -f "$RECIPE_DIR/config/prod.env" ] || { echo "adapter: cp config/prod.env.example config/prod.env first" >&2; return 1; }
    if grep -qE '^[A-Z_]+="?[^"#]*<' "$RECIPE_DIR/config/prod.env"; then
        echo "adapter: config/prod.env still has <placeholders> (WORKER_SSH, HEAD_IP and the node paths)" >&2; return 1
    fi
    # `use` and the unit's start call this before model-runner derives the fabric: its own bounded wait (FABRIC_WAIT_SEC,
    # read-only), so a boot start still rides out the link's first addressless seconds
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

# Removes the recorded containers on the recorded worker, which is what `serve.sh stop` does, without re-reading
# today's config file for the worker address.
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
