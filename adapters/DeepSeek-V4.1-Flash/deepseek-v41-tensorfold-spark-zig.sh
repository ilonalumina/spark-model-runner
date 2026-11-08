#!/usr/bin/env bash
# Adapter: DeepSeek-V4.1-Flash on TensorFold's CUDA Zig engine, 2x DGX Spark (jayleaton/deepseek-v41-tensorfold-spark, docs/ZIG-SERVE.md).
# Starts both ranks from the head with the same `docker run` as scripts/run-zig.sh, detached, except --max-tokens is the full context.
# The worker needs the image, model, Engram shards, assets and cache at the head's paths.
# The example config enables lossy expert pruning (TF_DSV41_EXPERT_TOPP=0.85); remove that line for the unpruned model.

ADAPTER_PORT=8888
ADAPTER_MODEL_ID="DeepSeek-V4.1-Flash-TF"
ADAPTER_HEAD_CONTAINER="tensorfold-zig-r0"
ADAPTER_WORKER_CONTAINER="tensorfold-zig-r1"
ADAPTER_READY_TIMEOUT="${ADAPTER_READY_TIMEOUT:-3600}"

_config() { echo "$RECIPE_DIR/config/prod-zig.env"; }

_cfg() {
    env -i bash -c 'set -a; . "$1" >/dev/null 2>&1; printf "%s" "${!2:-}"' _ "$(_config)" "$1"
}

_in_list() {
    case ",$2," in *",$1,"*) return 0 ;; esac
    return 1
}

_AUTH_FILE_CHECK='
import os, stat, sys
path, shared = sys.argv[1], [os.path.realpath(d) for d in sys.argv[2:] if d]
try:
    if not os.path.isabs(path) or any(c in path for c in ":\r\n") or os.path.realpath(path) != os.path.normpath(path):
        raise ValueError
    info = os.lstat(path)
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid not in (os.geteuid(), 0) or not 0 < info.st_size <= 1048576):
        raise ValueError
    if any(os.path.commonpath([path, d]) == d for d in shared):
        raise ValueError
except (OSError, ValueError):
    raise SystemExit("adapter: NATIVE_AUTH_FILE must be a nonempty regular file with mode 0600, owned by you or root, at an absolute symlink-free path outside the checkout and the model, engram, assets, prepared, state and cache directories")
'

_run_cmd() {
    env -i PATH="$PATH" bash -c '
        set -euo pipefail
        rank=$1 head=$2 worker=$3
        cd "$5"
        set -a; . "$4"; set +a
        case "${TF_DSV41_DEFAULT_EFFORT:-}" in ""|low|medium|high|xhigh) ;;
            *) echo "adapter: TF_DSV41_DEFAULT_EFFORT must be low, medium, high or xhigh" >&2; exit 2 ;; esac
        if [ "$rank" = 0 ] && [ -n "${TENSORFOLD_API_KEY:-}" ] && [ -z "${NATIVE_AUTH_FILE:-}" ]; then
            echo "adapter: native authentication requires NATIVE_AUTH_FILE; move TENSORFOLD_API_KEY out of the config into a mode-0600 file" >&2; exit 2
        fi
        export TF_EXPERIMENTAL=deepseek
        # GB10: NCCL cuMem allocations exhaust the driver pool at boot (NV_ERR_NO_MEMORY); off unless configured.
        export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
        names="$(compgen -v | python3 scripts/engine_settings.py forward)"
        rev="$(cat config/tensorfold.rev)"
        name=$head; [ "$rank" = 1 ] && name=$worker
        args=(docker run -d --restart no --init --name "$name" --gpus all --ipc=host --network host
              --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
              --label "org.tensorfold.revision=$rev" --label "org.tensorfold.rank=$rank"
              -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" -v "$ASSETS_DIR:/assets:ro"
              -v "$PREPARED_DIR:/prepared" -v "$STATE_DIR:/state" -v "$STATE_DIR/sessions:/sessions"
              -v "$CACHE_DIR:/cache")
        while IFS= read -r key; do
            [ -n "$key" ] && args+=(-e "$key=${!key}")
        done <<<"$names"
        args+=(-e TF_TP_WORLD=2 -e "TF_TP_RANK=$rank" -e TF_TP_DEVICE=0
               -e "TF_TP_MASTER=$TF_TP_MASTER" -e "TF_TP_PORT=${TF_TP_PORT:-29571}"
               -e "TF_COMM_BACKEND=${TF_COMM_BACKEND:-roce}"
               -e TF_DSV41_ASSETS=/assets -e TF_DSV41_ENGRAM_DIR=/engram
               -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin)
        [ -z "${CPUSET_CPUS:-}" ] || args+=(--cpuset-cpus "$CPUSET_CPUS")
        auth=""
        if [ "$rank" = 0 ] && [ -n "${NATIVE_AUTH_FILE:-}" ]; then
            python3 -c "$6" "$NATIVE_AUTH_FILE" "$PWD" "$MODEL_DIR" "$ENGRAM_DIR" "$ASSETS_DIR" "$PREPARED_DIR" "$STATE_DIR" "$CACHE_DIR"
            args+=(-v "$NATIVE_AUTH_FILE:/run/tensorfold/auth-keys:ro")
            auth=1
        fi
        args+=("$IMAGE")
        if [ "$rank" = 0 ]; then
            args+=(tensorfold-dsv41 serve /model --host "${HOST:-localhost}" --port "${PORT:-8000}"
                   --parallel "${TF_DSV41_SLOTS:-1}" --context "${TF_DSV41_CONTEXT:-4096}"
                   --name DeepSeek-V4.1-Flash-TF --max-tokens "${TF_DSV41_CONTEXT:-4096}")
            [ "${TF_DSV41_THINKING:-1}" != 0 ] || args+=(--no-thinking)
            [ -z "${TF_DSV41_DEFAULT_EFFORT:-}" ] || args+=(--reasoning-effort "$TF_DSV41_DEFAULT_EFFORT")
            [ -z "$auth" ] || args+=(--api-key-file /run/tensorfold/auth-keys)
        else
            args+=(tf-dsv41 follow /model /assets)
        fi
        printf "%q " "${args[@]}"
    ' _ "$1" "$ADAPTER_HEAD_CONTAINER" "$ADAPTER_WORKER_CONTAINER" "$(_config)" "$RECIPE_DIR" "$_AUTH_FILE_CHECK"
}

_identity_ok() {
    local bad=0 got hcas
    got="$(_cfg TF_TP_MASTER)"
    [ "$got" = "$HEAD_IP" ] || { echo "adapter: TF_TP_MASTER=$got, this launch's HEAD_IP is $HEAD_IP" >&2; bad=1; }
    got="$(_cfg NCCL_SOCKET_IFNAME)"
    [ "$got" = "$HEAD_IF" ] && [ "$got" = "$WORKER_IF" ] \
        || { echo "adapter: NCCL_SOCKET_IFNAME=$got; derived head $HEAD_IF, worker $WORKER_IF" >&2; bad=1; }
    got="$(_cfg NCCL_IB_HCA)"; hcas="$(sed 's/:[0-9]*//g' <<<"${got#=}")"
    _in_list "$HEAD_IB" "$hcas" && _in_list "$WORKER_IB" "$hcas" \
        || { echo "adapter: NCCL_IB_HCA=$got; must list derived head $HEAD_IB, worker $WORKER_IB" >&2; bad=1; }
    [ "$bad" = 0 ]
}

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

# Validates one rank's Engram shards in scripts/pack_engram.py's format.
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

_ASSETS_CHECK='
import os, sys
assets, cache, native = sys.argv[1], sys.argv[2], sys.argv[3] == "native"
bad = []
need = [os.path.join(assets, "rope.bin"), os.path.join(assets, "engram-host.bin"), os.path.join(cache, "pfdense-table.json")]
if native:
    need.append(os.path.join(cache, "dsv41-bias-vl", "bias_vl.safetensors"))
for p in need:
    if not os.path.isfile(p) or os.path.getsize(p) == 0:
        bad.append(p + " missing or empty")
aot = os.path.join(assets, "aot")
listing = []
for root, dirs, names in os.walk(assets):
    for name in names:
        p = os.path.join(root, name)
        listing.append(f"{os.path.relpath(p, assets)} {os.path.getsize(p)}")
if not any(l.startswith("aot/") for l in listing):
    bad.append(aot + " missing or empty")
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
print("\n".join(sorted(listing)))
'

_files_ready() {
    local m e a c img h_list w_list e0 e1 a0 a1
    m="$(_cfg MODEL_DIR)"; e="$(_cfg ENGRAM_DIR)"; a="$(_cfg ASSETS_DIR)"; c="$(_cfg CACHE_DIR)"; img="$(_cfg TF_DSV41_IMAGES)"
    h_list="$(python3 -c "$_MODEL_CHECK" "$m")" \
        || { echo "adapter: the model at $m on the head is not complete (docs/ZIG-SERVE.md step 1)" >&2; return 1; }
    w_list="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "python3 -c $(printf '%q' "$_MODEL_CHECK") '$m'")" \
        || { echo "adapter: the model at $m on the worker is not complete (docs/ZIG-SERVE.md step 1)" >&2; return 1; }
    [ "$h_list" = "$w_list" ] \
        || { echo "adapter: the model at $m differs between the head and the worker: files or sizes" >&2; return 1; }
    e0="$(python3 -c "$_ENGRAM_CHECK" "$e" 0)" \
        || { echo "adapter: the rank 0 Engram shards at $e on the head are not complete" >&2; return 1; }
    e1="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "python3 -c $(printf '%q' "$_ENGRAM_CHECK") '$e' 1")" \
        || { echo "adapter: the rank 1 Engram shards at $e on the worker are not complete" >&2; return 1; }
    local layer lo0 hi0 t0 lo1 hi1 t1
    while read -r layer lo0 hi0 t0 && read -r _ lo1 hi1 t1 <&3; do
        [ "$lo0" = 0 ] && [ "$lo1" = "$hi0" ] && [ "$hi1" = "$t0" ] && [ "$t1" = "$t0" ] \
            || { echo "adapter: Engram layer $layer: rank 0 rows $lo0..$hi0 of $t0, rank 1 rows $lo1..$hi1 of $t1" >&2; return 1; }
    done <<<"$e0" 3<<<"$e1"
    a0="$(python3 -c "$_ASSETS_CHECK" "$a" "$c" "$img")" \
        || { echo "adapter: the assets at $a / cache at $c on the head are not complete (docs/ZIG-SERVE.md, Generate runtime assets)" >&2; return 1; }
    a1="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "python3 -c $(printf '%q' "$_ASSETS_CHECK") '$a' '$c' '$img'")" \
        || { echo "adapter: the assets at $a / cache at $c on the worker are not complete (docs/ZIG-SERVE.md, Generate runtime assets)" >&2; return 1; }
    [ "$a0" = "$a1" ] \
        || { echo "adapter: the assets at $a differ between the head and the worker" >&2; return 1; }
}

_image_ready() {
    local image want fmt h w hr wr
    image="$(_cfg IMAGE)"; want="$(cat "$RECIPE_DIR/config/tensorfold.rev")"
    fmt='{"config":{{json .Config}},"layers":{{json .RootFS.Layers}},"os":{{json .Os}},"architecture":{{json .Architecture}},"variant":{{json .Variant}}}'
    hr="$(docker image inspect -f '{{index .Config.Labels "org.tensorfold.revision"}}' "$image" 2>/dev/null)" \
        || { echo "adapter: image $image is not on the head (docs/ZIG-SERVE.md, Build)" >&2; return 1; }
    wr="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
        "docker image inspect -f '{{index .Config.Labels \"org.tensorfold.revision\"}}' '$image'" 2>/dev/null)" \
        || { echo "adapter: image $image is not on the worker (docs/ZIG-SERVE.md, Build: docker load it there)" >&2; return 1; }
    [ "$hr" = "$want" ] && [ "$wr" = "$want" ] \
        || { echo "adapter: image engine revision head ${hr:-?}, worker ${wr:-?}; config/tensorfold.rev is $want (rebuild after updating the checkout)" >&2; return 1; }
    h="$(docker image inspect -f "$fmt" "$image" | sha256sum)"
    w="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "docker image inspect -f $(printf '%q' "$fmt") '$image'" | sha256sum)"
    [ "$h" = "$w" ] || { echo "adapter: image $image differs between the head and the worker (docs/ZIG-SERVE.md, Build: docker save on the head, docker load on the worker)" >&2; return 1; }
}

adapter_prepare() {
    [ -f "$(_config)" ] || { echo "adapter: $(_config) is missing (cp config/prod-zig.env.example config/prod-zig.env)" >&2; return 1; }
    if grep -qE '^[A-Z_]+="?[^"#]*<' "$(_config)"; then
        echo "adapter: $(_config) still has <placeholders>" >&2; return 1
    fi
    if [ -z "${HEAD_IF:-}" ] || [ -z "${WORKER_IB:-}" ]; then
        wait_for_fabric || { echo "adapter: fabric facts not derived (model-runner's wait_for_fabric)" >&2; return 1; }
    fi
    _identity_ok || return 1
    _image_ready || return 1
    _files_ready || return 1
}

# _running <container> [w]: true if it is running on the head, or on the worker with "w".
_running() {
    local q="docker inspect -f '{{.State.Running}}' '$1' 2>/dev/null"
    if [ "${2:-}" = w ]; then
        [ "$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$q")" = true ]
    else
        [ "$(eval "$q")" = true ]
    fi
}

adapter_start() {
    _identity_ok || return 1
    local state deadline reply stop auth cmd0 cmd1
    state="$(_cfg STATE_DIR)"
    stop="$(_cfg STOP_FILE)"; stop="${stop:-$state/native-ops/STOP}"
    [ ! -e "$stop" ] || { echo "adapter: durable STOP is set ($stop); remove it to start" >&2; return 1; }
    auth="$(_cfg NATIVE_AUTH_FILE)"
    cmd1="$(_run_cmd 1)" || { echo "adapter: the rank 1 command could not be built from $(_config)" >&2; return 1; }
    cmd0="$(_run_cmd 0)" || { echo "adapter: the rank 0 command could not be built from $(_config)" >&2; return 1; }
    mkdir -p "$(_cfg PREPARED_DIR)" "$state/sessions" || return 1
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "mkdir -p '$(_cfg PREPARED_DIR)' '$state/sessions' && $cmd1" >/dev/null \
        || { echo "adapter: rank 1 did not start on the worker" >&2; return 1; }
    eval "$cmd0" >/dev/null || { echo "adapter: rank 0 did not start" >&2; return 1; }
    deadline=$(( $(date +%s) + ADAPTER_READY_TIMEOUT ))
    local probe=/v1/models
    [ -z "$auth" ] || probe=/health
    until curl -fsS -m 5 -o /dev/null "http://127.0.0.1:${ADAPTER_PORT}${probe}"; do
        _running "$ADAPTER_HEAD_CONTAINER" || { echo "adapter: rank 0 exited during the load (docker logs $ADAPTER_HEAD_CONTAINER)" >&2; return 1; }
        _running "$ADAPTER_WORKER_CONTAINER" w || { echo "adapter: rank 1 exited during the load (docker logs $ADAPTER_WORKER_CONTAINER on the worker)" >&2; return 1; }
        [ "$(date +%s)" -lt "$deadline" ] || { echo "adapter: not serving after ${ADAPTER_READY_TIMEOUT}s" >&2; return 1; }
        sleep 5
    done
    # Serving keys are not diagnostic credentials: with native auth, /health is the readiness check.
    [ -z "$auth" ] || return 0
    reply="$(curl -fsS -m 300 "http://127.0.0.1:${ADAPTER_PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$ADAPTER_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23?\"}],\"temperature\":0,\"reasoning_effort\":\"low\"}")" \
        || { echo "adapter: the check request failed" >&2; return 1; }
    python3 -c 'import json, sys; c = json.loads(sys.argv[1])["choices"][0]["message"]["content"] or ""; sys.exit(0 if "391" in c else 1)' "$reply" \
        || { echo "adapter: the check reply does not read 391: $reply" >&2; return 1; }
}

# SIGTERM to rank 0 drains requests and stops both ranks; logs are kept in STATE_DIR/logs on each node (newest 10).
adapter_stop() {
    local logs t drain
    logs="${ADAPTER_LOG_DIR:-$(_cfg STATE_DIR)/logs}"
    t="$(date -u +%Y%m%dT%H%M%SZ)"
    drain="$(_cfg TF_DRAIN_S 2>/dev/null)"; drain="${drain:-20}"
    local save='mkdir -p "$1" && docker logs "$2" > "$1/$2-$3.log" 2>&1; ls -1t "$1/$2"-*.log 2>/dev/null | tail -n +11 | xargs -r rm -f'
    docker stop -t "$(( drain + 60 ))" "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
        "timeout 60 docker wait '$ADAPTER_WORKER_CONTAINER'" >/dev/null 2>&1 || true
    bash -c "$save" _ "$logs" "$ADAPTER_HEAD_CONTAINER" "$t" >/dev/null 2>&1 || true
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
        "bash -c $(printf '%q' "$save") _ '$logs' '$ADAPTER_WORKER_CONTAINER' '$t'" >/dev/null 2>&1 || true
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "docker rm -f '$ADAPTER_WORKER_CONTAINER'" >/dev/null 2>&1 || true
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
