#!/usr/bin/env bash
# Adapter: Qwen3.8-Flash-Next, NVIDIA FP4-Local, vLLM b12x, one DGX Spark (eugr/spark-vllm-docker recipes/qwen3.8-flash-next-nvfp4-solo.yaml).
# B12X_AUTOTUNE=0 skips a multi-hour kernel sweep; B12X_COMPILE_WORKERS=1 avoids running out of memory while compiling
# (both from vcruz305/Qwen3.8-Flash-Next-FP4-Spark-recipe).

ADAPTER_TOPOLOGY=single
ADAPTER_ID="qwen3.8-flash-next-nvfp4-solo"
ADAPTER_RECIPE="qwen3.8-flash-next-nvfp4-solo"
ADAPTER_PORT=8888
ADAPTER_MODEL_ID="Qwen3.8-Flash-Next"
ADAPTER_HEAD_CONTAINER="vllm_node"
ADAPTER_CHECKPOINT="nvidia/Qwen3.8-Flash-Next-FP4-Local"
ADAPTER_REVISION="b9ce18a6887e18c089fdea8509a33e63429a3300"
ADAPTER_MAX_MODEL_LEN=262144
ADAPTER_EXTRA_ARGS=()
ADAPTER_ENV=(B12X_AUTOTUNE=0 B12X_COMPILE_WORKERS=1)
# size<TAB>file at ADAPTER_REVISION
ADAPTER_MANIFEST="$(cat <<'MANIFEST'
166	.gitattributes
29021	README.md
8952	chat_template.jinja
75217	config.json
202	generation_config.json
3353259	merges.txt
47280974	model.safetensors.index.json
390	preprocessor_config.json
5000643504	tensors/model-00001-of-00021.safetensors
5002792096	tensors/model-00002-of-00021.safetensors
5002927400	tensors/model-00003-of-00021.safetensors
5003194672	tensors/model-00004-of-00021.safetensors
5003151448	tensors/model-00005-of-00021.safetensors
5002892148	tensors/model-00006-of-00021.safetensors
5002907708	tensors/model-00007-of-00021.safetensors
5003103756	tensors/model-00008-of-00021.safetensors
5002976316	tensors/model-00009-of-00021.safetensors
5002834136	tensors/model-00010-of-00021.safetensors
5002685828	tensors/model-00011-of-00021.safetensors
5002912920	tensors/model-00012-of-00021.safetensors
5003237060	tensors/model-00013-of-00021.safetensors
4845006312	tensors/model-00014-of-00021.safetensors
4860574920	tensors/model-00015-of-00021.safetensors
4950031028	tensors/model-00016-of-00021.safetensors
4950030896	tensors/model-00017-of-00021.safetensors
4950030896	tensors/model-00018-of-00021.safetensors
4950030904	tensors/model-00019-of-00021.safetensors
4950030904	tensors/model-00020-of-00021.safetensors
2700016864	tensors/model-00021-of-00021.safetensors
12809320	tokenizer.json
17928	tokenizer_config.json
385	video_preprocessor_config.json
6722759	vocab.json
MANIFEST
)"
# ---- common body: identical in every eugr/spark-vllm-docker adapter ----
# One checkout pinned at EUGR_COMMIT serves all of them (RECIPE_DIR is a symlink to it).
# The launcher runs under env -i with a --config file written here, so the checkout's .env is never read.
# Checkpoints are read from $HOME/.cache/huggingface on every node.

EUGR_COMMIT=0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d
EUGR_IMAGE=vllm-node-b12x
EUGR_WAIT_TIMEOUT="${EUGR_WAIT_TIMEOUT:-3600}"

_eugr_err() { echo "adapter: $*" >&2; }

_eugr_hf_hub() { printf '%s/.cache/huggingface/hub' "$HOME"; }

_eugr_config_file() { printf '%s/eugr-%s.env' "${STATE:?model-runner STATE is not set}" "$ADAPTER_ID"; }

# Prints the recipe to run: the checkout's own, or a copy in $STATE with only the model changed to ADAPTER_CHECKPOINT.
_eugr_recipe_ref() {
    local src="$RECIPE_DIR/recipes/$ADAPTER_RECIPE.yaml" orig dst n
    [ "$(grep -c '^model:' "$src")" -eq 1 ] || { _eugr_err "$src has $(grep -c '^model:' "$src") model: lines, expected 1"; return 1; }
    orig="$(sed -n 's/^model:[[:space:]]*//p' "$src")"
    if [ "$orig" = "$ADAPTER_CHECKPOINT" ]; then printf '%s\n' "$ADAPTER_RECIPE"; return 0; fi
    [ "$(grep -cxF "model: $orig" "$src")" -eq 1 ] && [ "$(grep -cF "vllm serve $orig " "$src")" -eq 1 ] \
        || { _eugr_err "$src does not have exactly one 'model: $orig' line and one 'vllm serve $orig ' line"; return 1; }
    dst="${STATE:?model-runner STATE is not set}/eugr-$ADAPTER_ID.yaml"
    awk -v o="$orig" -v n="$ADAPTER_CHECKPOINT" '
        $0 == "model: " o { print "model: " n; next }
        index($0, "vllm serve " o " ") { sub("vllm serve " o " ", "vllm serve " n " "); print; next }
        { print }' "$src" > "$dst.tmp.$$" || { rm -f "$dst.tmp.$$"; return 1; }
    n="$(diff "$src" "$dst.tmp.$$" | grep -c '^>')"
    [ "$n" -eq 2 ] || { rm -f "$dst.tmp.$$"; _eugr_err "derived recipe changes $n lines of $src, expected exactly the model: and vllm serve lines"; return 1; }
    mv -f "$dst.tmp.$$" "$dst"; printf '%s\n' "$dst"
}

_eugr_write_config() {
    local f tmp
    f="$(_eugr_config_file)"; tmp="$f.tmp.$$"
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then
        printf 'CLUSTER_NODES=%s,%s\nETH_IF=%s\nIB_IF=%s\nLOCAL_IP=%s\n' "$HEAD_IP" "$WORKER_IP" "$HEAD_IF" "$HEAD_IB" "$HEAD_IP" > "$tmp"
    else
        : > "$tmp"
    fi || { rm -f "$tmp"; _eugr_err "cannot write $tmp"; return 1; }
    mv -f "$tmp" "$f"
}

_eugr_args() {
    local e
    local ref; ref="$(_eugr_recipe_ref)" || return 1
    EUGR_ARGS=("$ref" --config "$(_eugr_config_file)" --name "$ADAPTER_HEAD_CONTAINER"
               --port "$ADAPTER_PORT" --max-model-len "$ADAPTER_MAX_MODEL_LEN" -d)
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then
        EUGR_ARGS+=(--nodes "$HEAD_IP,$WORKER_IP" --eth-if "$HEAD_IF" --ib-if "$HEAD_IB")
    else
        EUGR_ARGS+=(--solo)
    fi
    EUGR_ARGS+=(--env HF_HUB_OFFLINE=1)
    for e in "${ADAPTER_ENV[@]}"; do EUGR_ARGS+=(--env "$e"); done
    EUGR_ARGS+=(-- --served-model-name "$ADAPTER_MODEL_ID" --revision "$ADAPTER_REVISION" "${ADAPTER_EXTRA_ARGS[@]}")
}

_eugr_run() {
    _eugr_args || return 1
    ( cd "$RECIPE_DIR" && env -i PATH="$PATH" HOME="$HOME" USER="$(id -un)" LOGNAME="$(id -un)" \
        ./run-recipe.sh "${EUGR_ARGS[@]:0:1}" "$@" "${EUGR_ARGS[@]:1}" )
}

_eugr_ssh() { timeout -k 2 "$1" ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "${@:2}"; }

# Checks the snapshot on a node against ADAPTER_MANIFEST (read from stdin).
_eugr_check_py() {
    cat <<'PY'
import json, os, sys
snap, manifest = sys.argv[1], {}
for line in sys.stdin.read().splitlines():
    if line.strip():
        size, name = line.split("\t", 1); manifest[name] = int(size)
if not os.path.isdir(snap): sys.exit(f"no snapshot directory {snap}")
bad = []
for name, size in sorted(manifest.items()):
    p = os.path.join(snap, name)
    if not os.path.isfile(p): bad.append(f"missing {name}")
    elif os.path.getsize(p) != size: bad.append(f"{name} is {os.path.getsize(p)} bytes, manifest {size}")
if bad: sys.exit("; ".join(bad[:6]) + (f" (+{len(bad) - 6} more)" if len(bad) > 6 else ""))
try:
    json.load(open(os.path.join(snap, "config.json")))
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
except Exception as e: sys.exit(f"config.json or the index does not parse: {e}")
shards = sorted(set(idx.get("weight_map", {}).values()))
missing = [s for s in shards if s not in manifest]
if not shards or missing: sys.exit("the index names shards outside the manifest: " + ", ".join(missing[:5]) if missing else "the index lists no shards")
print(f"{len(manifest)} files, {sum(manifest.values()) / 2**30:.1f} GiB, {len(shards)} shards in the index")
PY
}

# _eugr_check_checkpoint <label> <remote 0|1>
_eugr_check_checkpoint() {
    local snap script out
    snap="$(_eugr_hf_hub)/models--${ADAPTER_CHECKPOINT//\//--}/snapshots/$ADAPTER_REVISION"
    script="$(_eugr_check_py)"
    if [ "$2" = 1 ]; then
        out="$(printf '%s\n' "$ADAPTER_MANIFEST" | _eugr_ssh 120 python3 -I -c "$(printf '%q' "$script")" "$(printf '%q' "$snap")" 2>&1)"
    else
        out="$(printf '%s\n' "$ADAPTER_MANIFEST" | python3 -I -c "$script" "$snap" 2>&1)"
    fi || { _eugr_err "$1: checkpoint $ADAPTER_CHECKPOINT@$ADAPTER_REVISION is not ready at $snap: $out"; return 1; }
    echo "adapter: $1 checkpoint OK: $out" >&2
}

# The nodes' images must share an Id, or the same content per docker/image_identity.py (save/load can change the Id).
_eugr_check_image() {
    local hid wid hfp wfp
    hid="$(docker image inspect -f '{{.Id}}' "$EUGR_IMAGE" 2>/dev/null)" && [ -n "$hid" ] \
        || { _eugr_err "head: image $EUGR_IMAGE is missing (prepare: ./run-recipe.sh $ADAPTER_RECIPE --setup)"; return 1; }
    EUGR_IMAGE_ID="$hid"
    [ "$ADAPTER_TOPOLOGY" = pair ] || return 0
    wid="$(_eugr_ssh 30 docker image inspect -f "'{{.Id}}'" "$EUGR_IMAGE" 2>/dev/null)" && [ -n "$wid" ] \
        || { _eugr_err "worker $WORKER_SSH: image $EUGR_IMAGE is missing"; return 1; }
    [ "$wid" = "$hid" ] && return 0
    EUGR_IMAGE_ID="head $hid, worker $wid"
    hfp="$(docker image inspect "$hid" | python3 -I "$RECIPE_DIR/docker/image_identity.py" 2>/dev/null)" || hfp=""
    wfp="$(_eugr_ssh 30 docker image inspect "$wid" 2>/dev/null | python3 -I "$RECIPE_DIR/docker/image_identity.py" 2>/dev/null)" || wfp=""
    if [ -n "$hfp" ] && [ "$hfp" = "$wfp" ]; then EUGR_IMAGE_ID="$EUGR_IMAGE_ID (same content $hfp)"; return 0; fi
    _eugr_err "image $EUGR_IMAGE differs: head $hid, worker $wid"; return 1
}

adapter_prepare() {
    local head st out
    [ -x "$RECIPE_DIR/run-recipe.sh" ] || { _eugr_err "$RECIPE_DIR/run-recipe.sh is missing"; return 1; }
    [ -f "$RECIPE_DIR/recipes/$ADAPTER_RECIPE.yaml" ] || { _eugr_err "no recipe $ADAPTER_RECIPE in $RECIPE_DIR"; return 1; }
    head="$(git -C "$RECIPE_DIR" rev-parse HEAD 2>/dev/null)" || head=""
    [ "$head" = "$EUGR_COMMIT" ] || { _eugr_err "$RECIPE_DIR is at ${head:-unknown}, not the reviewed $EUGR_COMMIT"; return 1; }
    st="$(git -C "$RECIPE_DIR" status --porcelain --untracked-files=no 2>/dev/null)" || st="?"
    [ -z "$st" ] || { _eugr_err "$RECIPE_DIR has tracked modifications: $st"; return 1; }
    env -i PATH="$PATH" HOME="$HOME" python3 -c 'import yaml' 2>/dev/null \
        || { _eugr_err "PyYAML is missing for python3 (run-recipe.sh would pip-install it); install python3-yaml first"; return 1; }
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then
        # The launcher connects to the worker as this account.
        case "$WORKER_SSH" in
            "$WORKER_IP"|"$(id -un)@$WORKER_IP") ;;
            *) _eugr_err "WORKER_SSH is $WORKER_SSH; the launcher connects as $(id -un)@$WORKER_IP"; return 1 ;;
        esac
        [ -n "${HEAD_IF:-}" ] || derive "$(( $(date +%s) + 30 ))" || { _eugr_err "fabric facts not derived"; return 1; }
        [ "$HEAD_IF" = "$WORKER_IF" ] && [ "$HEAD_IB" = "$WORKER_IB" ] \
            || { _eugr_err "the nodes' fabric differs (head $HEAD_IF/$HEAD_IB, worker $WORKER_IF/$WORKER_IB); this launcher takes one of each"; return 1; }
    fi
    _eugr_check_checkpoint head 0 || return 1
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then _eugr_check_checkpoint "worker $WORKER_SSH" 1 || return 1; fi
    _eugr_check_image || return 1
    _eugr_write_config || return 1
    out="$(_eugr_run --dry-run 2>&1)" || { printf '%s\n' "$out" | tail -20 >&2; _eugr_err "the launcher's dry run failed"; return 1; }
    local want
    for want in "vllm serve $ADAPTER_CHECKPOINT " "--port $ADAPTER_PORT" "--max-model-len $ADAPTER_MAX_MODEL_LEN" \
                "--served-model-name $ADAPTER_MODEL_ID" "--revision $ADAPTER_REVISION" "${ADAPTER_EXTRA_ARGS[*]}"; do
        grep -qF -- "$want" <<<"$out" || { printf '%s\n' "$out" | tail -20 >&2; _eugr_err "the dry-run command lacks '$want'"; return 1; }
    done
}

# _eugr_running <remote 0|1>: true if the container is running a vllm serve process.
_eugr_running() {
    local top
    if [ "$1" = 1 ]; then
        top="$(_eugr_ssh 30 docker top "$ADAPTER_WORKER_CONTAINER" -eo args 2>/dev/null)" || return 1
    else
        top="$(timeout -k 2 30 docker top "$ADAPTER_HEAD_CONTAINER" -eo args 2>/dev/null)" || return 1
    fi
    grep -q 'vllm serve' <<<"$top"
}

adapter_start() {
    local deadline t0
    _eugr_check_image || return 1
    _eugr_write_config || return 1
    echo "adapter: $ADAPTER_ID ($ADAPTER_RECIPE): $ADAPTER_CHECKPOINT@$ADAPTER_REVISION, image $EUGR_IMAGE $EUGR_IMAGE_ID, repo digests $(docker image inspect -f '{{join .RepoDigests ","}}' "$EUGR_IMAGE" 2>/dev/null || echo none)" >&2
    _eugr_run || { _eugr_err "the launcher returned non-zero"; return 1; }
    t0="$(date +%s)"; deadline=$(( t0 + EUGR_WAIT_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        adapter_health 2>/dev/null && { echo "adapter: /health ready after $(( $(date +%s) - t0 )) s" >&2; return 0; }
        # vllm serve runs under docker exec -d, so the container stays up if it dies.
        if [ $(( $(date +%s) - t0 )) -ge 60 ]; then
            _eugr_running 0 || { _eugr_err "vllm serve is not running in $ADAPTER_HEAD_CONTAINER on the head"; return 1; }
            if [ "$ADAPTER_TOPOLOGY" = pair ]; then
                _eugr_running 1 || { _eugr_err "vllm serve is not running in $ADAPTER_WORKER_CONTAINER on $WORKER_SSH"; return 1; }
            fi
        fi
        sleep 10
    done
    _eugr_err "/health not ready after $EUGR_WAIT_TIMEOUT s"; return 1
}

adapter_stop() {
    local rc=0
    timeout -k 2 120 docker stop -t 90 "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || rc=1
    timeout -k 2 30 docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then
        _eugr_ssh 120 docker stop -t 90 "$ADAPTER_WORKER_CONTAINER" >/dev/null 2>&1 || rc=1
        _eugr_ssh 30 docker rm -f "$ADAPTER_WORKER_CONTAINER" >/dev/null 2>&1 || true
    fi
    return "$rc"
}

adapter_health() {
    curl -fsS -m 10 -o /dev/null "http://127.0.0.1:${ADAPTER_PORT}/health"
}

adapter_describe() {
    printf 'dir    : %s (eugr/spark-vllm-docker %s, recipe %s)\n' "$RECIPE_DIR" "${EUGR_COMMIT:0:7}" "$ADAPTER_RECIPE"
    printf 'model  : %s (%s@%s)\n' "$ADAPTER_MODEL_ID" "$ADAPTER_CHECKPOINT" "${ADAPTER_REVISION:0:8}"
    printf 'api    : http://127.0.0.1:%s/v1\n' "$ADAPTER_PORT"
    printf 'head   : %s\n' "$ADAPTER_HEAD_CONTAINER"
    [ "$ADAPTER_TOPOLOGY" = pair ] && printf 'worker : %s on %s\n' "$ADAPTER_WORKER_CONTAINER" "$WORKER_SSH"
    return 0
}
