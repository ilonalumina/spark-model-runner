#!/usr/bin/env bash
# Adapter: GLM-5.3-Flash, NVIDIA FP4-Local (QAD), vLLM b12x, 2x DGX Spark (TP2).
# Recipe: eugr/spark-vllm-docker recipes/glm-5.3-flash.yaml (the launcher NVIDIA's model card names).
#
# Window: 1,048,576 (the model's maximum; the recipe's default is 500,000). A target, not a proven fit: the first
# load's KV pool and a full-window request decide it, and a load that cannot hold it is reported to QP with the
# measured limit; the window is not lowered here. Recipe env as published (B12X_AUTOTUNE / COMPILE_WORKERS are
# measured only at TP1).

ADAPTER_TOPOLOGY=pair
ADAPTER_RECIPE="glm-5.3-flash"
ADAPTER_PORT=8888
ADAPTER_MODEL_ID="GLM-5.3-Flash"
ADAPTER_HEAD_CONTAINER="vllm_node"
ADAPTER_WORKER_CONTAINER="vllm_node"
ADAPTER_CHECKPOINT="nvidia/GLM-5.3-Flash-FP4-Local"
ADAPTER_REVISION="76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"
ADAPTER_MAX_MODEL_LEN=1048576
ADAPTER_ENV=()
# size<TAB>file at ADAPTER_REVISION (Hub API, 2026-10-18)
ADAPTER_MANIFEST="$(cat <<'MANIFEST'
166	.gitattributes
30835	README.md
10950	chat_template.jinja
83897	config.json
194	generation_config.json
71988	hf_quant_config.json
4294380516	model-00001-of-00040.safetensors
398331164	model-00002-of-00040.safetensors
4187354632	model-00003-of-00040.safetensors
4059621056	model-00004-of-00040.safetensors
1204588600	model-00005-of-00040.safetensors
3386693718	model-00006-of-00040.safetensors
3386630980	model-00007-of-00040.safetensors
3386784380	model-00008-of-00040.safetensors
3386714856	model-00009-of-00040.safetensors
3386642180	model-00010-of-00040.safetensors
3386705796	model-00011-of-00040.safetensors
3386704188	model-00012-of-00040.safetensors
3386634084	model-00013-of-00040.safetensors
3386785778	model-00014-of-00040.safetensors
5080037446	model-00015-of-00040.safetensors
5080062512	model-00016-of-00040.safetensors
5080033556	model-00017-of-00040.safetensors
5080002106	model-00018-of-00040.safetensors
5080044836	model-00019-of-00040.safetensors
5080015448	model-00020-of-00040.safetensors
5080033720	model-00021-of-00040.safetensors
5080007626	model-00022-of-00040.safetensors
5080036888	model-00023-of-00040.safetensors
5079998124	model-00024-of-00040.safetensors
5080015080	model-00025-of-00040.safetensors
5080002452	model-00026-of-00040.safetensors
5080025960	model-00027-of-00040.safetensors
5080010694	model-00028-of-00040.safetensors
5080031820	model-00029-of-00040.safetensors
5080122474	model-00030-of-00040.safetensors
5080041150	model-00031-of-00040.safetensors
5080082340	model-00032-of-00040.safetensors
5080023370	model-00033-of-00040.safetensors
5079989500	model-00034-of-00040.safetensors
5080032548	model-00035-of-00040.safetensors
5080044714	model-00036-of-00040.safetensors
5080045746	model-00037-of-00040.safetensors
5080051898	model-00038-of-00040.safetensors
5080035462	model-00039-of-00040.safetensors
5080022666	model-00040-of-00040.safetensors
4726696	model-inputscales.safetensors
35266806	model.safetensors.index.json
909	processor_config.json
753692264	tensors/native-QB.safetensors
20217442	tokenizer.json
761	tokenizer_config.json
MANIFEST
)"

# ---- common body (identical in every eugr b12x adapter; tests/check_bodies.sh compares them) ----
#
# Launcher: eugr/spark-vllm-docker (run-recipe.sh -> run-recipe.py -> launch-cluster.sh), pinned at EUGR_COMMIT.
# One checkout serves every eugr adapter; RECIPE_DIR is a symlink to it. Moving the pin re-checks all of them.
#
# Isolation from saved configuration: the launcher reads the checkout's .env (CLUSTER_NODES, CLUSTER_LINKS,
# CONTAINER_NAME, LOCAL_IP, CONTAINER_* into the container) and inherited DOTENV_* variables. Every launcher
# call here runs under `env -i` and with --config naming a file this adapter writes, holding only the node and
# fabric identity model-runner derived. --config replaces the checkout's .env in both layers
# (run-recipe.py:1019 -> launch-cluster.sh --config, :246/:275).
#
# The HF cache: under env -i HF_HOME is unset, so the launcher mounts $HOME/.cache/huggingface on every node
# (launch-cluster.sh:6, the same path string on the worker). The checkpoint gate checks exactly that path.
#
# Stop never calls the launcher: docker stop / rm -f of the recorded container on the recorded hosts.

EUGR_COMMIT=0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d
EUGR_IMAGE=vllm-node-b12x
EUGR_WAIT_TIMEOUT="${EUGR_WAIT_TIMEOUT:-3600}"

_eugr_err() { echo "adapter: $*" >&2; }

_eugr_hf_hub() { printf '%s/.cache/huggingface/hub' "$HOME"; }

_eugr_config_file() { printf '%s/eugr-%s.env' "${STATE:?model-runner STATE is not set}" "$ADAPTER_RECIPE"; }

# The launcher's config for this launch: node and fabric identity only. Written atomically.
_eugr_write_config() {
    local f tmp
    f="$(_eugr_config_file)"; tmp="$f.tmp.$$"
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then
        # LOCAL_IP: the head's fabric address as model-runner states it (otherwise the launcher reads it off ETH_IF)
        printf 'CLUSTER_NODES=%s,%s\nETH_IF=%s\nIB_IF=%s\nLOCAL_IP=%s\n' "$HEAD_IP" "$WORKER_IP" "$HEAD_IF" "$HEAD_IB" "$HEAD_IP" > "$tmp"
    else
        : > "$tmp"
    fi || { rm -f "$tmp"; _eugr_err "cannot write $tmp"; return 1; }
    mv -f "$tmp" "$f"
}

# The launch command, arguments only (the caller adds --dry-run or nothing).
_eugr_args() {
    local e
    EUGR_ARGS=("$ADAPTER_RECIPE" --config "$(_eugr_config_file)" --name "$ADAPTER_HEAD_CONTAINER"
               --port "$ADAPTER_PORT" --max-model-len "$ADAPTER_MAX_MODEL_LEN" -d)
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then
        EUGR_ARGS+=(--nodes "$HEAD_IP,$WORKER_IP" --eth-if "$HEAD_IF" --ib-if "$HEAD_IB")
    else
        EUGR_ARGS+=(--solo)
    fi
    EUGR_ARGS+=(--env HF_HUB_OFFLINE=1)
    for e in "${ADAPTER_ENV[@]}"; do EUGR_ARGS+=(--env "$e"); done
    EUGR_ARGS+=(-- --served-model-name "$ADAPTER_MODEL_ID" --revision "$ADAPTER_REVISION")
}

_eugr_run() { # run-recipe.sh under an empty environment; extra args first
    _eugr_args
    ( cd "$RECIPE_DIR" && env -i PATH="$PATH" HOME="$HOME" USER="$(id -un)" LOGNAME="$(id -un)" \
        ./run-recipe.sh "${EUGR_ARGS[@]:0:1}" "$@" "${EUGR_ARGS[@]:1}" )
}

_eugr_ssh() { timeout -k 2 "$1" ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "${@:2}"; }

# The checkpoint gate, run with python3 -I on each node: snapshots/<revision>/ holds every manifest file at its
# manifest size; config.json and the index parse; every shard the index names is present and complete.
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

_eugr_check_checkpoint() { # <node label> <remote: 0|1>
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

# The image on each node: one Id, equal across nodes, or equal content by the launcher's own fingerprint
# (docker/image_identity.py: a docker save/load copy can carry a different Id under the containerd store).
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
    # run-recipe.sh installs PyYAML with pip when it is missing; prepare refuses instead.
    env -i PATH="$PATH" HOME="$HOME" python3 -c 'import yaml' 2>/dev/null \
        || { _eugr_err "PyYAML is missing for python3 (run-recipe.sh would pip-install it); install python3-yaml first"; return 1; }
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then
        # The launcher reaches the worker as `ssh <WORKER_IP>`, i.e. as this account.
        case "$WORKER_SSH" in
            "$WORKER_IP"|"$(id -un)@$WORKER_IP") ;;
            *) _eugr_err "WORKER_SSH is $WORKER_SSH; the launcher connects as $(id -un)@$WORKER_IP"; return 1 ;;
        esac
        [ -n "${HEAD_IF:-}" ] || derive "$(( $(date +%s) + 30 ))" || { _eugr_err "fabric facts not derived"; return 1; }
        # One --eth-if and one --ib-if serve every node.
        [ "$HEAD_IF" = "$WORKER_IF" ] && [ "$HEAD_IB" = "$WORKER_IB" ] \
            || { _eugr_err "the nodes' fabric differs (head $HEAD_IF/$HEAD_IB, worker $WORKER_IF/$WORKER_IB); this launcher takes one of each"; return 1; }
    fi
    _eugr_check_checkpoint head 0 || return 1
    if [ "$ADAPTER_TOPOLOGY" = pair ]; then _eugr_check_checkpoint "worker $WORKER_SSH" 1 || return 1; fi
    _eugr_check_image || return 1
    _eugr_write_config || return 1
    out="$(_eugr_run --dry-run 2>&1)" || { printf '%s\n' "$out" | tail -20 >&2; _eugr_err "the launcher's dry run failed"; return 1; }
    local want
    for want in "--port $ADAPTER_PORT" "--max-model-len $ADAPTER_MAX_MODEL_LEN" \
                "--served-model-name $ADAPTER_MODEL_ID" "--revision $ADAPTER_REVISION"; do
        grep -qF -- "$want" <<<"$out" || { printf '%s\n' "$out" | tail -20 >&2; _eugr_err "the dry-run command lacks '$want'"; return 1; }
    done
}

_eugr_running() { # <remote 0|1>: the container runs and holds a vllm serve process (host-side docker top, so
                  # the check does not depend on tools inside the image)
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
    echo "adapter: $ADAPTER_RECIPE: $ADAPTER_CHECKPOINT@$ADAPTER_REVISION, image $EUGR_IMAGE $EUGR_IMAGE_ID" >&2
    _eugr_run || { _eugr_err "the launcher returned non-zero"; return 1; }
    t0="$(date +%s)"; deadline=$(( t0 + EUGR_WAIT_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        adapter_health 2>/dev/null && { echo "adapter: /health ready after $(( $(date +%s) - t0 )) s" >&2; return 0; }
        # vllm serve runs under docker exec -d; a dead process leaves the container up, so check the process too.
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

# Recorded identity: the CLI has loaded ADAPTER_HEAD_CONTAINER / ADAPTER_WORKER_CONTAINER / WORKER_SSH from the
# launched record. No launcher, no configuration read.
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
