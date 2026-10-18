#!/usr/bin/env bash
# Adapter: MiMo-V2.6-Flash-RL (MXFP4 experts), SGLang TP=2/EP=2, 2x DGX Spark (MiaAI-Lab).
#
# Named for the recipe it drives: adapters/<model>/<recipe>.sh mirrors
# recipes/<model>/<recipe>/, and the CLI sets RECIPE_DIR from that name.
#
# CONTRACT (upstream start.sh @ 201be3e): the profile is ${ENV_FILE:-$ROOT/.env} in start.sh and
# stop.sh; every launcher call here exports ENV_FILE=<recipe>/.env, so an inherited ENV_FILE never
# reaches it. `set -euo pipefail`, `cd` to its own
# directory, `set -a; source .env; set +a` with NO capture of the environment
# first, then `${VAR:-default}`. So a key .env assigns wins over any export;
# a key it leaves alone falls to the environment, then to the default. The
# Qwen / V4.1 shape: this adapter computes the EFFECTIVE value of every
# identity, NFS and fabric key the launcher will run with (a child bash with
# the same options, seeded with the exports this adapter intends to make) and
# refuses naming the lines. The recipe's files are never written by this layer.
#
# Fabric: the launcher picks the RoCE port pair and both GID indices itself at
# every serve, from sysfs on both nodes (the first HCA pair in IB_HCA sharing
# a /24), so there is no GID to pin. What it cannot derive is the socket
# interface: one NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME serves both ranks, and
# those plus IB_HCA are verified against the derived facts at start.
#
# NFS: the worker mounts the head's checkpoint over NFSv4. With no explicit
# server address the launcher guesses from `ip route get` toward addresses
# that do not exist on this pair and can land on the LAN; the address must be
# stated, and it must be the head's fabric address.

ADAPTER_HEAD_CONTAINER="mimo26-tp2-head"
ADAPTER_WORKER_CONTAINER="mimo26-tp2-worker"
ADAPTER_PORT=8888

_KEYS="ENV_FILE PORT SERVED_MODEL_NAME HEAD_IP WORKER_IPS WORKER_HOSTS WORKER_USER SSH_IDENTITY HEAD_CTN WORKER_CTN WORKER_DIR WORKER_SSD_DIR NFS_SHARE NFS_SERVER_IP_1 NFS_CLIENTS MODEL_DIR EXPECTED_SHARDS IMAGE BASE_IMAGE IB_HCA NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME"

# _effective SEED=VALUE ... — print "KEY<TAB>value" for every key in _KEYS as
# the launcher would see it. Non-zero when .env does not source under the
# launcher's options.
_effective() {
    ( cd "$RECIPE_DIR" && env "$@" bash --noprofile --norc -c '
        set -euo pipefail
        # start.sh / stop.sh source ${ENV_FILE:-$ROOT/.env}; the seeds pin ENV_FILE to the recipe .env, and a
        # value .env assigns to ENV_FILE itself comes out below and is refused by _require.
        set -a
        source "$ENV_FILE" >/dev/null 2>&1
        set +a
        set +u
        # defaults, verbatim from start.sh / files/nfs-share.sh
        HEAD_IP="${HEAD_IP:-10.0.0.1}"
        WORKER_HOSTS="${WORKER_HOSTS:-$WORKER_IPS}"
        WORKER_USER="${WORKER_USER:-zurih}"
        SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/id_ed25519_shared}"
        GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enP7s7}"
        NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enP7s7}"
        IB_HCA="${IB_HCA:-rocep1s0f0,rocep1s0f1}"
        MODEL_DIR="${MODEL_DIR:-$HOME/NewModels/MiMo-V2.6-Flash-RL}"
        EXPECTED_SHARDS="${EXPECTED_SHARDS:-64}"
        BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5}"
        IMAGE="${IMAGE:-mimo26-spark:local}"
        HEAD_CTN="${HEAD_CTN:-mimo26-head}"
        WORKER_CTN="${WORKER_CTN:-mimo26-worker}"
        PORT="${PORT:-8888}"
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-v2.6-Flash}"
        NFS_SHARE="${NFS_SHARE:-1}"
        NFS_CLIENTS="${NFS_CLIENTS:-10.0.0.2,10.0.0.3,10.0.22.0/24,10.0.23.0/24,10.0.33.0/24}"
        WORKER_DIR="${WORKER_DIR:-/home/${WORKER_USER}/mimo26}"
        WORKER_SSD_DIR="${WORKER_SSD_DIR:-$WORKER_DIR/ssd}"
        # nfs_server_ip_for, worker index 0, explicit branches only, in its
        # order: NFS_SERVER_IPS, then NFS_SERVER_IP_1, then NFS_SERVER_IP.
        # Reported under the name .env.example uses.
        _nsi=""
        if [ -n "${NFS_SERVER_IPS:-}" ]; then _nsi="${NFS_SERVER_IPS%%,*}"; fi
        [ -n "$_nsi" ] || _nsi="${NFS_SERVER_IP_1:-${NFS_SERVER_IP:-}}"
        NFS_SERVER_IP_1="$_nsi"
        for k in '"$_KEYS"'; do printf "%s\t%s\n" "$k" "${!k}"; done' 2>/dev/null )
}

_load_effective() {
    local out k v rc=0
    out="$(_effective "$@")" || rc=$?
    [ "$rc" -eq 0 ] || return 3
    while IFS=$'\t' read -r k v; do
        [ -n "$k" ] && printf -v "_EFF_$k" '%s' "$v"
    done <<< "$out"
}

_identity_seeds() {
    printf '%s\n' "ENV_FILE=$RECIPE_DIR/.env" "HEAD_IP=$HEAD_IP" "WORKER_IPS=$WORKER_IP" "WORKER_HOSTS=${WORKER_SSH#*@}" \
        "WORKER_USER=${WORKER_SSH%@*}" "HEAD_CTN=$ADAPTER_HEAD_CONTAINER" "WORKER_CTN=$ADAPTER_WORKER_CONTAINER" \
        "PORT=$ADAPTER_PORT" "NFS_SHARE=1" "NFS_SERVER_IP_1=$HEAD_IP"
}

# The worker creates and mounts WORKER_DIR and WORKER_SSD_DIR as the ssh user
# (the SSD dir is mounted even at OFFLOAD_MODE=off); .env.example ships them
# under another user's home.
_worker_paths_ok() {
    local home="/home/${WORKER_SSH%@*}/" bad=0 k v
    for k in WORKER_DIR WORKER_SSD_DIR; do
        eval "v=\"\${_EFF_$k-}\""
        case "$v" in
            "$home"?*) ;;
            *) printf '  %s must be under %s\t(launcher would use %s)\n' "$k" "$home" "'$v'" >&2; bad=1 ;;
        esac
    done
    return "$bad"
}

_require() {
    local bad=0 kv k want have
    for kv in "$@"; do
        k="${kv%%=*}"; want="${kv#*=}"
        eval "have=\"\${_EFF_$k-}\""
        if [ "$have" != "$want" ]; then
            printf '  %s must be %s\t(launcher would use %s)\n' "$k" "'$want'" "'$have'" >&2
            bad=1
        fi
    done
    return "$bad"
}

# Comma-list membership, exact element match.
_in_list() {
    case ",$2," in *",$1,"*) return 0 ;; esac
    return 1
}

# Side-effect free. Identity, NFS, checkpoint, image and host prerequisites,
# all checked before `use` stops the running serve.
adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/.env" ]     || { echo "adapter: $RECIPE_DIR/.env is missing — cp .env.example .env and set the cluster block" >&2; return 1; }
    local seeds bad=0; mapfile -t seeds < <(_identity_seeds)
    _load_effective "${seeds[@]}" || { echo "adapter: $RECIPE_DIR/.env does not source under the launcher's 'set -euo pipefail'" >&2; return 1; }
    # NFS_SERVER_IP_1 stands for whichever of NFS_SERVER_IPS / NFS_SERVER_IP_1 /
    # NFS_SERVER_IP the launcher reads first.
    _require "${seeds[@]}" || bad=1
    _worker_paths_ok || bad=1
    _in_list "$WORKER_IP" "$_EFF_NFS_CLIENTS" \
        || { printf '  NFS_CLIENTS must list %s\t(launcher would use %s)\n' "$WORKER_IP" "'$_EFF_NFS_CLIENTS'" >&2; bad=1; }
    if [ "$bad" -ne 0 ]; then
        echo "adapter: $RECIPE_DIR/.env disagrees with the runner config / this adapter (lines above); set or delete those lines — nothing was stopped" >&2
        return 1
    fi

    # A missing checkpoint makes `serve` download ~171 GiB inside the unit.
    local d="$_EFF_MODEL_DIR" f n
    for f in config.json model.safetensors.index.json preprocessor_config.json \
             audio_tokenizer/model.safetensors model_mtp.safetensors; do
        [ -s "$d/$f" ] || { echo "adapter: $d/$f missing — ./start.sh download first" >&2; return 1; }
    done
    n="$(find "$d" -maxdepth 1 -name 'model_pp0_ep*_shard0.safetensors' 2>/dev/null | wc -l)"
    [ "$n" -ge "$_EFF_EXPECTED_SHARDS" ] || { echo "adapter: $d has $n of $_EFF_EXPECTED_SHARDS expert shards — ./start.sh download first" >&2; return 1; }

    # A stale or missing image makes `serve` build FROM the SGLang base and
    # ship it to the worker inside the unit's start budget.
    local label
    label="$(docker image inspect -f '{{index .Config.Labels "mimo26.base"}}' "$_EFF_IMAGE" 2>/dev/null || true)"
    [ "$label" = "$_EFF_BASE_IMAGE" ] \
        || { echo "adapter: image $_EFF_IMAGE is built from '${label:-nothing}', .env wants $_EFF_BASE_IMAGE — ./start.sh build first" >&2; return 1; }

    # Host tools the launcher calls: scripts/remote.py imports pexpect, and
    # nfs_rpc_ready runs rpcinfo; without either the worker path fails midway.
    python3 -c 'import pexpect' 2>/dev/null || { echo "adapter: python3 on the head cannot import pexpect (scripts/remote.py needs it)" >&2; return 1; }
    command -v rpcinfo >/dev/null     || { echo "adapter: rpcinfo is not on the head's PATH (files/nfs-share.sh needs it)" >&2; return 1; }
    [ -f "$_EFF_SSH_IDENTITY" ]       || { echo "adapter: SSH_IDENTITY $_EFF_SSH_IDENTITY does not exist — remote.py would fall back to a password" >&2; return 1; }

    ADAPTER_MODEL_ID="$_EFF_SERVED_MODEL_NAME"
}

# The launcher refuses while nvidia-smi lists a compute app, and sizes the KV
# pool from MemAvailable. After a switch the previous recipe's ranks are gone
# (the CLI verified that) but the driver may still be releasing memory. Same
# settle rule as the recipe's own stop.sh: no compute apps, then MemAvailable
# stable within 256 MiB across two reads.
_settle_head() {
    local deadline=$(( $(date +%s) + ${1:-180} )) last=0 now
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
            sleep 2; continue
        fi
        now="$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)"
        if [ "$last" -gt 0 ] && [ "$now" -ge "$last" ] && [ $(( now - last )) -lt 256 ]; then
            return 0
        fi
        last="$now"; sleep 2
    done
    echo "adapter: head GPU did not settle within ${1:-180}s" >&2
    return 1
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    local seeds bad=0 img label; mapfile -t seeds < <(_identity_seeds)
    seeds+=("NCCL_SOCKET_IFNAME=$HEAD_IF" "GLOO_SOCKET_IFNAME=$HEAD_IF")
    _load_effective "${seeds[@]}" || { echo "adapter: $RECIPE_DIR/.env does not source under the launcher's options" >&2; return 1; }
    _require "${seeds[@]}" || bad=1
    [ "$HEAD_IF" = "$WORKER_IF" ] \
        || { echo "  head interface $HEAD_IF != worker interface $WORKER_IF — one NCCL_SOCKET_IFNAME serves both ranks here" >&2; bad=1; }
    _in_list "$HEAD_IB" "$_EFF_IB_HCA"   || { printf '  IB_HCA must list %s\t(launcher would use %s)\n' "$HEAD_IB" "'$_EFF_IB_HCA'" >&2; bad=1; }
    _in_list "$WORKER_IB" "$_EFF_IB_HCA" || { printf '  IB_HCA must list %s\t(launcher would use %s)\n' "$WORKER_IB" "'$_EFF_IB_HCA'" >&2; bad=1; }
    if [ "$bad" -ne 0 ]; then
        echo "adapter: $RECIPE_DIR/.env disagrees with the derived fabric facts or the identity (lines above); set those lines or delete them so the exports apply, then relaunch" >&2
        return 1
    fi

    # The worker's image must carry the same base label, or serve rebuilds.
    img="$_EFF_IMAGE"
    label="$(ssh_bounded 20 "docker image inspect -f '{{index .Config.Labels \"mimo26.base\"}}' '$img' 2>/dev/null" || true)"
    [ "$label" = "$_EFF_BASE_IMAGE" ] \
        || { echo "adapter: image $img on the worker is built from '${label:-nothing}' — ./start.sh build ships it" >&2; return 1; }

    _settle_head 180 || return 1

    local kv; for kv in "${seeds[@]}"; do export "$kv"; done
    ./start.sh serve
}

# Stop by RECORDED identity first. The upstream stop sources today's .env for
# the worker and container names, so it runs only when that file still
# resolves to the launched identity. --wait-timeout keeps stop.sh's per-node
# GPU settle wait inside the unit's TimeoutStopSec.
adapter_stop() {
    local rc=0
    docker rm -f "$ADAPTER_HEAD_CONTAINER" >/dev/null 2>&1 || true
    ssh_bounded 30 "docker rm -f '$ADAPTER_WORKER_CONTAINER' >/dev/null 2>&1 || true" || rc=1
    if [ -f "$RECIPE_DIR/.env" ] && [ -x "$RECIPE_DIR/stop.sh" ]; then
        local seeds; mapfile -t seeds < <(_identity_seeds)
        if _load_effective "${seeds[@]}" && _require "${seeds[@]}" 2>/dev/null; then
            ( cd "$RECIPE_DIR" && for kv in "${seeds[@]}"; do export "$kv"; done && ./start.sh stop --wait-timeout 90 ) \
                || echo "adapter: upstream stop returned non-zero; containers were removed by recorded identity above" >&2
        else
            echo "adapter: upstream stop skipped — .env no longer resolves to the launched identity; containers were removed by recorded identity above" >&2
        fi
    fi
    return "$rc"
}

adapter_health() {
    curl -fsS -m 10 -o /dev/null "http://127.0.0.1:${ADAPTER_PORT}/health"
}

adapter_describe() {
    printf 'dir    : %s\n' "$RECIPE_DIR"
    printf 'model  : %s\n' "${ADAPTER_MODEL_ID:-MiMo-v2.6-Flash}"
    printf 'api    : http://127.0.0.1:%s/v1\n' "$ADAPTER_PORT"
    printf 'head   : %s\n' "$ADAPTER_HEAD_CONTAINER"
    printf 'worker : %s on %s\n' "$ADAPTER_WORKER_CONTAINER" "$WORKER_SSH"
}
