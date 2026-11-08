#!/usr/bin/env bash
set -u
ADAPTER=${ADAPTER:-adapters/DeepSeek-V4.1-Flash/deepseek-v41-tensorfold-spark-zig.sh}
RECIPE=${ZIG_RECIPE_DIR:?Set ZIG_RECIPE_DIR to the recipe checkout}

pass=0; fail=0
check() { if [ "$2" = 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL: $1"; fi; }
has() { case "$1" in *"$2"*) return 0;; *) return 1;; esac; }

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bin" "$work/state"

write_config() {
cat > "$work/prod-zig.env" <<EOF
IMAGE=qa/image:test
MODEL_DIR=/models/qa
ENGRAM_DIR=/engram
ASSETS_DIR=/assets
PREPARED_DIR=$work/prepared
STATE_DIR=$work/state
CACHE_DIR=/cache
TF_TP_MASTER=192.0.2.1
NCCL_SOCKET_IFNAME=enp1s0f0
NCCL_IB_HCA=mlx5_0
TF_DSV41_CONTEXT=32768
TF_DSV41_SLOTS=2
TF_DSV41_THINKING=0
TF_DSV41_DEFAULT_EFFORT=$1
TF_DRAIN_S=7
NATIVE_AUTH_FILE=$2
EOF
}

printf '#!/usr/bin/env bash\nprintf "docker %%s\\n" "$*" >> "%s/docker.log"\nexit 0\n' "$work" > "$work/bin/docker"
printf '#!/usr/bin/env bash\nprintf "ssh %%s\\n" "$*" >> "%s/ssh.log"\nexit 0\n' "$work" > "$work/bin/ssh"
printf '#!/usr/bin/env bash\nprintf "{\\"choices\\":[{\\"message\\":{\\"content\\":\\"391\\"}}]}"\nexit 0\n' > "$work/bin/curl"
chmod +x "$work/bin/docker" "$work/bin/ssh" "$work/bin/curl"

RECIPE_DIR="$RECIPE"; ADAPTER_HEAD_CONTAINER=qa-head; ADAPTER_WORKER_CONTAINER=qa-worker
export RECIPE_DIR ADAPTER_HEAD_CONTAINER ADAPTER_WORKER_CONTAINER
HEAD_IP=192.0.2.1; HEAD_IF=enp1s0f0; WORKER_IF=enp1s0f0; HEAD_IB=mlx5_0; WORKER_IB=mlx5_0; WORKER_SSH=qa-worker
export HEAD_IP HEAD_IF WORKER_IF HEAD_IB WORKER_IB WORKER_SSH
PATH="$work/bin:$PATH"

# shellcheck disable=SC1090
source "$ADAPTER"
ADAPTER_PORT=8080
_config() { echo "$work/prod-zig.env"; }
reset_logs() { : > "$work/docker.log"; : > "$work/ssh.log"; }
ran_any_rank() { grep -q 'docker run' "$work/docker.log" 2>/dev/null && return 0; grep -q 'docker run' "$work/ssh.log" 2>/dev/null && return 0; return 1; }

auth="$work/auth-keys"; printf 'key\n' > "$auth"; chmod 0644 "$auth"
write_config medium "$auth"; reset_logs
adapter_start >/dev/null 2>&1; rc=$?
check "a rejected auth file fails the start" $([ "$rc" != 0 ] && echo 0 || echo 1)
ran_any_rank && r=1 || r=0; check "a rejected auth file starts no rank" $r
grep -q 'mkdir -p' "$work/ssh.log" 2>/dev/null && r=1 || r=0; check "a rejected auth file does not reach the worker" $r

chmod 0600 "$auth"
write_config max "$auth"; reset_logs
adapter_start >/dev/null 2>&1; rc=$?
check "an invalid default effort fails the start" $([ "$rc" != 0 ] && echo 0 || echo 1)
ran_any_rank && r=1 || r=0; check "an invalid default effort starts no rank" $r

write_config medium "$auth"; reset_logs
adapter_start >/dev/null 2>&1; rc=$?
check "a valid config starts the pair" $([ "$rc" = 0 ] && echo 0 || echo 1)
grep -q 'docker run' "$work/ssh.log"; check "rank 1 runs on the worker" $?
grep -q 'docker run' "$work/docker.log"; check "rank 0 runs on the head" $?
has "$(grep 'docker run' "$work/docker.log")" 'tensorfold-dsv41 serve'; check "the head command is the serve command" $?
has "$(grep 'docker run' "$work/ssh.log")" 'tf-dsv41 follow'; check "the worker command is follow" $?

mkdir -p "$work/state/native-ops"; touch "$work/state/native-ops/STOP"
reset_logs
adapter_start >/dev/null 2>&1; rc=$?
check "a durable STOP refuses the start" $([ "$rc" != 0 ] && echo 0 || echo 1)
ran_any_rank && r=1 || r=0; check "a durable STOP starts no rank" $r

echo "passed $pass, failed $fail"
[ "$fail" = 0 ]
