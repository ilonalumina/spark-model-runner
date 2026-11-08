#!/usr/bin/env bash
set -u
ADAPTER=${ADAPTER:-adapters/DeepSeek-V4.1-Flash/deepseek-v41-tensorfold-spark-zig.sh}
RECIPE=${ZIG_RECIPE_DIR:?Set ZIG_RECIPE_DIR to the recipe checkout}

pass=0; fail=0
check() { # check <description> <condition-result>
  if [ "$2" = 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL: $1"; fi
}
has() { case "$1" in *"$2"*) return 0;; *) return 1;; esac; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bin"

cat > "$work/prod-zig.env" <<'EOF'
IMAGE=qa/image:test
MODEL_DIR=/models/qa
ENGRAM_DIR=/engram
ASSETS_DIR=/assets
PREPARED_DIR=/prepared
STATE_DIR=/state
CACHE_DIR=/cache
TF_TP_MASTER=192.0.2.1
NCCL_SOCKET_IFNAME=enp1s0f0
NCCL_IB_HCA=mlx5_0
TF_DSV41_CONTEXT=32768
TF_DSV41_SLOTS=2
TF_DSV41_THINKING=0
TF_DSV41_DEFAULT_EFFORT=medium
TF_DRAIN_S=7
QA_UNFORWARDED_SETTING=1
EOF

printf '#!/usr/bin/env bash\nprintf "%%s " "$@" >> "%s/docker.log"; printf "\\n" >> "%s/docker.log"\nexit 0\n' "$work" "$work" > "$work/bin/docker"
printf '#!/usr/bin/env bash\nexit 0\n' > "$work/bin/ssh"
chmod +x "$work/bin/docker" "$work/bin/ssh"

RECIPE_DIR="$RECIPE"
ADAPTER_HEAD_CONTAINER=qa-head
ADAPTER_WORKER_CONTAINER=qa-worker
export RECIPE_DIR ADAPTER_HEAD_CONTAINER ADAPTER_WORKER_CONTAINER

# shellcheck disable=SC1090
source "$ADAPTER"
ADAPTER_PORT=8080

_config() { echo "$work/prod-zig.env"; }

cmd0=$(_run_cmd 0 2>/dev/null); rc0=$?
cmd1=$(_run_cmd 1 2>/dev/null); rc1=$?
check "rank 0 argv builds" $rc0
check "rank 1 argv builds" $rc1

rev=$(cat "$RECIPE/config/tensorfold.rev")

has "$cmd0" "tensorfold-dsv41 serve /model"; check "rank 0 runs the serve command" $?
has "$cmd1" "tf-dsv41 follow /model /assets"; check "rank 1 runs follow with model and assets" $?
has "$cmd1" "tensorfold-dsv41 serve"; [ $? = 0 ] && r=1 || r=0; check "rank 1 does not run serve" $r

has "$cmd0" "--context 32768"; check "rank 0 passes the full context" $?
has "$cmd0" "--max-tokens 32768"; check "rank 0 passes max-tokens equal to the context (not reduced)" $?
has "$cmd0" "--no-thinking"; check "TF_DSV41_THINKING=0 adds --no-thinking" $?
has "$cmd0" "--reasoning-effort medium"; check "TF_DSV41_DEFAULT_EFFORT adds --reasoning-effort" $?

for cmd in "$cmd0" "$cmd1"; do
  has "$cmd" "-e TF_EXPERIMENTAL=deepseek"; check "argv exports TF_EXPERIMENTAL=deepseek" $?
  has "$cmd" "-e TF_DSV41_ASSETS=/assets"; check "argv sets TF_DSV41_ASSETS=/assets" $?
  has "$cmd" "-e TF_DSV41_ENGRAM_DIR=/engram"; check "argv sets TF_DSV41_ENGRAM_DIR=/engram" $?
  has "$cmd" "--label org.tensorfold.revision=$rev"; check "argv labels the image revision from config/tensorfold.rev" $?
  has "$cmd" "-e TF_TP_MASTER=192.0.2.1"; check "argv forwards the pinned TF_TP_MASTER name" $?
  has "$cmd" "QA_UNFORWARDED_SETTING"; [ $? = 0 ] && r=1 || r=0; check "a name the forwarder does not print stays out of argv" $r
  has "$cmd" "-e NCCL_CUMEM_ENABLE=0"; check "NCCL_CUMEM_ENABLE defaults to 0 on GB10" $?
done
has "$cmd0" "--label org.tensorfold.rank=0"; check "rank 0 carries rank label 0" $?
has "$cmd1" "--label org.tensorfold.rank=1"; check "rank 1 carries rank label 1" $?

grep -q 'org.tensorfold.revision' "$ADAPTER"; check "_image_ready reads the revision label" $?
grep -q 'ZIG_COMMIT' "$ADAPTER"; [ $? = 0 ] && r=1 || r=0; check "ZIG_COMMIT is no longer consulted" $r
grep -q '"\.Id"' "$ADAPTER"; [ $? = 0 ] && r=1 || r=0; check "_image_ready does not compare .Id" $r
grep -q 'RootFS.Layers' "$ADAPTER"; check "_image_ready hashes the ordered layers" $?
grep -q 'Architecture' "$ADAPTER"; check "_image_ready hashes the platform" $?

PATH="$work/bin:$PATH"
WORKER_SSH=qa-worker
ADAPTER_LOG_DIR="$work/logs"
_STOP_CFG=1
adapter_stop >/dev/null 2>&1
stop_line=$(grep 'stop -t' "$work/docker.log" | head -1)
has "$stop_line" "stop -t 67"; check "adapter_stop uses TF_DRAIN_S (7) plus the 60 s grace" $?

echo "passed $pass, failed $fail"
[ "$fail" = 0 ]
