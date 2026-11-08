#!/usr/bin/env bash
# Usage: ZIG_RECIPE_DIR=/path/to/deepseek-v41-tensorfold-spark tests/deepseek-v41-tensorfold-spark-zig.test.sh
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
src="${ZIG_RECIPE_DIR:-}"
[ -n "$src" ] && [ -f "$src/scripts/engine_settings.py" ] || { echo "SKIP: set ZIG_RECIPE_DIR to a checkout of jayleaton/deepseek-v41-tensorfold-spark"; exit 0; }

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
fail=0; pass=0
ok() { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL: $*"; }

new_recipe() {
    RECIPE_DIR="$work/recipe"; rm -rf "$RECIPE_DIR"; mkdir -p "$RECIPE_DIR"
    cp -r "$src/scripts" "$src/config" "$RECIPE_DIR/"
    REV="$(cat "$RECIPE_DIR/config/tensorfold.rev")"
    cat >"$RECIPE_DIR/config/prod-zig.env" <<CFG
IMAGE="example-model:test"
TF_TP_MASTER="192.0.2.1"
NCCL_SOCKET_IFNAME="eth9"
NCCL_IB_HCA="mlx_a,mlx_b"
MODEL_DIR="/m"
ENGRAM_DIR="/e"
ASSETS_DIR="/a"
PREPARED_DIR="/p"
STATE_DIR="/s"
CACHE_DIR="/c"
HOST=0.0.0.0
PORT=8080
TF_DSV41_SLOTS=4
TF_DSV41_CONTEXT=1048576
TF_DRAIN_S=30
TF_TP_ROCE_FAST=1
TF_SSE_KEEPALIVE_S=15
CFG
}

stub_bin() {
    mkdir -p "$work/bin"
    cat >"$work/bin/docker" <<'STUB'
#!/usr/bin/env bash
echo "docker $*" >>"$STUB_LOG"
case "$1 $2" in
  "image inspect") case "$*" in
      *org.tensorfold.revision*) echo "${STUB_REV:-}";;
      *RootFS.Layers*) echo "${STUB_IDENTITY:-same}";;
      *) echo "sha256:x";; esac;;
esac
STUB
    cat >"$work/bin/ssh" <<'STUB'
#!/usr/bin/env bash
echo "ssh $*" >>"$STUB_LOG"
for a; do last="$a"; done
case "$last" in
  *org.tensorfold.revision*) echo "${STUB_W_REV:-${STUB_REV:-}}";;
  *RootFS.Layers*) echo "${STUB_W_IDENTITY:-${STUB_IDENTITY:-same}}";;
esac
STUB
    chmod +x "$work/bin/docker" "$work/bin/ssh"
    export PATH="$work/bin:$PATH" STUB_LOG="$work/stub.log"
}

load_adapter() {
    # shellcheck disable=SC1090
    source "$here/adapters/DeepSeek-V4.1-Flash/deepseek-v41-tensorfold-spark-zig.sh"
    ADAPTER_PORT=8080
    WORKER_SSH=worker HEAD_IP=192.0.2.1
}

stub_bin
new_recipe
load_adapter

r0="$(_run_cmd 0)" || bad "rank 0 command failed"
r1="$(_run_cmd 1)" || bad "rank 1 command failed"

case "$r1" in *"tf-dsv41 follow /model /assets"*) ok;; *) bad "rank 1 runs tf-dsv41 follow: $r1";; esac
case "$r1" in *tf-dsv41-m1*) bad "rank 1 still runs tf-dsv41-m1";; *) ok;; esac
for want in "TF_DRAIN_S=30" "TF_SSE_KEEPALIVE_S=15" "TF_EXPERIMENTAL=deepseek" "TF_DSV41_ASSETS=/assets" \
            "TF_DSV41_ENGRAM_DIR=/engram" "org.tensorfold.revision=$REV" "org.tensorfold.rank=0" "--max-tokens 1048576"; do
    case "$r0" in *"$want"*) ok;; *) bad "rank 0 argv lacks $want";; esac
done
case "$r1" in *"org.tensorfold.rank=1"*) ok;; *) bad "rank 1 argv lacks its rank label";; esac
case "$r0" in *--no-thinking*|*--reasoning-effort*) bad "thinking flags without the settings";; *) ok;; esac

printf 'TF_DSV41_THINKING=0\nTF_DSV41_DEFAULT_EFFORT=high\n' >>"$RECIPE_DIR/config/prod-zig.env"
r0="$(_run_cmd 0)" || bad "rank 0 command failed with thinking settings"
case "$r0" in *--no-thinking*) ok;; *) bad "TF_DSV41_THINKING=0 gives --no-thinking";; esac
case "$r0" in *"--reasoning-effort high"*) ok;; *) bad "TF_DSV41_DEFAULT_EFFORT gives --reasoning-effort";; esac
case "$r0" in *"TF_DSV41_THINKING="*|*"TF_DSV41_DEFAULT_EFFORT="*) bad "recipe-only settings forwarded as environment";; *) ok;; esac

new_recipe
echo 'GLM53_TF_ROCE_FAST=1' >>"$RECIPE_DIR/config/prod-zig.env"
if _run_cmd 0 >/dev/null 2>&1; then bad "a legacy setting name is refused"; else ok; fi

new_recipe
export STUB_REV="$REV" STUB_IDENTITY=same
unset STUB_W_REV STUB_W_IDENTITY
if _image_ready 2>/dev/null; then ok; else bad "image with the pinned revision on both nodes is ready"; fi
export STUB_W_IDENTITY=other
if _image_ready 2>/dev/null; then bad "differing image content between nodes is refused"; else ok; fi
unset STUB_W_IDENTITY
export STUB_REV=0000000
if _image_ready 2>/dev/null; then bad "an image of another engine revision is refused"; else ok; fi

: >"$STUB_LOG"
adapter_stop
grep -q "docker stop -t 90 " "$STUB_LOG" && ok || bad "stop waits TF_DRAIN_S + 60 s"

new_recipe
sed -i '/^TF_DSV41_SLOTS=/d;/^TF_DSV41_CONTEXT=/d' "$RECIPE_DIR/config/prod-zig.env"
r0="$(_run_cmd 0)" || bad "rank 0 command failed without slots and context"
case "$r0" in *"--parallel 1 --context 4096"*"--max-tokens 4096"*) ok;; *) bad "slots and context default to 1 and 4096: $r0";; esac

new_recipe
echo 'CPUSET_CPUS=0-9' >>"$RECIPE_DIR/config/prod-zig.env"
r1="$(_run_cmd 1)" || bad "rank 1 command failed with CPUSET_CPUS"
case "$r1" in *"--cpuset-cpus 0-9"*) ok;; *) bad "CPUSET_CPUS gives --cpuset-cpus";; esac

new_recipe
echo 'TF_DSV41_DEFAULT_EFFORT=max' >>"$RECIPE_DIR/config/prod-zig.env"
if _run_cmd 0 >/dev/null 2>&1; then bad "an unknown TF_DSV41_DEFAULT_EFFORT is refused"; else ok; fi

new_recipe
echo 'TENSORFOLD_API_KEY=k' >>"$RECIPE_DIR/config/prod-zig.env"
if _run_cmd 0 >/dev/null 2>&1; then bad "TENSORFOLD_API_KEY without NATIVE_AUTH_FILE is refused"; else ok; fi

new_recipe
mkdir -p "$work/auth"; echo key >"$work/auth/keys"; chmod 600 "$work/auth/keys"
echo "NATIVE_AUTH_FILE=$work/auth/keys" >>"$RECIPE_DIR/config/prod-zig.env"
r0="$(_run_cmd 0)" || bad "rank 0 command failed with NATIVE_AUTH_FILE"
r1="$(_run_cmd 1)" || bad "rank 1 command failed with NATIVE_AUTH_FILE"
case "$r0" in *"$work/auth/keys:/run/tensorfold/auth-keys:ro"*"--api-key-file /run/tensorfold/auth-keys"*) ok;;
    *) bad "NATIVE_AUTH_FILE is mounted on rank 0 with --api-key-file: $r0";; esac
case "$r1" in *auth-keys*) bad "rank 1 gets the auth file";; *) ok;; esac
chmod 644 "$work/auth/keys"
if _run_cmd 0 >/dev/null 2>&1; then bad "an auth file not mode 0600 is refused"; else ok; fi

cat >"$work/bin/curl" <<'STUB'
#!/usr/bin/env bash
echo "curl $*" >>"$STUB_LOG"
case "$*" in *"/health"*) exit 0;; *) exit 22;; esac
STUB
chmod +x "$work/bin/curl"
HEAD_IF=eth9 WORKER_IF=eth9 HEAD_IB=mlx_a WORKER_IB=mlx_b
new_recipe
sed -i "s#^STATE_DIR=.*#STATE_DIR=\"$work/state\"#;s#^PREPARED_DIR=.*#PREPARED_DIR=\"$work/prep\"#" "$RECIPE_DIR/config/prod-zig.env"
mkdir -p "$work/state/native-ops"; touch "$work/state/native-ops/STOP"
: >"$STUB_LOG"
if adapter_start 2>/dev/null; then bad "a durable STOP refuses the start"; else ok; fi
grep -q "docker run" "$STUB_LOG" && bad "no rank starts under a durable STOP" || ok
rm "$work/state/native-ops/STOP"
chmod 600 "$work/auth/keys"
echo "NATIVE_AUTH_FILE=$work/auth/keys" >>"$RECIPE_DIR/config/prod-zig.env"
: >"$STUB_LOG"
if adapter_start 2>/dev/null; then ok; else bad "with NATIVE_AUTH_FILE, /health is the readiness check"; fi
grep -q "/v1/" "$STUB_LOG" && bad "no unauthenticated /v1 request with NATIVE_AUTH_FILE" || ok
chmod 644 "$work/auth/keys"
: >"$STUB_LOG"
if adapter_start 2>/dev/null; then bad "a rejected auth file fails the start"; else ok; fi
grep -q "docker run" "$STUB_LOG" && bad "no rank starts with a rejected auth file" || ok
chmod 600 "$work/auth/keys"
echo 'TF_DSV41_DEFAULT_EFFORT=max' >>"$RECIPE_DIR/config/prod-zig.env"
: >"$STUB_LOG"
if adapter_start 2>/dev/null; then bad "an invalid config fails the start"; else ok; fi
grep -q "docker run" "$STUB_LOG" && bad "no rank starts with an invalid config" || ok

echo "passed $pass, failed $fail"
[ "$fail" = 0 ]
