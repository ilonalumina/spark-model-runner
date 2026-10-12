#!/usr/bin/env bash
# Adapter: GLM-5.3-Flash EXL3, 2x DGX Spark (MiaAI-Lab recipe).
#
# Named for the recipe it drives: adapters/<model>/<recipe>.sh mirrors
# recipes/<model>/<recipe>/, and the CLI sets RECIPE_DIR from that name before
# sourcing this file. This file never states where the recipe lives.
#
# Translates the derived fabric facts into this recipe's variable names and
# calls its own launcher. The recipe is never edited and its .env is never
# written to: the launcher takes caller exports over .env for every key, so
# everything below is passed in from outside. A git pull in the recipe cannot
# conflict with this layer.
#
# Adapter contract (all adapters):
#   adapter_prepare   resolve ADAPTER_PORT / ADAPTER_MODEL_ID / container names
#                     and check the recipe's own config; no side effects. Runs
#                     in `use` before the current serve is stopped, and in
#                     _exec-start before the launched record is written.
#   adapter_start     launch, using the derived fabric facts.
#   adapter_stop      stop what the launched record names (the CLI has already
#                     loaded that identity into ADAPTER_* and WORKER_SSH),
#                     never what today's config says.
#   adapter_health    probe $ADAPTER_PORT as loaded; never re-resolve it.
#   adapter_describe  print the identity as loaded.

# Enforced on the launcher, not merely reported.
ADAPTER_PORT=8888
ADAPTER_MODEL_ID="GLM-5.3-Flash"
ADAPTER_HEAD_CONTAINER="glm53-exl3-head"
ADAPTER_WORKER_CONTAINER="glm53-exl3-worker"

adapter_prepare() {
    [ -x "$RECIPE_DIR/start.sh" ] || { echo "adapter: $RECIPE_DIR/start.sh is missing or not executable" >&2; return 1; }
    [ -f "$RECIPE_DIR/.env" ]     || { echo "adapter: $RECIPE_DIR/.env is missing — cp .env.example .env first" >&2; return 1; }
}

_contract() {
    # Node identity: the same HEAD_IP / WORKER_IP / WORKER_SSH that were probed,
    # already exported by the CLI from the config file. The launcher reads all
    # three names, so its own .env can never point at a different worker.
    export HEAD_IP WORKER_IP WORKER_SSH

    # Endpoint and container identity.
    export PORT="$ADAPTER_PORT"
    export SERVED_MODEL_NAME="$ADAPTER_MODEL_ID"
    export CONTAINER_HEAD="$ADAPTER_HEAD_CONTAINER"
    export CONTAINER_WORKER="$ADAPTER_WORKER_CONTAINER"
}

adapter_start() {
    cd "$RECIPE_DIR" || return 1
    _contract

    # Fabric pins, derived this launch.
    export HEAD_CX7_IF="$HEAD_IF"
    export HEAD_CX7_IB="$HEAD_IB"
    export HEAD_GID="$HEAD_GIDX"
    export WORKER_CX7_IF="$WORKER_IF"
    export WORKER_CX7_IB="$WORKER_IB"
    export WORKER_GID="$WORKER_GIDX"

    # BUILD=1 on every launch, not only the first. Without it the launcher takes
    # its pull branch, fetches the published GHCR tag and overwrites a current
    # repo build on both nodes, with nothing in the output but a changed layer
    # id. The published tag lags main. BUILD=1 also suppresses the worker pull,
    # so the rebuilt image ships to the worker instead.
    export BUILD=1

    # Weights and the worker's copy are already in place; re-checking them costs
    # minutes at every boot and can re-pull the whole checkpoint.
    export SKIP_DOWNLOAD=1
    export SKIP_SYNC=1

    ./start.sh
}

# This launcher's stop honours caller exports (CONTAINER_HEAD / CONTAINER_WORKER
# / WORKER_SSH), so the recorded identity loaded by the CLI reaches it. The
# upstream stop logs "stopped." unconditionally; the CLI verifies absence after.
adapter_stop() {
    cd "$RECIPE_DIR" || return 1
    _contract
    ./start.sh stop
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
