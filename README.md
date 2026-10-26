# spark-model-runner

`model-runner`: one selected model recipe running on one NVIDIA DGX Spark or on two (a
head and a worker), surviving reboot, with the fabric facts of a two-Spark recipe derived
at every launch instead of trusted from a file.

Each adapter declares its topology: a single-Spark recipe runs on the head alone; a
two-Spark recipe runs across two GB10 nodes joined over their ConnectX link at
tensor-parallel size 2. See [Scope](#scope) for what is tested and what it does not cover.

## Why it exists

The recipes are launchers, and good ones. Three things sit outside them:

- Their containers detach and the launcher exits. Nothing restarts a serve on boot, and
  the launcher's `stop` is the end of it.
- The RoCEv2 GID index renumbers whenever the fabric interface's addresses change, which
  includes every boot. A recorded index is a measurement, not a constant, and a stale one
  does not fail at startup: it kills that rank about 60 s in.
- The recipes cannot coexist, since each wants exclusive GPUs, and each two-Spark recipe
  spells the same three fabric facts (interface, HCA, GID index) in its own variable
  names.

The runner adds a systemd unit that owns the serve, a fabric probe that derives those
three facts on both nodes at every two-Spark launch, and one adapter per recipe that
translates the facts into that recipe's contract and calls its own launcher. Every serve still runs
through the recipe author's own launcher; the runner never edits a recipe or its config.

## Requirements

For any recipe:

- A DGX Spark (the head) where the account that runs the recipes is in the `docker`
  group.
- At least one supported recipe cloned on the head under
  `/opt/model-runner/recipes/<model>/<recipe>/`, with its own config file
  and weights in place per that recipe's README.
- `flock`, `timeout`, `curl`.

For a two-Spark recipe, also:

- A second DGX Spark (the worker) with the ConnectX link addressed on both ends (the
  recipes' documented default is 10.0.0.1 on the head, 10.0.0.2 on the worker) and RoCEv2
  working between them.
- The account that runs the recipes on the head reaches the worker over ssh without a
  password at every target used: `WORKER_SSH` in the config, and the recipe launcher's own
  worker target where it has one. Each target's account can use Docker on the worker.
- `show_gids` and `ibdev2netdev` on both nodes (they ship with DGX OS).

## Install

On the head:

```bash
git clone https://github.com/ilonalumina/spark-model-runner.git
cd spark-model-runner
./install.sh
```

This installs the `model-runner` command and the `model-runner` systemd unit.

`install.sh` copies the tool, the adapters, this README, `LICENSE` and the unit into
`/opt/model-runner` and `/etc/systemd/system`, seeds
`/etc/model-runner/config` from `config.example` if it is absent, sets the
unit's `User=` to the installing account, and enables the unit so the selected recipe
comes back after a reboot. It is idempotent: `git pull && ./install.sh` is the upgrade
path. It never touches runtime state (`active`, `launched`, `attempts`, the locks) and
never restarts a running serve.

Run it as the account that runs the recipes: the one in the `docker` group with the
ssh key to the worker. That account becomes the unit's `User=` and the owner of the
tree. `MODEL_RUNNER_OWNER=user:group` overrides it; the account and group
are resolved before anything is written, and an unknown one refuses the install with
nothing changed. Run as root without the override, the unit would run as root and the
tree would be reowned to root, so an upgrade is run as the same account as the install.

For two-Spark recipes, set the worker account in the config:

```
/etc/model-runner/config
  HEAD_IP=10.0.0.1
  WORKER_IP=10.0.0.2
  WORKER_SSH=<user>@10.0.0.2
  FABRIC_WAIT_SEC=120
```

This file is where the runner states the two nodes' identity. A single-Spark recipe
needs none of these keys and never reads them; a two-Spark recipe with any of the first
three missing is refused before anything is stopped, with the missing keys named. The
file is exported before anything else runs, so the fabric probe, the stop verification
and the status all address this worker, and the GLM EXL3 and DeepSeek V4.1 EXL3
launchers read `WORKER_SSH` directly. Two launchers build their own worker target from
their config file: Qwen Dual from `WORKER_USER` and `WORKER_IP` (with `WORKER_USER` unset,
the account ssh picks for that address), DSpark from `WORKER_HOST`. Their adapters verify
the address in those files against `WORKER_IP`, not the account. The runner's probes and
stops use `WORKER_SSH`; the launch uses the launcher's target. Both need the access above.
One account for all of them is the simplest configuration; different accounts work when
each has that access.

## Commands

```
model-runner use <model>/<recipe>   select a recipe: stop the current one,
                                    verify it is gone on every node it
                                    used, start the new one
model-runner start                  start the selected recipe
model-runner stop                   stop the running recipe
model-runner restart                stop, then start
model-runner status                 selected recipe, launched identity,
                                    health, unit state
model-runner list                   adapters available
```

`use`, `start` and `restart` queue the job and return at once; a launch takes ten to
twenty minutes and is watched in the journal:

```bash
journalctl -u model-runner -f
```

Ctrl-C there detaches from the journal only. `start`, `restart` and `use` refuse while a
launch is in progress, because a second start would kill it and begin again; `--force`
overrides. `stop` is never refused and stays synchronous, so `use` knows the stop
completed before it switches.

A first launch, once a recipe is cloned and configured:

```bash
model-runner list
model-runner use GLM-5.3-Flash/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
journalctl -u model-runner -f
model-runner status
```

A recipe started by hand has to be stopped first: the runner refuses to launch over
containers it did not create.

## Layout

```
/opt/model-runner/
  model-runner               CLI. Public: use | start | stop | restart | status | list
                             Internal (unit only): _exec-start | _exec-stop | _cleanup
  fabric-facts.sh            derives interface / HCA / GID index per node for two-Spark
                             recipes; every worker probe is bounded by a deadline
  adapters/<model>/<recipe>.sh
                             one adapter per recipe, named for the recipe it drives
  recipes/<model>/<recipe>/  the upstream recipe checkouts, one directory per model and
                             one per recipe of it, owned by the operator account: the
                             launchers write .env, logs/ and build stamps into their tree
  README.md
/etc/model-runner/
  config                     HEAD_IP, WORKER_IP, WORKER_SSH, FABRIC_WAIT_SEC (two-Spark
                             recipes)
  active                     the recipe the operator selected
  launched                   the identity this host actually launched: recipe,
                             topology, recipe dir, worker address, container names,
                             port (a single-Spark launch records no worker)
  attempts                   consecutive failed starts of the current selection
  switch.lock, lifecycle.lock
/etc/systemd/system/model-runner.service
```

The adapter tree and the recipe tree mirror each other. A recipe's name is its path
under `recipes/`, the same string names its adapter, and the CLI derives the recipe
directory from it, so adapter and checkout cannot disagree:

```
adapters/GLM-5.3-Flash/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.sh
recipes/GLM-5.3-Flash/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/
model-runner use GLM-5.3-Flash/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
```

## Lifecycle

systemd is the only lifecycle authority. The public commands call `systemctl`; the unit
calls the internal commands. Nothing creates or removes containers outside `ExecStart`,
`ExecStop` and `ExecStopPost`.

**`_exec-start`** (ExecStart)

1. If a `launched` record exists, from a failed start, an unclean shutdown or a stop that
   could not be verified, stop that recipe and verify it is gone on every node it used
   first, by the topology in the record. Cannot verify: fail, do not launch.
2. For a two-Spark recipe, refuse if `HEAD_IP`, `WORKER_IP` or `WORKER_SSH` is missing.
3. If containers with this adapter's names exist but there is no record, they were not
   launched by this unit: refuse. Stop them by hand.
4. For a two-Spark recipe, derive the fabric facts, waiting up to `FABRIC_WAIT_SEC` for
   the interface to carry its address. Every worker probe is bounded by the remaining
   deadline (ssh connect timeout plus a remote `timeout`), and the final diagnostic
   attempt has its own 10 s budget. The derived values are printed to the journal. A
   single-Spark recipe skips this step and contacts no worker.
5. Write `launched`, then call the recipe's launcher through its adapter.

**`_exec-stop`** (ExecStop): issue the recipe's stop, then prove its containers are
gone. Absent: clear the record, exit 0. Present, or a node unknown: keep the record,
exit non-zero. That non-zero is what makes `use` refuse to switch.

**`_cleanup`** (ExecStopPost): systemd skips `ExecStop` when start-up failed, and the
launchers leave their containers behind when readiness fails. This runs on every exit
path, targets the `launched` record rather than the selector, and keeps the record if
absence cannot be proven so the next start deals with it.

**`use <recipe>`**: takes the switch lock, checks the new recipe's configuration (worker
settings for a two-Spark recipe, then `adapter_prepare`), then stops the running recipe; a
failed stop blocks the switch, as does a record that survives it. Only then writes the
selector. `use` is the deliberate recovery path, so it also resets both retry limits,
the `attempts` file and the unit's failed state and start-rate counter, before
restarting.

Clearing a previous launch inside `_exec-start` happens in a subshell: the old record's
identity, including its topology and worker address, is loaded only to stop and verify
it, and cannot leak into the new launch or its record. Switching from a two-Spark recipe
to a single-Spark one therefore still stops and verifies the worker the old record names,
even when the config no longer names a worker. A record written before the topology line
existed reads as two-Spark.

## Boot and retries

`Restart=on-failure` with `RestartSec=90`: the worker may still be booting when the
head's network and Docker are ready, and each failed attempt is cleaned by `ExecStopPost`
before the next. Two bounds, doing different jobs:

- `StartLimitIntervalSec=2400` / `StartLimitBurst=10` in the unit is a rate limit: more
  than 10 starts inside 40 minutes is refused. It catches fast loops (fabric absent,
  worker down). It does not catch slow ones.
- The `attempts` file is the total: `MAX_START_ATTEMPTS` (10) consecutive failures and
  `_exec-start` exits 75, which `RestartPreventExitStatus=75` turns into "do not
  restart". The unit stays failed, which is the signal that the pair needs a person.
  The counter resets on a successful start and on `use`.

`TimeoutStartSec=4500` sits above the launchers' own readiness timeouts (3600 s at most),
so systemd never kills a launch the recipe would have completed.

## Verification

`verify_absent` queries the head, and for a two-Spark launch the worker, with bounded
commands (`timeout` on the local `docker ps`, connect timeout plus remote `timeout` over
ssh) and answers absent, present or unknown. A failed query on any node it checks is
unknown, never absent; present outranks unknown. Unknown keeps the launched record and
refuses a launch.

After a boot, the journal is the record of what was derived and verified:

```bash
journalctl -u model-runner -b | grep -E 'fabric-facts|verify|launched'
```

The `fabric-facts` lines are which GID indices this boot used. Compare them against a
serve that dies about 60 s in before suspecting anything else.

## Adapters

Each adapter is written from its launcher's source, because the launchers differ in how
they take configuration. Eighteen adapters, in three groups.

The four vLLM / EXL3 two-Spark recipes:

| | GLM-5.3-Flash EXL3 | Qwen3.8-Flash-Next Dual NVFP4 | DeepSeek-V4-Flash DSpark | DeepSeek-V4.1-Flash EXL3 |
|---|---|---|---|---|
| config file | `.env` | `.env` | `.env.dspark` | `.env` |
| environment vs file | caller exports win | file wins (`source .env`) | file wins (`set -a; source`), and the file is pushed to the worker | file wins for keys it assigns; the adapter computes the effective values |
| how the derived pins reach it | exported by the adapter | verified against the file; mismatch refuses with the lines to set | identity and interface/HCA verified against the file | exported, and verified as effective before the launch |
| GID index | `HEAD_GID` / `WORKER_GID` exported | single `IB_GID_INDEX` in the file, verified; refuses if the nodes disagree | the launcher validates RoCEv2 GIDs from sysfs on both ranks itself | `HEAD_GID` / `WORKER_GID` exported |
| containers | `glm53-exl3-head` / `-worker` | `vllm-fn` on both nodes | `<PROJECT>-vllm-dspark-1` on both, project `deepseek-v4-flash` | `dsv41-exl3-head` / `-worker` |
| stop reports | success unconditionally | success unconditionally | exit 1 on a failed remote step | success unconditionally |
| launch flags | `BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1` | `--no-download` (rsyncs the worker once) | none; exit 3 = ranks already running, accepted only if `/health` answers | none |

Where the file wins, the adapter verifies it rather than writing to it: a mismatch is
refused before launch.

The five TensorFold recipes all take their settings from the environment before their
config file, so these adapters export the identity (container names, port, served name,
and the worker for the MiaAI-Lab GLM launcher) and verify only what the launcher reads
from its file alone:

| | GLM-5.3-Flash TensorFold (MiaAI-Lab) | GLM-5.3-Flash TensorFold (jayleaton) | DeepSeek-V4.1-Flash TensorFold | DeepSeek-V4.1-Flash TensorFold Zig | Qwen3.8-Flash-Next TensorFold, one Spark |
|---|---|---|---|---|---|
| topology | two Sparks | two Sparks | two Sparks | two Sparks | one Spark |
| launcher | `start.sh` | `scripts/serve.sh start` | `scripts/serve.sh start` | the adapter runs both ranks' `docker run` as `scripts/run-zig.sh` composes it | `start.sh` |
| config file | `.env`, `scripts/local.sh` | `config/prod.env` | `config/prod.env` | `config/prod-zig.env` | `.env` |
| verified against the derived facts | nothing: the launcher derives the RoCE link from the route to `FABRIC_PEER` | `WORKER_SSH`, `HEAD_IP`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | `WORKER_SSH`, `HEAD_IP`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | `TF_TP_MASTER`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | n/a |
| checked before the stop of the current serve | prepared state, checkpoint and drafter on both nodes, TP=2 with the checkpoint copied | placeholders filled | model and Engram shards on both nodes, the recipe's `preflight static` | image, model, Engram shards, assets on both nodes | prepared state, checkpoint |
| containers | `glm53-flash-tf` on both nodes | `glm53-tf-r0` / `-r1` | `dsv41-tf-r0` / `-r1` | `tensorfold-zig-r0` / `-r1` | `qwen38-flash-next-tf` |
| image build and downloads | never from the unit (`PREPARE=0`); run `scripts/prepare.sh` by hand | by hand, per the recipe's README | by hand, per `docs/INSTALL.md` | by hand, `docs/ZIG-SERVE.md` steps 1 and 2 | never from the unit (`PREPARE=0`); run `scripts/prepare.sh` by hand |
| stop | rank logs saved as the recipe's `stop.sh` saves them, 30 s graceful stop, removed | removed on the recorded nodes | removed on the recorded nodes | SIGTERM to rank 0 (drain), rank 1 waited for, logs saved, removed | the recipe's `stop.sh` (30 s graceful stop, removed) |

Further vLLM and TensorFold recipes:

| | Qwen3.8-Flash-Next Single NVFP4 | DeepSeek-V4.1-Flash EXL3 vLLM (sfxnz) | MiMo-V2.6-Flash | Qwen3.8-Flash TensorFold, two Sparks |
|---|---|---|---|---|
| topology | one Spark | two Sparks | two Sparks | two Sparks |
| config file | `.env` | none | `.env` (`ENV_FILE`) | `.env`, `scripts/local.sh` |
| environment vs file | caller exports win only for the keys start.sh captures (`PORT`); `.env` wins for the rest, so `TP1_CONTAINER_NAME` is verified and then passed explicitly at start and stop | caller exports win for every key; `MODEL`, `HF_CACHE`, `SNAPSHOT_SHA` and `IMAGE` are exported and the launcher's early-exit controls cleared | file wins; the adapter pins `ENV_FILE` to the recipe's `.env`, computes the effective values and refuses a `.env` that redirects identity, NFS or `ENV_FILE` | environment first, so the node identity is exported (`WORKER`, `FABRIC_PEER`) |
| containers | `vllm-fn-tp1` | `dsv41-flash-exl3` on both nodes | `mimo26-tp2-head` / `-worker` | `qwen38-fn-tf` on both nodes |
| image build and downloads | by hand per the recipe's README | by hand (`docker/Dockerfile` as `canonical-e14`, the pinned pack on both nodes) | by hand (`./start.sh build`, the checkpoint) | never from the unit (`PREPARE=0`); run `scripts/prepare.sh` by hand |

The eugr/spark-vllm-docker recipes (vLLM b12x, NVIDIA FP4-Local checkpoints): one checkout shared by four adapters,
`recipes/<model>/<recipe>` symlinks to it, each adapter naming its recipe explicitly. An adapter whose checkpoint is
not the recipe's `model:` (the abliterated GLM build) runs a derived copy of the recipe written to the runner's state
directory, with exactly the `model:` line and the `vllm serve` model changed.

| | Qwen3.8-Flash-Next FP4-Local, one Spark | Qwen3.8-Flash-Next FP4-Local, two Sparks | GLM-5.3-Flash FP4-Local | GLM-5.3-Flash FP4-Local abliterated (Huihui) |
|---|---|---|---|---|
| recipe | `qwen3.8-flash-next-nvfp4-solo` | `qwen3.8-flash-next-nvfp4-cluster` | `glm-5.3-flash` | `glm-5.3-flash`, derived copy |
| topology | one Spark | two Sparks | two Sparks | two Sparks |
| configuration | every launcher call runs under `env -i` with `--config` naming a file the adapter writes in the runner's state directory (node and fabric identity only; empty for one Spark); the checkout's `.env` and inherited `DOTENV_*` are never read | same | same | same |
| checked before the stop of the current serve | the checkout at the reviewed commit with no tracked change, PyYAML present, the checkpoint at its pinned revision against the file manifest in the adapter (names, sizes, config and index) in `$HOME/.cache/huggingface` on each node, the image `vllm-node-b12x` on each node (equal Id or equal content), the launcher's dry run | same | same | same |
| containers | `vllm_node` | `vllm_node` on both nodes | `vllm_node` on both nodes | `vllm_node` on both nodes |
| stop | `docker stop -t 90` and remove, by recorded name on the recorded hosts; the launcher is not called | same | same | same |

### Verified against

An adapter is verified against the upstream commits below. "Launched" is a serve brought
up under the runner from that commit. "Read" is the newest commit at which the launcher
sections the adapter's contract depends on were read against it: for GLM EXL3 the whole
launcher; for Qwen Dual the configuration sourcing and the `ABLIT` / `HF_TOKEN` path; for
DSpark the exit codes, container names and sourcing; for DeepSeek V4.1 EXL3 the sourcing
block and the default assignments; for the TensorFold recipes the settings order, the
caller exports the launcher accepts, the keys read from the file alone, the prepared-state
and preflight helpers, and the stop script. A read is not a launch. "Not recorded" is an
adapter that has served under the runner from a commit that was not recorded. A newer upstream
commit is run at the user's own reading of the diff; the comments at the top of each
adapter say what to look for.

| adapter | recipe repository | launched at | read at |
|---|---|---|---|
| GLM-5.3-Flash EXL3 | MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks | f906ee9 | bc68f31 |
| Qwen3.8-Flash-Next Dual NVFP4 | MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks | d2ae28a | d2f54b7 |
| DeepSeek-V4-Flash DSpark | MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark | f3d7645 | 97e8733 |
| DeepSeek-V4.1-Flash EXL3 | MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks | 8404ac7 | 8404ac7 |
| GLM-5.3-Flash TensorFold (MiaAI-Lab) | MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold | 92bf731 | 3fd60df |
| GLM-5.3-Flash TensorFold (jayleaton) | jayleaton/glm53-tensorfold-spark | b463237 | f79bf9c |
| DeepSeek-V4.1-Flash TensorFold | jayleaton/deepseek-v41-tensorfold-spark | not recorded | 2dcce03 |
| DeepSeek-V4.1-Flash TensorFold Zig | jayleaton/deepseek-v41-tensorfold-spark | c34da993 | 2dcce03 |
| DeepSeek-V4.1-Flash-1017 TensorFold Zig | MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks | 27741e1 | 27741e1 |
| Qwen3.8-Flash-Next TensorFold, one Spark | MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold | not launched | 656879f |
| Qwen3.8-Flash-Next Single NVFP4 | MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark | not launched | b7c5da1 |
| Qwen3.8-Flash TensorFold, two Sparks | MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold | not launched | 6730319 |
| DeepSeek-V4.1-Flash EXL3 vLLM (sfxnz) | sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark | not launched | 3de9146 |
| MiMo-V2.6-Flash | MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks | not launched | 201be3e |
| Qwen3.8-Flash-Next FP4-Local, one Spark | eugr/spark-vllm-docker | not launched | 0bdc9c9 |
| Qwen3.8-Flash-Next FP4-Local, two Sparks | eugr/spark-vllm-docker | not launched | 0bdc9c9 |
| GLM-5.3-Flash FP4-Local | eugr/spark-vllm-docker | not launched | 0bdc9c9 |
| GLM-5.3-Flash FP4-Local abliterated (Huihui) | eugr/spark-vllm-docker | not launched | 0bdc9c9 |

### Per-recipe prerequisites

One-time, in the recipe's own config file, beyond what its README already asks for:

- **GLM-5.3-Flash EXL3**: nothing beyond the recipe's own `.env`. The adapter passes node identity,
  fabric pins and container names as exports, which this launcher takes over `.env`.
- **Qwen3.8-Flash-Next Dual**: `cp .env.sample .env`, then the cluster block has to match the derived
  facts: `IFACE`, `IB_HCA` (with the leading `=`), `IB_GID_INDEX`, `HEAD_IP`,
  `WORKER_IP`. The adapter refuses with the exact lines otherwise. This launcher's
  worker target is `WORKER_USER@WORKER_IP` (with `WORKER_USER` unset, the account ssh
  picks for that address); that account needs passwordless ssh and Docker on the worker.
  The first launch rsyncs the checkpoint to the worker.
- **DSpark**: `cp .env.dspark.example .env.dspark`, then `WORKER_HOST=<user>@10.0.0.2`,
  this launcher's ssh target, with an account that has passwordless ssh and Docker on the
  worker (the adapter verifies the address, not the account), `MASTER_ADDR`,
  `VLLM_HOST_IP`, `WORKER_VLLM_HOST_IP`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`,
  `TP_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` matching the pair. Also required by its
  launcher: the image on both nodes, the same recipe path on the worker owned by the
  operator account (or `WORKER_DIR` set), and the weights on both nodes. `DSPARK_RESTART_POLICY=no` makes the runner the only thing that starts this
  serve; with the shipped `unless-stopped`, dockerd also restores the ranks on reboot and
  the runner then stops and relaunches them cold. The adapter prints a note while the
  policy is not `no`; it does not choose.
- **DeepSeek-V4.1-Flash EXL3**: `cp .env.example .env` and the worker block. The adapter computes
  the configuration the launcher would run with and refuses if `.env` redirects an
  identity or fabric key away from the runner's facts.
- **GLM-5.3-Flash TensorFold (MiaAI-Lab)**: run `scripts/prepare.sh` once (image,
  checkpoint and drafter on both Sparks). The adapter exports `WORKER=$WORKER_SSH` and
  `FABRIC_PEER=$WORKER_IP` at every launch, and the recipe's prepared state records the
  worker target, so run `prepare.sh` with `WORKER` set to the value of `WORKER_SSH`; the
  adapter refuses until the prepared state matches what is on the nodes.
- **GLM-5.3-Flash TensorFold (jayleaton)** and **DeepSeek-V4.1-Flash TensorFold**:
  `cp config/prod.env.example config/prod.env`, then `WORKER_SSH`, `HEAD_IP`, the node
  paths and the CX7 names; the adapter refuses while a `<placeholder>` is left or a value
  differs from the derived facts. The DeepSeek recipe's own watchdog and boot units
  (`scripts/systemd`) stay unused: the runner owns start, stop and boot, and its own
  watchdog timer does the recipe watchdog's job for every recipe.
- **DeepSeek-V4.1-Flash TensorFold Zig**: `cp config/prod-zig.env.example
  config/prod-zig.env` and fill it in, build the image with `ZIG_COMMIT` set to the
  checkout's commit, and put the image, the model, the Engram shards, the assets and the
  cache directory on both nodes at the same paths (`docs/ZIG-SERVE.md`). The adapter
  starts both ranks from the head, so the worker needs no checkout.
  The example config enables lossy decode-time expert pruning with
  `TF_DSV41_EXPERT_TOPP=0.85`, which can change model responses. To disable pruning,
  remove that setting from `config/prod-zig.env` and run `model-runner restart`.
  The upstream recipe documents an unset value as no pruning (`docs/ZIG-SERVE.md`).
- **Qwen3.8-Flash-Next TensorFold, one Spark**: run `scripts/prepare.sh` once. Nothing in
  the runner's config is needed for it.
- **Qwen3.8-Flash-Next Single NVFP4**: `cp .env.sample .env`, set `IMAGE`, and set `TP1_MODEL_ID` to a checkpoint in the
  local cache (`nvidia/Qwen3.8-Flash-Next-NVFP4`). The recipe's own supervisor and heartbeat units stay unused.
- **Qwen3.8-Flash TensorFold, two Sparks**: `scripts/local.sh` with `WORKER` set to the value of `WORKER_SSH`, then
  `scripts/prepare.sh` once (image on both nodes, checkpoint copied to the worker).
- **DeepSeek-V4.1-Flash EXL3 vLLM (sfxnz)**: the image `dsv41-flash-exl3-sm121:canonical-e14` and the pack revision
  `2.0bpw-mcg-viterbi-lmhead-mxfp8` of `sfxnz/DeepSeek-V4.1-Flash-EXL3` in `$HOME/.cache/huggingface` on both nodes.
- **MiMo-V2.6-Flash**: `cp .env.example .env`, then the identity, NFS and worker-path keys the adapter names; the
  image (`./start.sh build`) and the checkpoint.
- **eugr/spark-vllm-docker recipes**: clone it once to `recipes/spark-vllm-docker` at the reviewed commit and link
  `recipes/Qwen3.8-Flash-Next/qwen3.8-flash-next-nvfp4-solo`, `.../qwen3.8-flash-next-nvfp4-cluster` and
  `recipes/GLM-5.3-Flash/glm-5.3-flash` and `recipes/GLM-5.3-Flash/glm-5.3-flash-huihui-abliterated` to it; PyYAML on the head; `./build-and-copy.sh --exp-b12x` on the head (and
  `--no-build -c <worker>` for the two-Spark recipes); `hf download <checkpoint> --revision <the adapter's revision>`
  on every node the recipe uses. The worker is reached as this account at `WORKER_IP`.

### Writing an adapter

Clone the recipe to `recipes/<model>/<recipe>/`, then add one file at
`adapters/<model>/<recipe>.sh`, the same path with `.sh` appended. It does not set
`RECIPE_DIR`; the CLI derives that from the name before sourcing the file. It sets
`ADAPTER_TOPOLOGY=single` for a recipe that runs on the head alone; unset, or `pair`, is
a two-Spark recipe. It defines:

- `adapter_prepare`: resolve `ADAPTER_PORT`, `ADAPTER_MODEL_ID`, `ADAPTER_HEAD_CONTAINER`,
  `ADAPTER_WORKER_CONTAINER` (two-Spark only) and check the recipe's own config. No side
  effects. `use` runs it before the current serve is stopped; `_exec-start` runs it before
  the launched record is written.
- `adapter_start`: launch. A two-Spark adapter gets the derived fabric facts (`HEAD_IF`,
  `HEAD_IB`, `HEAD_GIDX`, `WORKER_IF`, `WORKER_IB`, `WORKER_GIDX`); for a recipe whose file
  wins over the environment, verify the file against them first and refuse with the lines
  to set.
- `adapter_stop`: stop what the launched record names. The CLI has loaded that identity
  into `ADAPTER_*` and `WORKER_SSH` before calling. An upstream stop script that
  re-sources its own config stops whatever today's file names, so remove the recorded
  containers on the recorded worker directly and run the upstream stop afterwards only
  when today's file still names that worker, or pass the recorded names to it where the
  environment wins.
- `adapter_health`: probe `$ADAPTER_PORT` as loaded; never re-read it from the file.
- `adapter_describe`: print the identity as loaded.

Those variable names are the contract: read them at call time inside the functions,
never bake them into strings at load time. `ssh_bounded` from `fabric-facts.sh` is
available for worker-side commands with a deadline.

`install.sh` prunes adapters the source no longer has, except one the launched record
still names; stop and cleanup need it to find what to remove, so it is retained with a
warning until that serve is verified down.

## What it does not do

- **Restarts, never fails over.** `model-runner-watchdog.timer` runs `model-runner _watch`
  every minute: three failed `/health` probes of the active recipe in a row (a rank gone, or
  a TensorFold serve's own 503 `stalled`) request a restart of the unit, at most once every
  30 minutes (`WATCH_FAILS`, `WATCH_MIN_HEAL`). It stands down during a `use`, launch, stop or
  cleanup and while the unit is not active, and its request (`--job-mode=fail`) is refused by
  systemd while a stop is queued, so a deliberate stop always wins. Pause it for benchmarks:
  `sudo systemctl stop model-runner-watchdog.timer`.
- **No stable port across recipes.** A consumer needs the endpoint and the model id for
  whichever recipe is active; `status` prints both.
- **Starts nothing on the worker itself.** The recipe launchers start rank 1 over ssh;
  the DeepSeek TensorFold Zig adapter does the same with the launcher's own `docker run`.
  The runner probes the worker's GID table and container list the same way, and at stop
  and cleanup the two-Spark adapters remove the recorded containers on the recorded
  worker directly (`docker rm -f` over ssh), running the upstream stop where it cannot
  target another worker; GLM EXL3's stop goes through its launcher.

## Scope

Supported: one recipe at a time on one head, either alone (single-Spark recipes) or with
one worker (two-Spark recipes), through the adapters above. The two-Spark adapters
with a "launched at" commit have run on machines. Single-Spark operation and switching
between the two topologies in both directions are exercised against mocks of Docker,
ssh, systemd and the launchers, not yet on a machine.

Not supported: the recipes' TP=3 / TP=4 launchers (`start-tp3.sh`, `start-tp4.sh`,
`WORKER2_HOST`, the MiaAI-Lab TensorFold recipe's `TP=3`), which need a worker list the
runner does not have, and recipes that serve on more than one head at once.

## License

AGPL-3.0. See `LICENSE`.
