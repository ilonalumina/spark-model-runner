#!/usr/bin/env bash
# install.sh — place model-runner from this checkout into its live
# locations on the head node.
#
# The checkout is the source; /opt/model-runner is the install.
# Deploy is:
#     git pull && ./install.sh
# Idempotent. Re-copies the tool, adapters, README, LICENSE and the unit every
# time.
# Never touches runtime state: {active,launched,attempts} and the locks under
# /etc/model-runner are left alone, and config is only seeded when
# absent.
#
# Recipes are not installed by this script. They are upstream checkouts under
# /opt/model-runner/recipes/<model>/<recipe>, cloned by the
# operator (see README, "Adding a recipe").
#
# The unit runs as the account that runs this script: that account is in the
# docker group and holds the ssh key to the worker. Override with
# MODEL_RUNNER_OWNER=user:group.

set -euo pipefail

NAME=model-runner
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="/opt/$NAME"
STATE="/etc/$NAME"
OWNER="${MODEL_RUNNER_OWNER:-$(id -un):$(id -gn)}"
RUN_USER="${OWNER%%:*}"
RUN_GROUP="${OWNER#*:}"
UNIT="$NAME.service"

# The owner is resolved before anything is written: the unit runs as this
# account and the tree is chowned to it, so an account or group unknown to
# this host refuses the install with nothing changed.
id -u "$RUN_USER" >/dev/null 2>&1 \
    || { echo "install: no such user '$RUN_USER' (set MODEL_RUNNER_OWNER=user:group)" >&2; exit 1; }
getent group "$RUN_GROUP" >/dev/null 2>&1 \
    || { echo "install: no such group '$RUN_GROUP' (set MODEL_RUNNER_OWNER=user:group)" >&2; exit 1; }

sudo mkdir -p "$DEST/adapters" "$DEST/recipes" "$STATE"

sudo install -m 0755 "$SRC/$NAME"             "$DEST/$NAME"
sudo install -m 0644 "$SRC/fabric-facts.sh"   "$DEST/fabric-facts.sh"
sudo install -m 0644 "$SRC/README.md"         "$DEST/README.md"
sudo install -m 0644 "$SRC/LICENSE"           "$DEST/LICENSE"
sudo install -m 0644 "$SRC/config.example"    "$DEST/config.example"
# adapters/<model>/<recipe>.sh — preserve the tree
while IFS= read -r a; do
    sudo install -D -m 0644 "$a" "$DEST/adapters/${a#"$SRC"/adapters/}"
done < <(find "$SRC/adapters" -type f -name '*.sh')

# The installed adapter tree is a copy of the source tree (no state, no
# recipes; those live beside it), so an adapter the source no longer has is
# removed rather than left to show up in `list`, except the one the launched
# record still names: stop and cleanup load that adapter to find what to
# remove, so it is retained until the launch is verified down.
launched_recipe=""
[ -s "$STATE/launched" ] && launched_recipe="$(sed -n 's/^RECIPE=//p' "$STATE/launched")"
while IFS= read -r a; do
    rel="${a#"$DEST"/adapters/}"
    if [ ! -f "$SRC/adapters/$rel" ]; then
        if [ -n "$launched_recipe" ] && [ "$rel" = "$launched_recipe.sh" ]; then
            echo "install: retaining $rel — the launched record names it; stop that serve, then reinstall to remove it" >&2
            continue
        fi
        echo "install: removing adapter no longer in source: $rel"
        sudo rm -f "$a"
    fi
done < <(find "$DEST/adapters" -type f -name '*.sh')
sudo find "$DEST/adapters" -type d -empty -delete 2>/dev/null || true

# Seed config once; after that it is the operator's file.
if [ ! -f "$STATE/config" ]; then
    sudo install -m 0644 "$SRC/config.example" "$STATE/config"
    echo "install: seeded $STATE/config from config.example — review it (two-Spark recipes need WORKER_SSH)"
fi

# The launcher runs as the operator (docker group, node-to-node ssh key) and
# writes into its recipe tree; the tool writes its state files.
sudo chown -R "$OWNER" "$DEST" "$STATE"

# The unit's User= is the resolved account, written literally (no replacement
# syntax in the way) into a temporary rendering; the installed unit is replaced
# only once that rendering carries the line, so a failure here leaves the
# previous unit in place.
unit_tmp="$(mktemp)"
trap 'rm -f "$unit_tmp"' EXIT
while IFS= read -r line; do
    case "$line" in
        User=*) printf 'User=%s\n' "$RUN_USER" ;;
        *)      printf '%s\n' "$line" ;;
    esac
done < "$SRC/$UNIT" > "$unit_tmp"
grep -Fxq "User=$RUN_USER" "$unit_tmp" \
    || { echo "install: rendering $UNIT failed; the installed unit is unchanged" >&2; exit 1; }
sudo install -m 0644 "$unit_tmp" "/etc/systemd/system/$UNIT"
sudo systemctl daemon-reload
# Enabled, so the selected recipe comes back after a reboot; enable never starts
# anything by itself, so a running serve is untouched here as well.
sudo systemctl enable --quiet "$UNIT"

# On PATH, pointing at the install, never at the checkout.
sudo ln -sfn "$DEST/$NAME" "/usr/local/bin/$NAME"

echo "install: $DEST and /etc/systemd/system/$UNIT updated (User=$RUN_USER)"
if systemctl is-active --quiet "$UNIT"; then
    echo "install: $UNIT is active — the running serve is untouched; a restart picks up the new tool"
fi
