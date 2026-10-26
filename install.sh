#!/usr/bin/env bash
# Installs this checkout to /opt/model-runner on the head node. Idempotent; runtime state in /etc/model-runner is kept.
# Upgrade with: git pull && ./install.sh
# Runs the unit as the invoking account (needs the docker group and ssh to the worker); override with MODEL_RUNNER_OWNER=user:group.

set -euo pipefail

NAME=model-runner
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="/opt/$NAME"
STATE="/etc/$NAME"
OWNER="${MODEL_RUNNER_OWNER:-$(id -un):$(id -gn)}"
RUN_USER="${OWNER%%:*}"
RUN_GROUP="${OWNER#*:}"
UNIT="$NAME.service"

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
while IFS= read -r a; do
    sudo install -D -m 0644 "$a" "$DEST/adapters/${a#"$SRC"/adapters/}"
done < <(find "$SRC/adapters" -type f -name '*.sh')

# Remove adapters no longer in the source, except the one the running launch needs to stop.
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

if [ ! -f "$STATE/config" ]; then
    sudo install -m 0644 "$SRC/config.example" "$STATE/config"
    echo "install: seeded $STATE/config from config.example — review it (two-Spark recipes need WORKER_SSH)"
fi

# The root watchdog opens these read-only, so they must exist and belong to the runner account.
sudo touch "$STATE/switch.lock" "$STATE/lifecycle.lock"

sudo chown -R "$OWNER" "$DEST" "$STATE"

# Render User= into a temporary copy; the installed unit changes only if that succeeds.
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
sudo systemctl enable --quiet "$UNIT"

sudo install -m 0644 "$SRC/$NAME-watchdog.service" "/etc/systemd/system/$NAME-watchdog.service"
sudo install -m 0644 "$SRC/$NAME-watchdog.timer"   "/etc/systemd/system/$NAME-watchdog.timer"
sudo systemctl daemon-reload
sudo systemctl enable --now --quiet "$NAME-watchdog.timer"

sudo ln -sfn "$DEST/$NAME" "/usr/local/bin/$NAME"

echo "install: $DEST and /etc/systemd/system/$UNIT updated (User=$RUN_USER)"
if systemctl is-active --quiet "$UNIT"; then
    echo "install: $UNIT is active — the running serve is untouched; a restart picks up the new tool"
fi
