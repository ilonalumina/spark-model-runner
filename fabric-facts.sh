#!/usr/bin/env bash
# Derives each node's fabric interface, HCA and RoCEv2 GID index at launch; the GID index renumbers on every boot.

# show_gids lists a v1 and a v2 row per address; RoCEv2 needs the v2 row.
_derive_local() {
    local ip="$1" dev idx netdev

    read -r dev idx netdev < <(
        show_gids 2>/dev/null | awk -v ip="$ip" '$5 == ip && $6 == "v2" { print $1, $3, $7; exit }'
    ) || true

    if [ -z "${dev:-}" ]; then
        echo "fabric-facts: no RoCEv2 GID entry carries $ip" >&2
        echo "fabric-facts: the interface is probably addressless; show_gids output follows" >&2
        show_gids >&2 || true
        return 1
    fi

    # A down HCA still has a GID table. Captured first: grep -q in a pipe would SIGPIPE ibdev2netdev under pipefail.
    local links
    if ! links="$(ibdev2netdev 2>/dev/null)"; then
        echo "fabric-facts: ibdev2netdev failed; the port state of $dev is unknown" >&2
        return 1
    fi
    if ! grep -Eq "^${dev}[[:space:]].*\(Up\)" <<<"$links"; then
        echo "fabric-facts: $dev carries $ip but its port is not Up" >&2
        ibdev2netdev >&2 || true
        return 1
    fi

    printf '%s %s %s\n' "$netdev" "$dev" "$idx"
}

# ssh_bounded <seconds> <remote command...>: runs the remote command with the connection and execution time-bounded.
ssh_bounded() {
    local budget="$1"; shift
    [ "$budget" -ge 1 ] 2>/dev/null || budget=1
    timeout -k 2 "$(( budget + 2 ))" \
        ssh -o BatchMode=yes -o ConnectTimeout="$budget" "$WORKER_SSH" \
            "timeout ${budget}s bash -c $(printf '%q' "$*")"
}

# derive <deadline-epoch-seconds>: exports HEAD_IF HEAD_IB HEAD_GIDX WORKER_IF WORKER_IB WORKER_GIDX.
derive() {
    local deadline="$1" h w remaining

    h="$(_derive_local "$HEAD_IP")" || {
        echo "fabric-facts: head derivation failed" >&2
        return 1
    }

    remaining=$(( deadline - $(date +%s) ))
    [ "$remaining" -ge 1 ] || remaining=1
    w="$(ssh_bounded "$remaining" "$(declare -f _derive_local); _derive_local $WORKER_IP")" || {
        echo "fabric-facts: worker derivation failed over $WORKER_SSH (budget ${remaining}s)" >&2
        return 1
    }

    read -r HEAD_IF HEAD_IB HEAD_GIDX <<<"$h"
    read -r WORKER_IF WORKER_IB WORKER_GIDX <<<"$w"
    export HEAD_IF HEAD_IB HEAD_GIDX WORKER_IF WORKER_IB WORKER_GIDX
}

report() {
    printf 'fabric-facts head   : if=%s hca=%s gid=%s ip=%s\n' \
        "$HEAD_IF" "$HEAD_IB" "$HEAD_GIDX" "$HEAD_IP"
    printf 'fabric-facts worker : if=%s hca=%s gid=%s ip=%s ssh=%s\n' \
        "$WORKER_IF" "$WORKER_IB" "$WORKER_GIDX" "$WORKER_IP" "$WORKER_SSH"
}
