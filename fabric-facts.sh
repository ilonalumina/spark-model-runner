#!/usr/bin/env bash
# Derive the fabric facts for the two nodes at launch time.
#
# Exists because the RoCEv2 GID index is a position in a table that renumbers.
# It moves whenever an interface's addresses change, which includes every boot
# that brings the fabric up. A recorded index is a measurement, not a constant;
# a stale one does not fail at startup, it kills that rank about 60 s in with
# ibv_modify_qp errno 61.
#
# Emits, per node: interface, HCA, GID index. Recipe-neutral on purpose; the
# adapters translate these into whatever names their recipe uses.
#
# Sourced by the runner, which has already sourced and exported the config
# file (HEAD_IP, WORKER_IP, WORKER_SSH). Those exports are what the child
# launcher sees too, so the worker probed here is the worker launched.
#
# Every remote probe is bounded: ssh gets a connect timeout and the remote
# command runs under `timeout`, both inside the budget the caller passes, so a
# stalled worker cannot outlive the fabric window.

# Resolve one node's fabric facts from its own GID table.
#
# show_gids prints: DEV PORT INDEX GID IPv4 VER NETDEV
# The same IPv4 appears twice, one index apart: a v1 row and a v2 row. RoCEv2
# needs the v2 row, so match on the VER column rather than taking the first
# line carrying the address.
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

    # A down HCA still has a GID table. Pinning one is a silent hang rather than
    # an error, so refuse here where the message is readable.
    # Read whole before matching: the runner sets pipefail, and grep -q leaving the pipe at the first match
    # kills ibdev2netdev with SIGPIPE, which pipefail reports as a failed match (a port read as down). A failed
    # ibdev2netdev is still a refusal: its output is not trusted.
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

# ssh_bounded <seconds> <remote command...>
# Connection and remote execution both fall inside <seconds>; minimum 1.
#
# No --foreground: in that mode GNU timeout does not time out the command's
# children, so a killed ssh could leave a child holding the output pipe and the
# caller's $(...) would wait on it anyway. Without it the whole process group is
# killed. -k gives a stubborn one a KILL two seconds after TERM.
ssh_bounded() {
    local budget="$1"; shift
    [ "$budget" -ge 1 ] 2>/dev/null || budget=1
    timeout -k 2 "$(( budget + 2 ))" \
        ssh -o BatchMode=yes -o ConnectTimeout="$budget" "$WORKER_SSH" \
            "timeout ${budget}s bash -c $(printf '%q' "$*")"
}

# derive <deadline-epoch-seconds>
# Sets and exports HEAD_IF HEAD_IB HEAD_GIDX WORKER_IF WORKER_IB WORKER_GIDX.
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

# Print what was derived, so a failed launch leaves the evidence in the journal
# rather than requiring the operator to re-derive it by hand afterwards.
report() {
    printf 'fabric-facts head   : if=%s hca=%s gid=%s ip=%s\n' \
        "$HEAD_IF" "$HEAD_IB" "$HEAD_GIDX" "$HEAD_IP"
    printf 'fabric-facts worker : if=%s hca=%s gid=%s ip=%s ssh=%s\n' \
        "$WORKER_IF" "$WORKER_IB" "$WORKER_GIDX" "$WORKER_IP" "$WORKER_SSH"
}
