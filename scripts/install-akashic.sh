#!/bin/sh
set -eu

origin="${AKASHIC_INSTALL_ORIGIN:-https://github.com/kachofugetsu09/akashic-agent.git}"
requested_commit=""
forward_args=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --commit)
            [ "$#" -ge 2 ] || { echo "--commit requires a value" >&2; exit 2; }
            requested_commit="$2"
            forward_args="$forward_args --commit $2"
            shift 2
            ;;
        --yes|--no-activate)
            forward_args="$forward_args $1"
            shift
            ;;
        *)
            echo "unsupported bootstrap argument: $1" >&2
            exit 2
            ;;
    esac
done

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }

if [ -z "$requested_commit" ]; then
    requested_commit="$(git ls-remote --exit-code "$origin" refs/heads/main | awk 'NR == 1 {print $1}')"
    forward_args="$forward_args --commit $requested_commit"
fi
case "$requested_commit" in
    *[!0-9a-f]*|'') echo "target commit must be a lowercase SHA" >&2; exit 2 ;;
esac
[ "${#requested_commit}" -eq 40 ] || { echo "target commit must be 40 characters" >&2; exit 2; }

temporary="$(mktemp -d -t akashic-install.XXXXXX)"
trap 'rm -rf "$temporary"' EXIT HUP INT TERM
git -C "$temporary" init -q
git -C "$temporary" remote add origin "$origin"
git -C "$temporary" fetch --quiet --depth=1 origin "$requested_commit"
git -C "$temporary" checkout --quiet --detach FETCH_HEAD

# Arguments are restricted above to flags and lowercase SHA, so field splitting is intentional.
# shellcheck disable=SC2086
python3 "$temporary/scripts/akashic_release/cli.py" install \
    --source-checkout "$temporary" --origin "$origin" $forward_args
