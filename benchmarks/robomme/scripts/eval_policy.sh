#!/usr/bin/env bash
set -eu

exec python "$(dirname "${BASH_SOURCE[0]}")/../client.py" "$@"
