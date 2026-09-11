#!/usr/bin/env bash
# Foreground. Sources .env if present.
set -a; [ -f "$(dirname "$0")/.env" ] && . "$(dirname "$0")/.env"; set +a
cd "$(dirname "$0")" || exit 1
exec /Users/long.nguyen5/miniconda3/bin/python3 -m dtrelay
