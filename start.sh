#!/usr/bin/env bash
# Idempotent background start.
cd "$(dirname "$0")" || exit 1
pgrep -f "python3 -m dtrelay" >/dev/null && exit 0
nohup ./run.sh >> bot.log 2>&1 &
