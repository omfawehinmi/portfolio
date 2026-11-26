#!/usr/bin/env bash
# Quick tunnel so other agents/VMs can hit this Vite without sharing loopback.
set -u
LOG=/tmp/portfolio-dev-logs/tunnel.log
mkdir -p "$(dirname "$LOG")"
cd /tmp
# Prefer binary if present; else npx
if command -v cloudflared >/dev/null 2>&1; then
  CF=cloudflared
elif [ -x /tmp/cloudflared ]; then
  CF=/tmp/cloudflared
else
  CF="npx --yes cloudflared@latest"
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting tunnel -> http://127.0.0.1:5173" | tee -a "$LOG"
# shellcheck disable=SC2086
exec $CF tunnel --url http://127.0.0.1:5173 --no-autoupdate 2>&1 | tee -a "$LOG"
