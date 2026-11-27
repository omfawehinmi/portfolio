#!/usr/bin/env bash
# Keep Vite on 0.0.0.0:5173. Survives Vite crashes; log to file.
set -u
cd /agent/repos/portfolio
LOG=/tmp/portfolio-dev-logs/vite.log
mkdir -p "$(dirname "$LOG")"
export PATH="/usr/local/bin:/usr/bin:$PATH"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog start pid=$$" | tee -a "$LOG"
while true; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting vite :5173" | tee -a "$LOG"
  npm run dev -- --host 0.0.0.0 --port 5173 --strictPort >>"$LOG" 2>&1
  code=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] vite exited $code - restarting in 2s" | tee -a "$LOG"
  sleep 2
done
