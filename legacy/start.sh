#!/bin/bash
set -e
PIDFILE=/tmp/hcu-train-web.pid
LOGFILE=/tmp/hcu-train-web.log
PORT="${PORT:-7766}"
HOST="${HOST:-0.0.0.0}"

case "${1:-start}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running on PID $(cat "$PIDFILE")"
      exit 0
    fi
    : > "$LOGFILE"
    HOST="$HOST" PORT="$PORT" python3 /tmp/launch_backend.py &
    sleep 2
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      echo "started PID=$PID  → http://$HOST:$PORT/"
      echo "log:    $LOGFILE"
    else
      # extract pid by name
      PID=$(pgrep -f "/tmp/launch_backend.py" | head -1)
      echo "$PID" > "$PIDFILE"
      echo "started PID=$PID  → http://$HOST:$PORT/"
    fi
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; fi
    pkill -f "/tmp/launch_backend.py" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "stopped"
    ;;
  restart) "$0" stop; "$0" start ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running PID=$(cat "$PIDFILE") on port $PORT"
    else
      echo "not running"
    fi
    ;;
  *) echo "usage: $0 start|stop|restart|status"; exit 1 ;;
esac
