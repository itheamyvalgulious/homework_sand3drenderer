#!/usr/bin/env bash
# 一键启动:SAND 静态 PPT 站 + render server(gunicorn WSGI, 16200)
# 用法: ./serve.sh 启动; Ctrl-C 全部停止
# 可用环境变量覆盖端口: PPT_PORT=8100 RENDER_PORT=17000 ./serve.sh
set -e
cd "$(dirname "$0")"

VENV=.venv/bin

pick_port() {
  # 从给定端口起探测第一个空闲端口
  local p=$1
  while (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; do
    exec 3>&- 3<&- 2>/dev/null || true
    p=$((p + 1))
  done
  echo "$p"
}

PPT_PORT=$(pick_port "${PPT_PORT:-16490}")
RENDER_PORT=$(pick_port "${RENDER_PORT:-16200}")

PIDS=()
cleanup() {
  echo ""
  echo "Stopping servers..."
  kill "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting PPT static site  -> http://127.0.0.1:${PPT_PORT}/"
"$VENV/python" -m http.server "$PPT_PORT" --directory site --bind 0.0.0.0 &
PIDS+=($!)

echo "Starting render server    -> http://127.0.0.1:${RENDER_PORT}/  (gunicorn WSGI)"
"$VENV/gunicorn" -w 1 --threads 4 -b "0.0.0.0:${RENDER_PORT}" script.viz_server:app &
PIDS+=($!)

echo ""
echo "Both servers running. Press Ctrl-C to stop."
wait
