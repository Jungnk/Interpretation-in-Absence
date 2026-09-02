#!/bin/bash
# 전시 실행 스크립트 — 크래시 시 자동 재시작
cd "$(dirname "$0")"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 시작" >> exhibition.log
    caffeinate -d -i /Users/jung/miniforge3/envs/face-inpainting/bin/python screen2.py
    EXIT_CODE=$?

    # q / ESC 정상 종료 (exit code 0) → 루프 탈출
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 정상 종료" >> exhibition.log
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 비정상 종료 (코드 $EXIT_CODE) — 5초 후 재시작" >> exhibition.log
    echo "비정상 종료 (코드 $EXIT_CODE) — 5초 후 재시작..."
    sleep 5
done
