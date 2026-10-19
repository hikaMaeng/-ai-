#!/usr/bin/env bash
# 노트북을 실행해 출력이 든 사본을 notebooks/_out/ 에 쓴다(강사용 검증). 원본 notebooks/*.ipynb 는 출력 없이 둔다
#   bash tools/run_notebooks.sh 01 02      ← 번호를 주면 그 노트북만, 없으면 전부
#   실행 장소 : 저장소 루트 compose 의 Jupyter 컨테이너(ai4-jupyter) 안 — 새 컨테이너를 만들지 않는다
#     먼저 저장소 루트에서 docker compose up -d   (GPU 번호는 루트 .env 의 LAB_GPU)
#   GPU 없이(CPU 로) : WM_GPU=none bash tools/run_notebooks.sh
set -euo pipefail
cd "$(dirname "$0")/../.."          # 저장소 루트(docker-compose.yml 이 있는 곳)
if [ -z "$(docker compose ps --status running -q jupyter)" ]; then
  echo "Jupyter 컨테이너가 꺼져 있다 → 저장소 루트에서 docker compose up -d" >&2; exit 1
fi
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(01 02 03 04 05)
ENVARGS=(); if [ "${WM_GPU:-}" = "none" ]; then ENVARGS=(-e CUDA_VISIBLE_DEVICES=); fi
for n in "${want[@]}"; do
  nb=$(ls 6-2/notebooks/${n}_*.ipynb | head -1); name=$(basename "$nb")
  echo "== $name $(date +%H:%M:%S)"
  start=$(date +%s)
  MSYS_NO_PATHCONV=1 docker compose exec -T "${ENVARGS[@]}" -w /work/6-2/notebooks jupyter \
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 --output-dir _out "$name" 2>&1 | grep -v -E "^\[NbConvertApp\] (Writing|Converting)" || true
  echo "   $(( $(date +%s) - start ))초"
done
