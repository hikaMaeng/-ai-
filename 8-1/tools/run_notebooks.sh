#!/usr/bin/env bash
# 노트북을 실행해 출력이 든 사본을 notebooks/_out/ 에 쓴다(강사용 검증). 원본 notebooks/*.ipynb 는 출력 없이 둔다
#   bash tools/run_notebooks.sh 01 02      ← 번호를 주면 그 노트북만, 없으면 전부
#   실행 장소 : 저장소 루트 compose 의 Jupyter 컨테이너(ai4-jupyter) 안 — 새 컨테이너를 만들지 않는다
#     먼저 저장소 루트에서 docker compose up -d   (GPU 번호는 루트 .env 의 LAB_GPU)
#   서버 없이 코드 경로만 시험 : AGENT_DRY=1 bash tools/run_notebooks.sh 01   (tools/dry/dry_NN.py 의 가짜 LLM 을 씀 — 숫자는 의미 없음)
#   LM Studio 주소 · 모델 · 모드 : 루트 .env 의 LMSTUDIO_URL · AGENT_MODEL · AGENT_MODE · AGENT_MIN_TOKENS 를 컨테이너가 받는다
#     이번 실행만 바꾸기       : LMSTUDIO_URL=http://host.docker.internal:12345/v1 AGENT_MODEL=qwen3.5-4b-mtp bash tools/run_notebooks.sh 01
set -euo pipefail
cd "$(dirname "$0")/../.."          # 저장소 루트(docker-compose.yml 이 있는 곳)
if [ -z "$(docker compose ps --status running -q jupyter)" ]; then
  echo "Jupyter 컨테이너가 꺼져 있다 → 저장소 루트에서 docker compose up -d" >&2; exit 1
fi
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(01 02 03 04 05 06)
for n in "${want[@]}"; do
  nb=$(ls 8-1/notebooks/${n}_*.ipynb | head -1); name=$(basename "$nb")
  echo "== $name $(date +%H:%M:%S)"
  start=$(date +%s)
  ENVARGS=()
  for v in LMSTUDIO_URL AGENT_MODEL AGENT_MODE AGENT_MIN_TOKENS; do
    if [ -n "${!v:-}" ]; then ENVARGS+=(-e "$v=${!v}"); fi
  done
  if [ -n "${AGENT_DRY:-}" ]; then ENVARGS+=(-e "AGENT_DRY=$n"); fi
  MSYS_NO_PATHCONV=1 docker compose exec -T "${ENVARGS[@]}" -w /work/8-1/notebooks jupyter \
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 --output-dir _out "$name" 2>&1 | grep -v -E "^\[NbConvertApp\] (Writing|Converting)" || true
  echo "   $(( $(date +%s) - start ))초"
done
