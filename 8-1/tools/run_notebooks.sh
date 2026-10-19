#!/usr/bin/env bash
# 노트북을 실행해 출력이 든 사본을 notebooks/_out/ 에 쓴다(강사용 검증). 원본 notebooks/*.ipynb 는 출력 없이 둔다
#   bash tools/run_notebooks.sh 01 02      ← 번호를 주면 그 노트북만, 없으면 전부
#   서버 없이 코드 경로만 시험 : AGENT_DRY=1 bash tools/run_notebooks.sh 01   (tools/dry/dry_NN.py 의 가짜 LLM 을 씀 — 숫자는 의미 없음)
#   LM Studio 포트 · 모델       : LMSTUDIO_PORT=12345 AGENT_MODEL=qwen3.5-4b-mtp bash tools/run_notebooks.sh 01
#   이미지                      : AGENT_IMAGE=ai4-lab/jupyter-agent (기본 ai4-lab/jupyter-voice — matplotlib · pandas · requests 가 들어 있음)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(cd .. && pwd -W 2>/dev/null || (cd .. && pwd))"
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(01 02 03 04 05)
IMG="${AGENT_IMAGE:-ai4-lab/jupyter-voice}"
for n in "${want[@]}"; do
  nb=$(ls notebooks/${n}_*.ipynb | head -1); name=$(basename "$nb")
  echo "== $name $(date +%H:%M:%S)"
  start=$(date +%s)
  ENVARGS=(-e "LMSTUDIO_URL=http://host.docker.internal:${LMSTUDIO_PORT:-1234}/v1" -e "AGENT_MODEL=${AGENT_MODEL:-}" -e "AGENT_MODE=${AGENT_MODE:-auto}")
  [ -n "${AGENT_DRY:-}" ] && ENVARGS+=(-e "AGENT_DRY=$n")
  MSYS_NO_PATHCONV=1 docker run --rm --add-host host.docker.internal:host-gateway "${ENVARGS[@]}" \
    -v "$ROOT:/work" -w /work/8-1/notebooks "$IMG" \
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 --output-dir _out "$name" 2>&1 | grep -v -E "^\[NbConvertApp\] (Writing|Converting)" || true
  echo "   $(( $(date +%s) - start ))초"
done
