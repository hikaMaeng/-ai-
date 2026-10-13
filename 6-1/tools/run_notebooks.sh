#!/usr/bin/env bash
# 노트북을 실행해 출력이 든 사본을 notebooks/_out/ 에 쓴다(강사용 검증). 원본 notebooks/*.ipynb 는 출력 없이 둔다
#   bash tools/run_notebooks.sh 01 02      ← 번호를 주면 그 노트북만, 없으면 전부
#   GPU 번호 : VOICE_GPU=1 bash tools/run_notebooks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(cd .. && pwd -W 2>/dev/null || (cd .. && pwd))"
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(01 02 03 04 05 06 07)
for n in "${want[@]}"; do
  nb=$(ls notebooks/${n}_*.ipynb | head -1); name=$(basename "$nb")
  echo "== $name $(date +%H:%M:%S)"
  start=$(date +%s)
  MSYS_NO_PATHCONV=1 docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="${VOICE_GPU:-0}" \
    -v "$ROOT:/work" -v ai4-lab_hfcache:/cache -w /work/6-1/notebooks ai4-lab/jupyter-voice \
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 --output-dir _out "$name" 2>&1 | grep -v -E "^\[NbConvertApp\] (Writing|Converting)" || true
  echo "   $(( $(date +%s) - start ))초"
done
