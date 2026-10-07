#!/bin/bash
# ComfyUI-GGUF 노드가 쓰는 파이썬 패키지를 ComfyUI 의 venv 에 설치(이미 있으면 건너뜀)
#   이미지의 시작 스크립트가 /userscripts_dir/*.sh 를 이름 순서로 실행한다
set -e
if python3 -c "import gguf, sentencepiece" 2>/dev/null; then
  echo "== gguf · sentencepiece 이미 설치됨"
else
  echo "== gguf · sentencepiece 설치"
  python3 -m pip install "gguf>=0.13.0" sentencepiece protobuf
fi
