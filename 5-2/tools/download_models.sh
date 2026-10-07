#!/usr/bin/env bash
# 5-2 실습 모델 받기 — ../4-1/comfyui/models 아래에 ComfyUI 폴더 규칙대로 둔다 (약 77GB)
#
#   cd ./-ai-/5-2
#   bash tools/download_models.sh            # 전부
#   bash tools/download_models.sh --core     # 05 셀 11(인코더 바꿔치기)용 Qwen3.5 둘(약 14GB)은 빼고
#
#   · 이미 받은 파일은 크기가 맞으면 건너뜀. 끊기면 다시 실행하면 이어 받음(curl -C -)
#   · 받는 곳은 전부 Hugging Face 공개 저장소(로그인 불필요). 라이선스는 README 「모델」 표
#   · 윈도는 Git Bash 에서 실행
set -euo pipefail
HF=https://huggingface.co
DIR="$(cd "$(dirname "$0")/../../4-1" && pwd)/comfyui/models"
CORE=0; [ "${1:-}" = "--core" ] && CORE=1

get() {   # get <폴더> <파일 이름> <크기> <저장소 경로>
  local out="$DIR/$1/$2"
  mkdir -p "$DIR/$1"
  if [ -f "$out" ] && [ "$(wc -c < "$out")" = "$3" ]; then echo "있음   $1/$2"; return; fi
  echo "받기   $1/$2 ($(( $3 / 1000000 ))MB)"
  curl -fL -C - --retry 5 -o "$out" "$HF/$4"
  [ "$(wc -c < "$out")" = "$3" ] || { echo "크기 다름 : $out"; exit 1; }
}

# SD1.5 — 01~03 원리 · 05 기본형 (U-Net 디퓨전)
get checkpoints      v1-5-pruned-emaonly.safetensors      4265146304  stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
# FLUX.1 — DiT 12B (4비트 GGUF) + T5-XXL · CLIP-L + VAE 16채널
get diffusion_models flux1-dev-Q4_K_S.gguf                6805988640  city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf
get diffusion_models flux1-schnell-Q4_K_S.gguf            6783943712  city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_K_S.gguf
get text_encoders    t5xxl_fp8_e4m3fn.safetensors         4893934904  comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors
get text_encoders    clip_l.safetensors                   246144152   comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
get vae              ae.safetensors                       335304388   Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors
# Qwen-Image — MMDiT 20B (4비트 GGUF) + Qwen2.5-VL-7B + VAE 16채널
get diffusion_models qwen-image-Q4_K_M.gguf               13065746976 city96/Qwen-Image-gguf/resolve/main/qwen-image-Q4_K_M.gguf
get text_encoders    Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf   5444830080  unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf
get vae              qwen_image_vae.safetensors           253806246   Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
# Z-Image — 단일 스트림 DiT 6B + Qwen3-4B + VAE(FLUX 와 같은 파일)
get diffusion_models z_image_bf16.safetensors             12309866400 Comfy-Org/z_image/resolve/main/split_files/diffusion_models/z_image_bf16.safetensors
get text_encoders    qwen_3_4b.safetensors                8044982048  Comfy-Org/z_image/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
get vae              z_image_ae.safetensors               335304388   Comfy-Org/z_image/resolve/main/split_files/vae/ae.safetensors
# SDXL VAE — 03 셀 8 의 4채널 비교
get vae              sdxl_vae.safetensors                 334641164   stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
# Qwen3.5 — 05 셀 11 조건 인코더 바꿔치기
if [ $CORE = 0 ]; then
get text_encoders    qwen3.5_4b_bf16.safetensors          9319828320  Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_4b_bf16.safetensors
get text_encoders    qwen3.5_2b_bf16.safetensors          4548221488  Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_2b_bf16.safetensors
fi
# ComfyUI-GGUF 노드 (GGUF 파일을 읽음)
NODE="$(dirname "$DIR")/custom_nodes/ComfyUI-GGUF"
[ -d "$NODE" ] || git clone --depth 1 https://github.com/city96/ComfyUI-GGUF "$NODE"
echo "완료 : $DIR"
