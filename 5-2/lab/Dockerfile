# 5-2 실습용 JupyterLab: 5-1 GPU 이미지(ai4-lab/jupyter-clip:gpu = PyTorch CUDA + transformers) + diffusers
#   5-1 이미지를 그대로 두고 그 위에만 쌓는다
#
#   전제 : 5-1 에서 GPU 이미지를 한 번 빌드했다
#     cd ../5-1 && docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
#
#   diffusers   : SD1.5 를 부품(글 인코더 · U-Net · VAE · 스케줄러)으로 나눠 직접 돌림(01~03)
#   accelerate  : diffusers 가 큰 웨이트를 빨리 올릴 때 씀
#   requests    : ComfyUI API 호출(03 · 05)
#   gguf        : 양자화 모델 파일(.gguf)의 웨이트 이름 · 모양만 읽기(04) — 모델을 올리지 않음
FROM ai4-lab/jupyter-clip:gpu

RUN pip install --no-cache-dir diffusers==0.40.* accelerate==1.15.* requests "gguf>=0.13"

ENV HF_HOME=/cache/hf \
    HF_HUB_VERBOSITY=error \
    TRANSFORMERS_VERBOSITY=error \
    DIFFUSERS_VERBOSITY=error
