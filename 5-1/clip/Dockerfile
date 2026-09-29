# 5-1 실습용 JupyterLab: 4-1 Jupyter 이미지 + CPU PyTorch + transformers (CLIP 구동)
#   4-1 이미지를 그대로 두고 그 위에만 쌓는다 → 4-1 · 4-2 실습 이미지는 무거워지지 않는다
#   GPU 없이 CPU 로 돈다. CLIP ViT-B/32 는 이미지 150장 인코딩에 CPU 로 30초 안팎
FROM ai4-lab/jupyter

# PyTorch 는 CPU 전용 휠(약 200MB)을 받는다. 기본 PyPI 휠은 CUDA 포함이라 수 GB
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.* \
 && pip install --no-cache-dir transformers==5.17.* safetensors

# 모델 캐시는 hfcache 볼륨(/cache/hf)에 → 컨테이너를 지워도 다시 받지 않는다
#   HF_HUB_VERBOSITY : 로그인 없이 받을 때 뜨는 안내 문구를 숨김(받기는 그대로 된다)
ENV HF_HOME=/cache/hf \
    HF_HUB_VERBOSITY=error \
    TRANSFORMERS_VERBOSITY=error
