# 5-1 실습용 JupyterLab: 4-1 Jupyter 이미지 + PyTorch + transformers (CLIP 구동)
#   4-1 이미지를 그대로 두고 그 위에만 쌓는다 → 4-1 · 4-2 실습 이미지는 무거워지지 않는다
#
#   TORCH_INDEX : PyTorch 를 받을 곳
#     기본(CPU)  https://download.pytorch.org/whl/cpu   약 200MB · GPU 없는 PC 도 그대로 동작
#     GPU        https://pypi.org/simple                리눅스용 PyPI 기본 휠 = CUDA 포함(수 GB)
#                → docker-compose.gpu.yml 이 이 값으로 바꿔 빌드한다
FROM ai4-lab/jupyter

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch==2.14.* \
 && pip install --no-cache-dir transformers==5.17.* safetensors

# 모델 캐시는 hfcache 볼륨(/cache/hf)에 → 컨테이너를 지워도 다시 받지 않는다
#   HF_HUB_VERBOSITY : 로그인 없이 받을 때 뜨는 안내 문구를 숨김(받기는 그대로 된다)
ENV HF_HOME=/cache/hf \
    HF_HUB_VERBOSITY=error \
    TRANSFORMERS_VERBOSITY=error
