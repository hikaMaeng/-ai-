# 실습 JupyterLab 이미지 — 모든 강의(3-1 ~ 8-1)가 이 이미지 하나 · 컨테이너 하나(ai4-jupyter)를 쓴다
#
#   파이썬 커널 둘
#     ① 「Python 3 (기본)」  (커널 이름 python3, 시스템 python 3.12)
#        3-1 · 4-1 · 4-2 · 5-1 · 5-2 · 6-2 · 8-1
#        torch 2.14(CUDA 13.0) · transformers 5.17 · diffusers 0.40 · psycopg · pgvector …
#     ② 「Python 3 (음성 · 6-1)」 (커널 이름 voice, venv /opt/voice)
#        6-1 만. torch 2.11 · torchaudio 2.11 · transformers 4.57.6 · qwen-asr · qwen-tts
#        torchaudio 는 2.11 이 마지막 판이고 qwen-asr · qwen-tts 가 transformers 4.57.x 를 고정 → ① 과 섞지 않는다
#
#   빌드 : 저장소 루트에서 docker compose build jupyter
#   MPLBACKEND 는 지정하지 않는다(Agg 로 두면 JupyterLab 화면에 그림이 나오지 않는다)
FROM python:3.12-slim

# fonts-nanum : 그래프 한글 표시
# gcc · g++ · libc6-dev : triton · torch.compile 이 커널을 그 자리에서 컴파일(6-1 의 Mimi · Qwen3-ASR · AudioSeal)
# sox : qwen-tts 가 기동 때 찾는 명령 · libsndfile1 · ffmpeg : 음성 파일 읽기 · 쓰기(6-1)
RUN apt-get update && apt-get install -y --no-install-recommends \
      fonts-nanum gcc g++ libc6-dev sox libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── 커널 ① 기본 ─────────────────────────────────────────────────────
#   jupyterlab · numpy · pandas · pyarrow · matplotlib · requests · psycopg · pgvector : 3-1 · 4-1 · 4-2 (옛 4-1/jupyter/requirements.txt)
#   torch · torchvision · transformers · safetensors : 5-1 (torchvision 은 Qwen3-VL 프로세서가 씀, 0.29 ↔ torch 2.14)
#   diffusers · accelerate · gguf : 5-2 (gguf 는 .gguf 웨이트 이름 · 모양만 읽음)
#   scikit-image : 6-2 의 05 가 손 계산 PSNR 과 비교할 때 씀
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.14.* torchvision==0.29.* \
 && pip install --no-cache-dir \
      jupyterlab==4.4.* ipykernel \
      numpy==2.3.* pandas==2.3.* pyarrow==21.* matplotlib==3.10.* requests==2.32.* \
      "psycopg[binary]==3.2.*" pgvector==0.4.* \
      transformers==5.17.* diffusers==0.40.* accelerate==1.15.* "gguf>=0.13" \
      safetensors huggingface_hub pillow scikit-image

# ── 커널 ② 음성(6-1) ────────────────────────────────────────────────
#   옛 6-1 이미지는 FROM ai4-lab/jupyter(옛 4-1/jupyter/requirements.txt) 위에 쌓았다 → 같은 바탕 패키지를 먼저 깐다
#     (pyarrow : voice_common 이 FLEURS parquet 을 읽음. numpy · pandas · matplotlib 판도 옛 이미지와 맞춤)
#   그 뒤는 옛 6-1/lab/Dockerfile 의 설치 순서 그대로(qwen-tts 는 --no-deps : 의존성을 transformers 4.57.6 판에 맞춰 둔다)
#   torchvision : Qwen2.5-Omni 프로세서가 영상 처리기를 함께 불러옴(오디오만 써도 필요)
RUN python -m venv /opt/voice \
 && /opt/voice/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/voice/bin/pip install --no-cache-dir \
      numpy==2.3.* pandas==2.3.* pyarrow==21.* matplotlib==3.10.* requests==2.32.* \
      "psycopg[binary]==3.2.*" pgvector==0.4.* \
 && /opt/voice/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.11.* torchaudio==2.11.* torchvision==0.26.* \
 && /opt/voice/bin/pip install --no-cache-dir transformers==4.57.6 accelerate==1.12.0 safetensors \
      librosa soundfile editdistance pandas matplotlib ipykernel \
 && /opt/voice/bin/pip install --no-cache-dir qwen-asr==0.0.6 \
 && /opt/voice/bin/pip install --no-cache-dir --no-deps qwen-tts==0.1.1 \
 && /opt/voice/bin/pip install --no-cache-dir onnxruntime einops audioseal==0.2.0

# 커널 등록 : 같은 JupyterLab 의 커널 목록에 둘 다 보이게
RUN python -m ipykernel install --prefix /usr/local --name python3 --display-name "Python 3 (기본)" \
 && /opt/voice/bin/python -m ipykernel install --prefix /usr/local --name voice --display-name "Python 3 (음성 · 6-1)"

# JupyterLab 기본 설정: 노트북의 모든 셀을 항상 그린다(windowing 끔)
#   기본값은 화면 밖 셀을 그리지 않아, 탭을 다시 열면 노트북이 빈 화면으로 보이는 일이 있다
RUN mkdir -p /usr/local/share/jupyter/lab/settings \
 && printf '{"@jupyterlab/notebook-extension:tracker": {"windowingMode": "none"}}\n' \
      > /usr/local/share/jupyter/lab/settings/overrides.json

# /work = 저장소 루트 · /cache = hfcache 볼륨(모델 · 데이터셋 캐시, compose 에서 마운트)
#   HF_HUB_VERBOSITY : 로그인 없이 받을 때 뜨는 안내 문구를 숨김(받기는 그대로 된다)
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_VERBOSITY=error \
    TRANSFORMERS_VERBOSITY=error \
    DIFFUSERS_VERBOSITY=error
WORKDIR /work

# 로그인 토큰 없이 바로 쓰는 로컬 실습용 설정 (외부에 노출하지 말 것 — compose 가 127.0.0.1 에만 연다)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--IdentityProvider.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/work", \
     "--LabApp.default_url=/lab"]
