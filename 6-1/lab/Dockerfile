# 6-1 실습용 JupyterLab: 4-1 Jupyter 이미지 + 음성용 PyTorch 묶음
#   5-1 · 5-2 이미지(torch 2.14 · transformers 5.17)와 섞지 않고 따로 만든다
#     · torchaudio 는 2.11 이 마지막 판 → torch 도 2.11 로 맞춤(Qwen3-TTS 가 torchaudio 의 fbank 를 씀)
#     · qwen-asr · qwen-tts 가 transformers 4.57.x 를 고정
#
#   gcc · g++ : triton · torch.compile 이 커널을 그 자리에서 컴파일(Mimi · Qwen3-ASR · AudioSeal 이 씀)
#   sox   : qwen-tts 가 기동 때 찾는 명령
#   torchvision : Qwen2.5-Omni 프로세서가 영상 처리기를 함께 불러옴(오디오만 써도 필요)
FROM ai4-lab/jupyter

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libc6-dev sox libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch==2.11.* torchaudio==2.11.* torchvision==0.26.* \
 && pip install --no-cache-dir transformers==4.57.6 accelerate==1.12.0 safetensors \
      librosa soundfile editdistance pandas matplotlib \
 && pip install --no-cache-dir qwen-asr==0.0.6 \
 && pip install --no-cache-dir --no-deps qwen-tts==0.1.1 \
 && pip install --no-cache-dir onnxruntime einops audioseal==0.2.0

ENV HF_HOME=/cache/hf \
    HF_HUB_VERBOSITY=error \
    TRANSFORMERS_VERBOSITY=error
