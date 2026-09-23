FROM python:3.12-slim

# 그래프 한글 표시용 나눔 글꼴
RUN apt-get update && apt-get install -y --no-install-recommends fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# /work = 저장소 루트 (compose 에서 마운트). 4-1 의 load.py 를 노트북에서 import 할 수 있게 한다
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/work/4강실습/loader \
    HF_HUB_DISABLE_PROGRESS_BARS=1
WORKDIR /work

# 로그인 토큰 없이 바로 쓰는 로컬 실습용 설정 (외부에 노출하지 말 것)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--IdentityProvider.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/work", \
     "--LabApp.default_url=/lab/tree/4-2강실습/notebooks"]
