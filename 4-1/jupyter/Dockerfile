FROM python:3.12-slim

# 그래프 한글 표시용 나눔 글꼴
RUN apt-get update && apt-get install -y --no-install-recommends fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# JupyterLab 기본 설정: 노트북의 모든 셀을 항상 그린다(windowing 끔)
#   기본값은 화면 밖 셀을 그리지 않아, 탭을 다시 열면 노트북이 빈 화면으로 보이는 일이 있다
RUN mkdir -p /usr/local/share/jupyter/lab/settings \
 && printf '{"@jupyterlab/notebook-extension:tracker": {"windowingMode": "none"}}\n' \
      > /usr/local/share/jupyter/lab/settings/overrides.json

# /work = 저장소 루트 (compose 에서 마운트)
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1
WORKDIR /work

# 로그인 토큰 없이 바로 쓰는 로컬 실습용 설정 (외부에 노출하지 말 것)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--IdentityProvider.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/work", \
     "--LabApp.default_url=/lab"]
