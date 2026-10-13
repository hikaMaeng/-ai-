# 6-2 실습용 JupyterLab: 4-1 Jupyter 이미지 + PyTorch(합성 데이터 · 직접 구현한 시뮬레이터만 쓰므로 다른 모델 라이브러리는 없음)
#   음성 · 이미지 실습 이미지와 섞지 않고 따로 만든다(버전 충돌 방지)
FROM ai4-lab/jupyter

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch==2.11.* \
 && pip install --no-cache-dir numpy matplotlib pandas

ENV MPLBACKEND=Agg
