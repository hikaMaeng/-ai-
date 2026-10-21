# CUDA 첫 초기화가 느릴 때 어느 층에서 시간을 쓰는지 가른다 — ai4-jupyter 안에서 실행
#
#   docker compose exec jupyter python /work/docker/lab/cuda_init_check.py
#
# 단계마다 새 프로세스를 띄워 각 단계가 그 프로세스의 「첫 호출」이 되게 한다(한 프로세스 안에서는 두 번째부터 빠르다)
#   ① 드라이버 파일 읽기 : WSL2 에서 libcuda 등은 이미지가 아니라 Windows DriverStore 를 9p 로 공유한 경로에서 읽힌다
#                          (페이지 캐시에 남아 있으면 9p 를 거치지 않으므로, 느린 상태에서 처음 돌린 값이 의미 있다)
#   ② nvidia-smi -L     : NVML(관리 라이브러리) 경로
#   ③ cuInit            : 드라이버 API 초기화(torch 없이 ctypes 로 직접)
#   ④ cudaGetDeviceCount : torch 에 딸린 CUDA 런타임(libcudart)
#   ⑤ torch.cuda.is_available() : 노트북이 실제로 기다리는 값(import torch 시간은 따로 표시)
# ③ 이 느리면 이미지 · compose 가 아니라 드라이버 · WSL2 GPU 통로 쪽 문제다 → README 「CUDA 첫 초기화가 느릴 때」
import glob
import os
import subprocess
import sys
import time

STEPS = {
    "cuInit": """
import ctypes, time
lib = ctypes.CDLL("libcuda.so.1")
t = time.time(); r = lib.cuInit(0); dt = time.time() - t
n = ctypes.c_int(); lib.cuDeviceGetCount(ctypes.byref(n))
print(f"{dt:.2f}|반환 {r} · 장치 {n.value}")
""",
    "cudaGetDeviceCount": """
import ctypes, glob, time
p = sorted(glob.glob("/usr/local/lib/python3.12/site-packages/nvidia/**/libcudart.so*", recursive=True))[0]
lib = ctypes.CDLL(p)
n = ctypes.c_int(); t = time.time(); r = lib.cudaGetDeviceCount(ctypes.byref(n)); dt = time.time() - t
print(f"{dt:.2f}|반환 {r} · 장치 {n.value} · {p.split('/')[-1]}")
""",
    "torch.cuda.is_available": """
import time
t0 = time.time(); import torch; t1 = time.time()
ok = torch.cuda.is_available(); dt = time.time() - t1
print(f"{dt:.2f}|{ok} · 장치 {torch.cuda.device_count()} · import torch {t1 - t0:.1f}s · torch {torch.__version__}")
""",
}


def read_driver_files():
    total, t = 0, time.time()
    for p in glob.glob("/usr/lib/wsl/drivers/*/*"):
        if os.path.isfile(p):
            with open(p, "rb") as f:
                total += len(f.read())
    dt = time.time() - t
    if not total:
        return dt, "WSL 드라이버 공유 없음(WSL2 가 아닌 환경)"
    return dt, f"{total / 2**20:.0f} MB · {total / 2**20 / max(dt, 1e-6):.0f} MB/s"


def run(code):
    t = time.time()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    out = r.stdout.strip().splitlines()
    if r.returncode or not out or "|" not in out[-1]:
        return time.time() - t, "실패 : " + (r.stderr.strip().splitlines() or ["(출력 없음)"])[-1]
    dt, note = out[-1].split("|", 1)
    return float(dt), note


def main():
    print(f"시각(UTC) {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} · "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(없음)')}")
    rows = [("드라이버 파일 읽기(9p)", *read_driver_files())]
    t = time.time()
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    rows.append(("nvidia-smi -L", time.time() - t, f"GPU {len(smi.stdout.strip().splitlines())}개 보임"))
    for name, code in STEPS.items():
        rows.append((name, *run(code)))
    for name, dt, note in rows:
        print(f"{name:<26} {dt:8.2f} s   {note}")


if __name__ == "__main__":
    main()
