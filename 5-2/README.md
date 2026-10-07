# 5-2 실습 — 이미지 생성 : 디퓨전을 부품으로 열어 보고, ComfyUI 로 최신 모델을 부리기

> 현대 AI의 원리와 구조 #5-2 이미지 생성 실습 · 뉴런데브클래스
> 전제 : **NVIDIA GPU**(VRAM 16GB 이상 권장, 4~8GB 면 01~04 만) + 도커가 GPU 를 넘길 수 있는 PC. 5-1 의 GPU 이미지(`ai4-lab/jupyter-clip:gpu`)를 한 번 빌드했다
> 실습 화면 : Jupyter **http://localhost:8891** → `5-2/notebooks/`. ComfyUI 화면 **http://localhost:8188** 은 노트북이 저장한 워크플로를 열어 보거나 직접 돌려 볼 때만
> 모델 : `../4-1/comfyui/models` 에 둔다(약 77GB) → `bash tools/download_models.sh`

| 노트북 | 하는 일 | 엔진 | 장표 | 시간(실측) |
|---|---|---|---|---|
| `01_디퓨전_원리.ipynb` | SD1.5 를 부품으로 나눠 텐서 흐름 · 정방향 · 학습 손실 · 완성본 추정 · 손으로 쓴 루프 · 샘플러 · timestep | Jupyter(diffusers) | Diffusion 개념 ~ sampler | RTX 4080 104초 |
| `02_UNet과_프롬프트.ipynb` | U-Net 층별 모양 · skip · 교차 어텐션 크기 · 단어별 어텐션 지도 · 풀링 vs 토큰 · 조건 바꿔치기 · CFG · 조건 공간 | Jupyter(diffusers) | U-Net denoiser ~ U-Net과 타임스텝 · 교차 어텐션 · CFG | RTX 4080 76초 |
| `03_VAE와_latent.ipynb` | 48배 압축 · 4채널 · 스케일 계수 · μ σ 와 KL · latent 한 칸 · 4채널 vs 16채널 글자 복원 | Jupyter + ComfyUI | VAE ~ Latent 채널 | 30초(ComfyUI 모델이 올라가 있을 때) |
| `04_DiT와_플로매칭.ipynb` | 패치로 자르기 · 웨이트 모양만 읽어 세 모델 구조 · 어텐션 비용 · 플로 매칭 장난감 · 시프트 | Jupyter | DiT · DiT 텐서 흐름 · 플로 매칭 | 47초 |
| `05_ComfyUI로_최신모델.ipynb` | 워크플로 = 부품 목록 · SD1.5 · FLUX.1-dev · Qwen-Image · Z-Image 비교 · 단계별 추정 · CFG · 스텝 · 시프트 · 한국어 · 인코더 바꿔치기 | ComfyUI(API) | Qwen-Image 분석 · ComfyUI 섹션 · 세 모델 비교 | 44분 |

- 01~04 : 장표의 텐서 모양 · 손실 · 주장을 SD1.5 와 작은 실험으로 **숫자로** 확인한다. 직접 계산한 값과 라이브러리 값이 같은지(최대 차이 0)도 본다
- 05 : 같은 부품이 최신 모델에서 어떻게 바뀌었는지(DiT · 플로 매칭 · LLM/VLM 조건 · 16채널 VAE)를 ComfyUI 로 돌려 비교한다. 생성은 ComfyUI, 노트북은 **워크플로를 만들어 보내고 결과를 받는** 쪽

```
5-2/
├─ docker-compose.yml        comfy(ComfyUI, 8188) + lab(Jupyter, 8891) — 같은 도커 네트워크, 노트북 → http://comfy:8188
├─ lab/Dockerfile            ai4-lab/jupyter-clip:gpu + diffusers · accelerate · requests · gguf
├─ comfy/userscripts/        ComfyUI 기동 때 실행 : GGUF 노드의 파이썬 패키지 설치
├─ notebooks/                01~05. 코드 셀 첫 줄 = "# 셀 N · 제목", 설정은 셀 1
├─ workflows/                05 셀 3 이 저장 : *_t2i.json(화면 형식) · *_t2i_api.json(API 형식) → ComfyUI 워크플로 목록 「5-2」
├─ outputs/                  ComfyUI 결과 · 실험 산출물 (.gitignore)
├─ results/                  노트북 출력 그림(장표에 끌어다 놓는 용도)
├─ screens/                  ComfyUI 실제 실행 화면 캡처 · slide/ = 번호 표시 그림(강의 덱 33~40장)
└─ tools/                    강사용
   ├─ build_notebook_01.py ~ 05.py   노트북 생성기(셀 내용은 여기서 고친다)
   ├─ api_to_ui.py                   API 형식 워크플로 → 화면 형식(노드 위치 · 선)
   ├─ screens/                       화면 캡처 : capture.js(헤드리스 Chrome) · annotate.py(번호 표시) · preview_strip.py
   └─ download_models.sh             모델 받기 → ../4-1/comfyui/models
```

## 실행 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA, VRAM 16GB 이상 권장. 05 의 Qwen-Image(4비트 13GB + 조건 인코더 5.4GB)를 ComfyUI 가 번갈아 올림 |
| 확인 | `docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L` 가 GPU 를 출력해야 함 |
| 맥 · GPU 없는 PC | 지원하지 않음(5-2 는 GPU 필수). 05 결과는 실습 장표의 그림으로 본다 |
| 디스크 | 모델 약 77GB(`--core` 면 약 63GB) + 이미지(ComfyUI 11.8GB · lab 6.5GB+) |

```bash
cd ./-ai-/5-2
bash tools/download_models.sh
docker compose up -d --build
```

- 첫 기동 : `comfy` 가 ComfyUI 와 PyTorch 를 `comfyrun` 볼륨에 설치(약 5~10분, 인터넷 필요). `docker logs -f ai4-comfy` 에서 `To see the GUI go to` 가 나오면 준비 끝
- GPU 고르기 : `COMFY_GPU=1 LAB_GPU=1 docker compose up -d` — 번호는 `nvidia-smi -L` 순서. 기본은 둘 다 0
  - 다른 프로그램이 VRAM 을 쓰고 있는 GPU 는 피한다(`nvidia-smi` 로 확인). 남은 VRAM 이 모자라면 ComfyUI 가 모델을 나눠 올리며 크게 느려진다(실측 : Qwen-Image 한 스텝 6초 → 47초)
  - 둘이 같은 GPU 면 05 셀 12 처럼 `free()` 로 ComfyUI 메모리를 비운 뒤 Jupyter 쪽 계산을 한다
  - 강사 PC 실측 기준 : 둘 다 RTX 4080 16GB(3090 은 다른 작업이 사용 중)
- 미리보기 : compose 가 ComfyUI 를 `--preview-method latent2rgb` 로 띄움 → 실행 중 KSampler 안에 단계별 완성본 추정이 보인다(VAE 없이 latent 를 색으로 근사한 거친 그림)
- 도커 데스크톱은 `device_ids` 로 GPU 를 나누지 못해 `CUDA_VISIBLE_DEVICES` 로 지정한다(compose 주석)
- 모델 파일을 윈도 드라이브에서 바인드 마운트로 읽으므로 **첫 로드가 느리다**(RTX 4080 실측 : FLUX 12B 4비트 첫 장 131.6초, 모델이 올라가 있을 때 같은 크기 한 장 27초(05 셀 7)). 두 번째부터는 ComfyUI 가 메모리에 둔 것을 씀

## 모델

`../4-1/comfyui/models` 아래(ComfyUI 폴더 규칙). `download_models.sh` 가 크기까지 확인한다.

| 폴더 | 파일 | 크기 | 쓰는 곳 | 출처 · 라이선스 |
|---|---|---|---|---|
| checkpoints | `v1-5-pruned-emaonly.safetensors` | 4.27GB | 01~03 · 05 셀 4 | stable-diffusion-v1-5 · CreativeML OpenRAIL-M |
| diffusion_models | `flux1-dev-Q4_K_S.gguf` | 6.81GB | 04 · 05 | city96/FLUX.1-dev-gguf · FLUX.1-dev 비상업 라이선스 |
| diffusion_models | `flux1-schnell-Q4_K_S.gguf` | 6.78GB | 05 셀 8 | city96/FLUX.1-schnell-gguf · Apache 2.0 |
| text_encoders | `t5xxl_fp8_e4m3fn.safetensors` · `clip_l.safetensors` | 4.89 · 0.25GB | FLUX 조건 | comfyanonymous/flux_text_encoders |
| vae | `ae.safetensors` · `z_image_ae.safetensors`(같은 파일) | 0.34GB × 2 | 03 · 05 | Comfy-Org 재포장 |
| diffusion_models | `qwen-image-Q4_K_M.gguf` | 13.07GB | 04 · 05 | city96/Qwen-Image-gguf · Apache 2.0 |
| text_encoders | `Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf` | 5.44GB | Qwen-Image 조건 | unsloth/Qwen2.5-VL-7B-Instruct-GGUF |
| vae | `qwen_image_vae.safetensors` | 0.25GB | 03 · 05 | Comfy-Org/Qwen-Image_ComfyUI |
| diffusion_models | `z_image_bf16.safetensors` | 12.31GB | 04 · 05 | Comfy-Org/z_image · Apache 2.0 |
| text_encoders | `qwen_3_4b.safetensors` | 8.04GB | Z-Image 조건 | Comfy-Org/z_image |
| vae | `sdxl_vae.safetensors` | 0.33GB | 03 셀 8 | stabilityai/sdxl-vae |
| text_encoders | `qwen3.5_4b_bf16.safetensors` · `qwen3.5_2b_bf16.safetensors` | 9.32 · 4.55GB | 05 셀 11 만(`--core` 면 생략) | Comfy-Org/Qwen3.5 |

- 크기는 2026-10-06 Hugging Face 응답(`X-Linked-Size`)과 강사 PC 파일이 바이트까지 같음을 확인했다

## 01 · 디퓨전 원리 (SD1.5, RTX 4080 · float16)

| 셀 | 실측 |
|---|---|
| 2 | 부품 파라미터 : 글 인코더 123.1M(11.5%) · U-Net 859.5M(80.6%) · VAE 83.7M(7.8%) · 합계 1,066.2M. 스케줄러 웨이트 0, 학습 노이즈 단계 1,000 |
| 3 | 토큰 (1, 77) → 조건 c (1, 77, 768) / z_T (1, 4, 64, 64) / U-Net 출력 = 입력 모양 / 20단계(U-Net 20×2회) 1.15초(저장 출력 `_out` 기준. 앞선 실행은 0.95초 — 시간은 실행마다 다름) / 디코더 (1, 3, 512, 512) = latent 의 48배 |
| 4 | a_t · b_t : t 0 → 1.000 · 0.029, t 300 → 0.768 · 0.640, t 500 → 0.526 · 0.851, t 999 → 0.068 · 0.998 (beta 0.00085 → 0.012, scaled_linear) |
| 5 | ε 손실 t 20 → 0.7344, t 550 → 0.0657, t 980 → 0.0034 / 원본 추정 손실 t 20 → 0.0140, t 980 → 0.5757 — 같은 목표인데 t 마다 손실 크기가 반대 방향으로 다름 |
| 6 | 최종 그림과 상관 — 구도(저주파) 1단계 0.813 · 5단계 0.895 · 8단계 0.986, 디테일(고주파) 1단계 0.220 · 5단계 0.366 · 8단계 0.707 · 15단계 0.927 |
| 7 | 파이프라인 vs 손으로 쓴 루프 최대 차이 0 · DDIM step vs 두 줄 계산 최대 차이 0 |
| 8 | 같은 시드 재현 : 다섯 샘플러 모두 True. 50스텝 결과의 구도 vs DDIM — Euler 0.909 · DPM++ 2M 0.950(같은 그림으로 모임), DDPM 0.681 · Euler a 0.626(다른 그림). 10↔50스텝 구도 일치 DPM++ 2M 0.914 · Euler 0.868 · DDIM 0.669 |
| 9 | time_proj (1000, 320) → time_embedding (1000, 1280). 직접 계산한 사인 · 코사인과 최대 차이 0 |

- 처음 가설 중 틀린 것 : "DDIM 은 스텝 수를 바꿔도 구도가 유지된다" — 10↔50스텝 구도 일치가 DDIM 0.669 로 다섯 중 가장 낮았다. 장표에 넣지 않는다

## 02 · U-Net과 프롬프트 (SD1.5, RTX 4080)

| 셀 | 실측 |
|---|---|
| 3 | 해상도 64 → 32 → 16 → 8 → 8 → 16 → 32 → 64, 채널 320 → 640 → 1280. skip 12개, 이어 붙인 뒤 채널 예 : up 0 은 2560(1280 + 1280) |
| 4 | 64×64 skip 을 4×4 평균으로 흐림 — 남은 노이즈 양 : 그대로 24.3 · 마지막 5단계만 48.9 · 마지막 10단계만 84.8 · 전부 95.5. 그림이 흐려지는 것이 아니라 **노이즈가 안 지워짐** |
| 5 | 교차 어텐션 확률표 64×64 해상도 [4096 × 77] 315,392칸/헤드, 셀프 [4096 × 4096] 16,777,216칸/헤드. 교차 어텐션 층 16개. 직접 계산 처리기로 바꿔도 출력 최대 차이 0.0020(float16) |
| 6 | 확률 몫(그림 칸 평균) : <\|startoftext\|> 0.913 · red 0.003 · car 0.011 · beach 0.022. 지도는 red · car → 자동차, beach → 모래 · 하늘 |
| 7 | CLIP 일치도 토큰 77줄 vs 풀링 1개 복제 : red car 0.337 vs 0.218 · red cube on blue sphere 0.335 vs 0.232 · cat yellow hat 0.329 vs 0.185 · two dogs one cat 0.323 vs 0.163 (풀링 쪽은 질감만 남은 그림) |
| 8 | A(red car on the beach) → k 단계 뒤 B(blue boat on the sea) : k ≤ 5 면 배, k ≥ 8 이면 자동차(20단계 중) |
| 9 | \|ε_c − ε_u\| / \|ε_u\| = 0.006 ~ 0.018(조건이 바꾸는 몫은 예측의 0.6~1.8%). CLIP 일치도 w 1 → 0.202 · 3 → 0.316 · 7.5 → 0.337 · 15 → 0.343. GPU 에서 w 1(호출 1번)과 w > 1(2배치)의 시간 차이는 거의 없음(0.90~0.99초) |
| 10 | 768칸 순서만 섞음 → CLIP 0.159(그대로 0.337). 5-1 CLIP 텍스트 타워(폭 512) → `RuntimeError: mat1 and mat2 shapes cannot be multiplied (77x512 and 768x320)` |

- 처음 가설 중 틀린 것 : "skip 의 세부를 지우면 흐린 그림" — 실제로는 노이즈가 남는다(셀 4). 셀 4 의 설명을 실측에 맞춰 바꿨다
- 셀 7 주의 : U-Net 은 토큰 77줄로만 학습했으므로 풀링 쪽 저하에는 "정보가 뭉개짐"과 "학습 때 못 본 입력"이 겹친다(노트북에 명시)

## 03 · VAE와 latent

| 셀 | 실측 |
|---|---|
| 3 | (1, 3, 512, 512) 786,432 → (1, 4, 64, 64) 16,384 = 48배. 왕복 PSNR 38.1dB |
| 5 | 반려동물 40장 latent 표준편차 5.186(채널별 3.921~6.083) → × 0.18215 = 0.945. 1/표준편차 = 0.19282 |
| 6 | μ 절댓값 평균 4.046 · σ 평균 0.00027(최대 0.00161). KL 382,167 × 가중치 10⁻⁶ = 0.38. z = μ + σ·ε 직접 계산 = 라이브러리(차이 0). μ 디코드 vs 샘플 디코드 PSNR 73.6dB |
| 7 | latent 1칸 → 그림 57 × 32 픽셀 영역이 바뀜(대응 블록 8 × 8) |
| 8 | doc_3 (512 × 624) 왕복 PSNR : SD1.5 24.8 · SDXL 27.7(4채널) / FLUX 36.3 · Qwen-Image 35.7(16채널). ComfyUI 를 3090 에서 돌린 첫 실행은 SDXL 27.6 · Qwen-Image 35.8(GPU 가 다르면 소수 끝자리가 다름). latent 값 19,968 vs 79,872 |

- 근거 : KL 가중치 `kl_weight: 0.000001` 은 latent-diffusion 저장소 `configs/autoencoder/autoencoder_kl_32x32x4.yaml`

## 04 · DiT와 플로 매칭

| 셀 | 실측 |
|---|---|
| 3 | [16, 128, 128] → 4,096 × 64 → 되접기 차이 0 (5-1 ViT-B/32 : 49 × 3,072) |
| 4 | FLUX.1-dev 11.90B · 패치 64 → 3072 · 글 4096 → 3072 · 풀링 768 → 3072 · 블록 19 + 38 / Qwen-Image 20.43B · 64 → 3072 · 3584 → 3072 · 60 + 0 / Z-Image 6.15B · 64 → 3840 · 2560 → 3840 · 0 + 30. 출력층 모두 → 64, timestep 입력 256 |
| 5 | DiT 확률표 칸(글 100토큰 포함) 512² 1.3M · 1024² 17.6M · 1328² 48.8M |
| 6 | 데이터까지 거리 노이즈 예측 vs 플로 매칭 : 2스텝 0.858 vs 0.314 · 4스텝 0.218 vs 0.062 · 8스텝 0.059 vs 0.028 · 64스텝 0.018 vs 0.016 (1스텝은 둘 다 실패 0.809 vs 1.695) |
| 7 | 20스텝 중 σ > 0.5 인 스텝 : shift 1 → 10 · 3 → 15 · 3.1 → 16 · 6 → 18 |

- 처음 가설 중 틀린 것 : "플로 매칭은 생성 경로가 더 곧다" — 64스텝 곧음 0.65 로 노이즈 예측(0.77)보다 낮았다. 곧은 것은 학습 때 잇는 선이고 생성 경로는 휠 수 있다(노트북에 명시)
- 시프트 식 `s·σ / (1 + (s − 1)·σ)` 는 실행 중인 ComfyUI 0.39 의 `comfy/model_sampling.py` `time_snr_shift` 에서 확인

## 05 · ComfyUI 로 최신 모델 (RTX 4080 16GB, 전체 44분)

시간은 ComfyUI 실행 기록(시작~끝). 모델을 바꿔 올리는 경우 F: 드라이브에서 읽는 시간이 들어간다("로드 포함").

| 셀 | 실측 |
|---|---|
| 2 | ComfyUI 0.39.0 · 노드 종류 975개 |
| 3 | 노드 → 부품 대응표 · 워크플로 넷 저장(`workflows/*_t2i.json` 화면용 · `*_api.json`) |
| 4 | SD1.5 512² 20단계 21.6초(로드 포함) → 시드만 바꿔 1.9초 |
| 5 | "NEURON DEV" 헤드라인 : FLUX.1-dev 131.6초 "NEURON FIET" · Qwen-Image 1328² 295.9초 "NEURON DEV" · Z-Image 148.0초 "NEURON DE"(잘림) — 모두 로드 포함 |
| 6 | Z-Image 30단계 중 1 · 2 · 4 · 8 · 15 · 30 단계 완성본 추정 : 4단계에 구도, 8단계에 글자 |
| 7 | Qwen-Image cfg 1 · 2.5 · 5 는 비슷, 8 은 배경이 사라지고 단순. FLUX 가이던스 1 흐림 · 3.5 "NEURON FIET" · 7 "NEURON DEV", 가이던스 3.5 + cfg 3.5 는 어둡고 44초(27초의 약 1.6배) |
| 8 | FLUX-dev 4단계 6초(흐림) · 8단계 11초 · 20단계 28초, schnell 4단계 선명. Z-Image euler · res_multistep · euler_ancestral 같은 구도(ancestral 만 세부 다름) |
| 9 | Z-Image 시프트 1 · 3 · 6 같은 구도, 이 시드에서는 shift 1 만 헤드라인을 끝까지 씀 |
| 10 | 한국어 「한복 고양이 · "뉴런데브" 나무 간판」 : FLUX 프롬프트 무시 · Qwen-Image 간판 "누건대브" · Z-Image "뉴런데브" 정확 |
| 11 | Z-Image 조건 인코더 바꿔치기 : Qwen3-4B 정상 44초 · Qwen3.5-4B(폭 같음) 엉뚱한 그림 77초 · Qwen3.5-2B `Given normalized_shape=[2560] … got input of size[1, 46, 2048]` |
| 12 | Z-Image 내 프롬프트 115.0초 |

- 시드 하나의 결과다. 모델 간 우열이 아니라 경향으로 읽는다
- 결과 그림 : `results/`(노트북 출력 그림을 옮김 — 실습 장표에 끌어다 놓는 용도)
- 처음 가설 중 틀린 것 : "권장 시프트(3)가 가장 좋다" — 이 시드에서는 shift 1 만 헤드라인을 다 썼다(셀 9)

## 문제 해결

| 증상 | 원인 · 조치 |
|---|---|
| 셀 2 `AssertionError: GPU 가 보이지 않음` | 컨테이너에 GPU 가 안 넘어감. `docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L` 확인. 윈도는 드라이버 설치 뒤 재부팅 |
| ComfyUI 화면에서 `*_api.json` 이 빈 그래프 | 화면은 API 형식을 그래프로 열지 못함 → `*_t2i.json`(화면 형식)을 연다 |
| ComfyUI 로그 `Can't find mmproj file … Qwen-Image-Edit will be broken` | 편집 모델용 경고. 이 실습(글 → 그림)에는 영향 없음 |
| 05 첫 장이 몇 분 걸림 | 모델 파일 첫 읽기(바인드 마운트). 두 번째부터 빠름 |
| `CUDA out of memory` (GPU 한 장) | ComfyUI 가 모델을 쥐고 있음 → 05 의 `free()` 실행 또는 `docker restart ai4-comfy` |

## 강사용

```bash
# 노트북 : 셀 내용은 tools/build_notebook_0N.py 에서 고치고 컨테이너 안에서 다시 만든다
docker exec -w /work/5-2/tools ai4-gen python build_notebook_01.py
# 검증 실행(출력은 notebooks/_out/, .gitignore)
docker exec -w /work/5-2/notebooks ai4-gen jupyter nbconvert --to notebook --execute --output-dir _out 01_디퓨전_원리.ipynb
```

## ComfyUI 화면 캡처 (강의 덱에 넣었던 그림)

실제 실행 화면을 헤드리스 Chrome 으로 찍어 강의 장표에 넣었다(2026-10-06, RTX 4080). 2026-10-07 기준 강의 덱(46장)에서 확인되는 것은 「Diffusion 이미지 생성」 · 「sampler」 · 「CFG (4) — ComfyUI 화면에서」 · 「플로 매칭 — ComfyUI 화면」 이고, 표의 나머지 장 이름(미리보기 · 모델별 그래프 · 인코더 바꿔치기 오류 등)은 현재 덱에서 찾지 못했다(그림 파일은 보관).

| 파일 | 내용 | 강의 장 |
|---|---|---|
| `screens/c1_sd15_graph.png` | SD1.5 워크플로 실행 결과(23초) | ComfyUI 화면 — SD1.5 실행 결과 |
| `screens/c2_ksampler.png` | KSampler 확대(위젯 = 이론) | KSampler — 이론이 위젯 하나씩 |
| `screens/slide/S7_ksampler_three.png` | 같은 캡처에서 시드 · 샘플러 이름 · 스케줄러 줄만(`tools/screens/ksampler_three.py`) | 7장 Diffusion 이미지 생성 3번 「seed · sampler · scheduler」 |
| `screens/c3_sampler_list.png` · `c4_scheduler_list.png` | 샘플러 이름 40여 개 · 스케줄러 9개 | 샘플러 · 스케줄러 — 두 목록 |
| `screens/slide/S9_sampler_list.png` | 같은 캡처에서 「샘플러 이름」 칸 + 펼친 목록, 본문에 나오는 샘플러 10개 표시(`tools/screens/sampler_list.py`) | 「sampler」 장 |
| `screens/slide/N_cfg_comfy.png` | `c1_sd15_graph` · `c5_flux_graph` 에서 CFG 관련 칸만 잘라 ①~⑦ 표시 — SD1.5 긍정 · 빈 부정 인코딩, KSampler 두 조건 입력, cfg 7.5 / FLUX 가이드 3.5, cfg 1, 부정 노드(`tools/screens/cfg_view.py`) | 「CFG (4) — ComfyUI 화면에서」 |
| `screens/zseq_*.png` → `c9_zimage_preview_strip.png` | Z-Image 30단계 실행 중 1.5초마다 KSampler 미리보기 → 1 · 4 · 5 · 8 · 12 · 18 · 30단계 | 실행 중 미리보기 |
| `screens/c5_flux_graph.png` · `c6_qwen_graph.png` · `c7_zimage_done.png` | FLUX · Qwen-Image · Z-Image 그래프 | 각 모델 장 |
| `screens/slide/FLOW_nodes.png` | `c6_qwen_graph` 에서 Unet Loader · 모델 샘플링(시프트 · 샘플링 flow) · KSampler(스텝 · euler · simple)만 잘라 ①~⑥ 표시(`tools/screens/flow_nodes.py`) | 「플로 매칭 — ComfyUI 화면」 |
| `screens/c8_encoder_swap_error.png` · `c8b_encoder_swap_detail.png` | Z-Image 조건 인코더를 Qwen3.5-2B 로 바꾼 실행 실패 · 오류 로그 | 조건 인코더 바꿔치기 — 화면의 오류 |

- 단계 번호는 캡처 순간 KSampler 진행 막대의 길이에서 읽었다(`tools/screens/preview_strip.py`)
- ComfyUI 미리보기 = 매 단계 콜백이 받은 x0(완성본 추정)를 그린 것 — ComfyUI `latent_preview.py` 의 `prepare_callback`
- 다시 찍기 : `node tools/screens/capture.js` → `docker run --rm -v "<5-2/screens>:/s" -v "<5-2/tools/screens>:/w" ai4-lab/jupyter-clip python /w/annotate.py`
- 로컬 그림을 구글 슬라이드에 넣을 때 : Codex 의 `batch_update_presentation` 에 `image_uris` = 로컬 경로 **하나**(여러 개를 넣으면 실패) → 그림마다 한 번씩 호출
- 캡처는 두 번 돌았다 : 17:16 첫 실행(덱에 넣은 그림) · 17:42 재실행(README 를 고치던 셸 명령의 백틱이 `capture.js` 를 실행 — 의도하지 않음). 두 실행의 그래프 캡처로 만든 번호 표시 그림 7장은 픽셀 차이 0. 미리보기 띠만 프레임 시점이 달라(재실행엔 4단계 프레임이 없음) 첫 실행본을 유지했다
- `screens/slide/D_strip.png`(미리보기 띠를 덱용으로 번호 없이 이은 그림)는 저장소에 올리지 않는다(.gitignore) — 푸시 보안 훅이 이 이미지를 토큰으로 오인해 차단. `c9_zimage_preview_strip.png` 에서 `tools/screens/preview_strip.py` 로 다시 만든다
- `screens/zseq_*.png`(재실행 프레임 54장)는 저장소에 넣지 않는다(.gitignore)

## 강의 보강 실험 — mid 어텐션 끄기 (강의 덱 「mid 어텐션을 끄면 — 실측」, 2026-10-07)

`tools/experiments/mid2.py` — SD1.5 U-Net 의 mid 블록 어텐션만 건너뛰고(입력을 그대로 넘김) 같은 프롬프트 · 시드로 생성, RTX 4080

| 측정 | 값 |
|---|---|
| 조건 | 프롬프트 6 × 시드 2, DDIM 20단계, cfg 7.5 |
| 구도(16×16 축소 그림) 켬↔끔 상관 | 평균 0.811 · 최소 0.408 |
| CLIP 일치도(ViT-B/32) | 켬 0.316 · 끔 0.296 |

- 구성 확인 : down 3 · up 0 은 어텐션 0개(합성곱만), mid 는 resnet → 셀프 + 교차 어텐션 → resnet
- 그림 : `mid.py` · `mid_strip.py` → `screens/slide/M_mid_onoff.png`(차 · 고양이 · 등대, 시드 42)
- 읽는 법 : 끄면 배치가 바뀜. 차 예처럼 더 나빠지지 않는 경우도 있어 「품질 저하」가 아니라 「배치 결정」으로 서술

## 강의 보강 실험 — CFG w 와 CLIP 일치도 (강의 덱 「CFG (2) — w를 바꿔 생성하면」, 2026-10-07)

CLIP 일치도 = CLIP ViT-B/32 의 그림 벡터와 프롬프트 벡터의 코사인. 「0.20 · 0.34 가 무슨 뜻인가」에 답하려고 무관한 글과의 값을 잣대로 함께 잰다. RTX 4080.

- `tools/experiments/cfg_strip.py` → `screens/slide/N_cfg_w.png` : 실습 02 셀 9 와 같은 계산(「a red car on the beach」, 시드 42, DDIM 20, 빈 프롬프트 = ε_u)을 512px 로 다시 저장. 재현값 w 1 · 3 · 7.5 · 15 → 0.202 · 0.316 · 0.337 · 0.343(셀 9 출력과 같음)
- `tools/experiments/cfg_scale.py` : 프롬프트 6개 × 시드 1 · 2(그림 12장), w 별 평균

| w | 자기 프롬프트 | 다른 프롬프트 5개 평균 |
|---|---|---|
| 1 | 0.262 | 0.144 |
| 3 | 0.303 | 0.135 |
| 7.5 | 0.316 | 0.137 |
| 15 | 0.323 | 0.138 |

- 실행 : `docker exec ai4-gen python /work/5-2/tools/experiments/cfg_scale.py` (Git Bash 에서는 `MSYS_NO_PATHCONV=1` 앞에 붙임)
- 「다른 프롬프트 5개」에는 일부 겹치는 글(고양이 · 소파, 빨간 물체)이 섞여 있어 완전히 무관한 글보다 약간 높을 수 있음

## 강의 보강 실험 — DDPM step 수 (강의 덱 「sampler」 1번 감수, 2026-10-07)

`tools/experiments/ddpm_steps.py` — SD1.5, cfg 7.5, 시드 42, 프롬프트 3개(차 · 등대 · 고양이) 평균, RTX 4080. 그림 : `outputs/exp/ddpm_steps.png`(차, 왼쪽부터 DDPM 20 · 50 · 1000 · DDIM 20)

| sampler | step | CLIP 일치도 | 남은 노이즈 양 | 시간(1장) |
|---|---|---|---|---|
| DDPM | 20 | 0.327 | 17.3 | 1.5초 |
| DDPM | 50 | 0.325 | 18.4 | 2.4초 |
| DDPM | 1000 | 0.323 | 20.3 | 47.2초 |
| DDIM | 20 | 0.322 | 22.5 | 1.1초 |

- 「DDPM 은 step 을 줄이면 품질 급락」은 재현되지 않음 — diffusers DDPMScheduler 는 1,000단계 중 일부만 밟는 재간격 방식. 대신 step 수마다 구도가 바뀜(재주입 노이즈가 달라짐)
- 남은 노이즈 양 = 실습 02 셀 4 와 같은 계산(저주파를 뺀 나머지의 표준편차). DDPM 20 은 약간 부드러움
- 시간 1.5초는 첫 실행의 준비 시간이 섞인 값
- 1000 step 은 `steps_offset=0` 으로 실행(기본값 1 이면 t=1000 색인 오류)
