"""6-1 실습 노트북 05 생성기 (강사용).

    python build_notebook_05.py   → ../notebooks/05_뉴럴_코덱.ipynb (출력 없이)
"""
from nb import Book

b = Book("05_뉴럴_코덱.ipynb")

b.md('''
# 실습 05 · 뉴럴 코덱 — 소리를 정수로, 정수를 다시 소리로

장표 「뉴럴 코덱 = 음성 토크나이저」 · 「코드북 수와 음질」 · 「오디오 토큰 심화」를 EnCodec(2022) 과 Mimi(2024) 로 확인한다.
RVQ 를 **손으로 계산해 라이브러리 코드와 같은지** 보고, 코드북을 몇 개 쓰느냐에 따라 말이 얼마나 살아남는지 잰다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 모델 · 받아쓰기 잣대(Qwen3-ASR) | – |
| 3 | EnCodec — 비트레이트별 코드 모양 · 초당 토큰 · 복원 후 CER | 뉴럴 코덱 2번 · 코드북 수와 음질 |
| 4 | RVQ 를 손으로 — 잔차가 단계마다 줄어드는가 · 라이브러리 코드와 일치 | 뉴럴 코덱 도해 |
| 5 | Mimi — 초당 12.5 × 코드북 8 · 코드북 수별 복원 CER | 코드북 수와 음질 · 오디오 토큰 심화 2번 |
| 6 | 10초 발화 = 토큰 몇 개 — 줄 세우면 LLM 이 감당할 길이인가 | 2026 코덱 LM |

- 실행 : GPU 권장. EnCodec(약 90MB) · Mimi(약 380MB) · Qwen3-ASR(4.7GB, 03 에서 받았으면 다시 받지 않음)
- 복원한 소리를 직접 들어 보는 것이 숫자보다 먼저다 — 셀마다 재생 막대
''')

b.md('''
### 셀 1 · 설정
''')
b.code('''
# 셀 1 · 설정
IDX = 0                         # 코덱에 넣을 FLEURS 문장
''')

b.md('''
### 셀 2 · 모델 · 받아쓰기 잣대
- 코덱 복원 품질을 「사람이 알아듣는가」로 측정하기 위해, 복원한 소리를 Qwen3-ASR 로 다시 받아 적어 CER 을 측정한다
- EnCodec · Mimi 는 24kHz 소리를 받는다 → 16kHz FLEURS 를 24kHz 로 다시 샘플링해 넣음
''')
b.code('''
# 셀 2 · 모델 · 받아쓰기 잣대
import sys, time; sys.path.insert(0, ".")
from voice_common import *
import torch
from transformers import EncodecModel, MimiModel
from qwen_asr import Qwen3ASRModel
df = fleurs_ko(); x16 = wav_of(df, IDX); ref = df.iloc[IDX]["transcription"]
x24 = librosa.resample(x16, orig_sr=SR, target_sr=24000)
enc = EncodecModel.from_pretrained("facebook/encodec_24khz").cuda().eval()
mimi = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
asr = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0")
def tr(y24): return asr.transcribe(audio=[(librosa.resample(y24, orig_sr=24000, target_sr=SR), SR)], language="Korean")[0].text
n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
print(f"EnCodec {n(enc):.1f}M · Mimi {n(mimi):.1f}M")
print("원본 :", ref); print("원본 받아쓰기 CER", round(cer(strip_punct(tr(x24)), strip_punct(ref)), 3))
X = torch.tensor(x24)[None, None].cuda()
''')

b.md('''
### 셀 3 · EnCodec — 비트레이트별
- 1초를 **75 프레임**으로 자르고(24,000 ÷ 320), 프레임마다 코드북 n 개의 번호(0~1023 = 10비트)를 낸다
- 비트레이트 = 75 × n × 10 비트 → 1.5kbps(n=2) ~ 24kbps(n=32). 원본 16비트 PCM 은 24,000 × 16 = 384kbps
- 코드 모양 [코드북 수 × 프레임 수] — 장표 「정수 코드열」 표의 그 행렬
''')
b.code('''
# 셀 3 · EnCodec — 비트레이트별
rows = []; outs = {}
for bw in (1.5, 3.0, 6.0, 12.0, 24.0):
    with torch.no_grad():
        e = enc.encode(X, bandwidth=bw); codes = e.audio_codes[0, 0]          # [코드북 수 × 프레임]
        y = enc.decode(e.audio_codes, e.audio_scales)[0][0, 0].float().cpu().numpy()[:len(x24)]
    outs[bw] = y; h = tr(y)
    rows.append((bw, codes.shape[0], codes.shape[1], round(codes.shape[1] / (len(x24) / 24000), 1), codes.shape[0] * round(codes.shape[1] / (len(x24) / 24000)), round(cer(strip_punct(h), strip_punct(ref)), 3)))
R3 = pd.DataFrame(rows, columns=["kbps", "코드북 수", "프레임 수", "초당 프레임", "초당 정수 개수", "복원 CER"])
print(R3.to_string(index=False)); R3.to_csv(RESULTS / "05_encodec.csv", index=False)
print("코드 앞부분(코드북 4 × 프레임 6) :\\n", codes[:4, :6].cpu().numpy())
for bw in (1.5, 6.0, 24.0): play(outs[bw], 24000, label=f"EnCodec {bw}kbps")
''')

b.md('''
### 셀 4 · RVQ 를 손으로
- 프레임 하나의 벡터(128차원)를 코드북 1 에서 **가장 가까운 대표 벡터**로 바꾸고, 남은 차이(잔차)를 코드북 2 에서 다시 가장 가까운 것으로 … 반복
- 측정기를 기준기로 1차 보정하고, 남은 오차를 더 정밀한 기준기로 다시 보정하는 것과 같은 구조 — 단계마다 남은 오차가 줄어든다
- 손 계산 번호가 라이브러리 번호와 **모두 같은지** 확인
''')
b.code('''
# 셀 4 · RVQ 를 손으로
NQ = 8
with torch.no_grad():
    z = enc.encoder(X)[0].T                                   # [프레임 × 128] 인코더 출력(양자화 전 연속 벡터)
    lib = enc.encode(X, bandwidth=6.0).audio_codes[0, 0]      # 6kbps = 코드북 8개
    r = z.clone(); mine = []; norms = [float(r.norm(dim=1).mean())]
    for q in range(NQ):
        cb = enc.quantizer.layers[q].codebook.embed           # [1024 × 128] 대표 벡터
        d = (r ** 2).sum(1, keepdim=True) - 2 * r @ cb.T + (cb ** 2).sum(1)[None]
        k = d.argmin(1); mine.append(k); r = r - cb[k]; norms.append(float(r.norm(dim=1).mean()))
mine = torch.stack(mine)
print("인코더 출력", tuple(z.shape), "· 코드북 하나", tuple(enc.quantizer.layers[0].codebook.embed.shape))
print("손 계산 번호 = 라이브러리 번호 :", f"{float((mine == lib).float().mean()) * 100:.1f}%")
print("남은 오차(평균 크기) :", [round(v, 2) for v in norms])
fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(range(NQ + 1), norms, "o-"); ax.set_xlabel("쓴 코드북 수"); ax.set_ylabel("남은 오차 크기")
ax.set_title("RVQ — 코드북을 하나 더 쓸 때마다 남은 오차가 준다"); plt.tight_layout(); plt.savefig(RESULTS / "05_rvq_residual.png", dpi=150); plt.show()
''')

b.md('''
### 셀 5 · Mimi — 초당 12.5 × 코드북 8
- 1초 = **12.5 프레임**(EnCodec 의 1/6), 프레임마다 코드북 8개(각 2048 = 11비트) → 초당 정수 100개 · 1.1kbps
- Mimi 의 1번 코드북은 자기지도 음성 모델(WavLM)의 표현을 따라 하도록 학습했다 = 「의미(semantic)」 코드북. 나머지 7개 = 음향 잔차
- 코드북 1 · 2 · 4 · 8 개만으로 복원 → 다시 받아 적기
''')
b.code('''
# 셀 5 · Mimi
rows = []; mouts = {}
for q in (1, 2, 4, 8):
    with torch.no_grad():
        c = mimi.encode(X, num_quantizers=q).audio_codes            # [1 × 코드북 × 프레임]
        y = mimi.decode(c).audio_values[0, 0].float().cpu().numpy()[:len(x24)]
    mouts[q] = y; h = tr(y)
    rows.append((q, tuple(c.shape[1:]), round(c.shape[2] / (len(x24) / 24000), 2), round(q * 11 * 12.5 / 1000, 2), round(cer(strip_punct(h), strip_punct(ref)), 3), h))
R5 = pd.DataFrame(rows, columns=["코드북 수", "코드 모양", "초당 프레임", "kbps", "복원 CER", "받아쓰기"])
print(R5.drop(columns="받아쓰기").to_string(index=False)); R5.to_csv(RESULTS / "05_mimi.csv", index=False)
for _, r in R5.iterrows(): print(f"코드북 {r['코드북 수']} : {r['받아쓰기']}")
for q in (1, 2, 4, 8): play(mouts[q], 24000, label=f"Mimi 코드북 {q}개")
''')

b.md('''
### 셀 6 · 10초 발화 = 토큰 몇 개
- 코덱 토큰을 LLM 이 하나씩 예측하려면 한 줄로 세워야 한다 → 코드북 수 × 프레임 수
- 텍스트 토큰(01 셀 3 : 한국어 초당 3.3개)과 비교. 그래서 코덱 LM 은 **1번 코드북만 LM 이, 나머지는 따로**(VALL-E 의 AR + NAR, Qwen3-TTS · Qwen3-Omni 의 0번 코드북 + MTP)
''')
b.code('''
# 셀 6 · 10초 발화 = 토큰 몇 개
rows = [("EnCodec 6kbps(VALL-E 가 쓴 설정)", 75, 8), ("Mimi(Moshi)", 12.5, 8), ("Qwen3-TTS 토크나이저 12Hz", 12.5, 16)]
T = pd.DataFrame([(a, f, q, int(10 * f * q), int(10 * f)) for a, f, q in rows], columns=["코덱", "초당 프레임", "코드북", "10초를 한 줄로", "10초 프레임 수(LM 이 한 칸씩 갈 때)"])
print(T.to_string(index=False)); print("비교 : 10초 한국어 텍스트 ≈ 33 토큰(01 셀 3 : FLEURS 382문장 초당 3.3)")
del enc, mimi, asr; gpu_free()
''')

b.save()
