"""6-1 실습 노트북 01 생성기 (강사용). 셀 내용을 여기서 고치고 다시 만든다.

    python build_notebook_01.py   → ../notebooks/01_소리를_숫자로.ipynb (출력 없이)
"""
from nb import Book

b = Book("01_소리를_숫자로.ipynb")

b.md('''
# 실습 01 · 소리를 숫자로 — 파형 · STFT · 멜 · 로그, 그리고 「초당 몇 개」

장표 「음성 모달 문제」 · 「초당 몇 개」 · 「로그-멜 입력」을 숫자로 확인한다. 모델 없이 계산만 한다.
Whisper 가 받는 입력 [128×3000] 을 **손으로 계산해 라이브러리 값과 같은지** 본다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 데이터 받기 — FLEURS 한국어 382문장 | – |
| 3 | 파형 = 1초에 숫자 16,000개 · 같은 1초의 텍스트 토큰 수 | 음성 모달 문제 1번 |
| 4 | STFT 를 손으로 — 25ms 창 · 10ms 간격 → 초당 100줄 | 로그-멜 입력 1 · 2번 |
| 5 | 멜 필터 128개 — 낮은 소리는 촘촘히, 높은 소리는 성기게 | 로그-멜 입력 3번 |
| 6 | 로그 · 정규화 · 30초 채우기 → [128×3000] = Whisper 입력과 최대 차이 | 로그-멜 입력 4번 · Whisper 구조 1번 |
| 7 | 초당 몇 개 — 파형에서 토큰까지 | 초당 몇 개 |
| 8 | 한 파형에 섞인 것 — 같은 문장 · 다른 화자 · 잡음 | 음성 모달 문제 3번 |

- 실행 : **Run → Run All Cells**. GPU 없어도 됨. 처음 한 번 데이터 291MB 를 받음(hfcache 볼륨)
- 소리는 셀마다 재생 막대로 들어 볼 수 있다
''')

b.md('''
### 셀 1 · 설정
- `IDX` : 셀 3~6 에서 쓸 문장 번호(0~381)
''')
b.code('''
# 셀 1 · 설정
IDX = 0
''')

b.md('''
### 셀 2 · 데이터 받기
- FLEURS(Google) 한국어 test : 문장 270개를 화자 여럿이 읽은 382줄. 라이선스 CC BY 4.0
- 열 : `transcription`(정답 글) · `gender`(남/여) · `sec`(길이)
''')
b.code('''
# 셀 2 · 데이터 받기
import sys; sys.path.insert(0, ".")
from voice_common import *
df = fleurs_ko()
print(len(df), "줄 · 문장", df["id"].nunique(), "개 · 평균", round(df["sec"].mean(), 1), "초")
df[["id", "gender", "sec", "transcription"]].head()
''')

b.md('''
### 셀 3 · 파형 = 1초에 숫자 16,000개
- 마이크는 공기 압력을 1초에 16,000번 측정해서 숫자로 적는다(샘플링레이트 16kHz). 숫자 하나 = 그 순간의 진폭
- 16kHz 로 측정하면 8kHz 까지의 소리만 담긴다(측정 간격의 절반 — 계측의 샘플링 정리와 같음). 사람 말소리는 대부분 그 안
- 같은 문장의 **텍스트 토큰** 수와 비교 : Qwen3 토크나이저(오디오 LLM · Qwen3-ASR 의 LLM 이 쓰는 것)
''')
b.code('''
# 셀 3 · 파형 = 1초에 숫자 16,000개
from transformers import AutoTokenizer
x = wav_of(df, IDX); ref = df.iloc[IDX]["transcription"]
print("샘플링레이트", SR, "· 길이", len(x), "개 =", round(len(x) / SR, 2), "초")
print("1.000초부터 8개 숫자 :", np.round(x[SR:SR + 8], 4), "(앞부분은 말 없는 구간이라 0 근처)")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
n_tok = len(tok(ref).input_ids)
print(f"정답 글 : {ref}")
print(f"텍스트 토큰 {n_tok}개 → 초당 {n_tok / (len(x) / SR):.1f}개 · 파형은 초당 {SR:,}개 → {SR / (n_tok / (len(x) / SR)):,.0f}배")

# 382문장 전체 : 초당 텍스트 토큰
tps = np.array([len(tok(t).input_ids) / s for t, s in zip(df["transcription"], df["sec"])])
print(f"382문장 초당 텍스트 토큰 : 평균 {tps.mean():.1f} · 중앙 {np.median(tps):.1f}")

fig, ax = plt.subplots(2, 1, figsize=(11, 4))
t = np.arange(len(x)) / SR
ax[0].plot(t, x, lw=0.4); ax[0].set_title("파형 전체"); ax[0].set_xlabel("초")
s0 = int(1.0 * SR); ax[1].plot(t[s0:s0 + 400] * 1000, x[s0:s0 + 400], ".-", lw=0.6, ms=2)
ax[1].set_title("1.000초부터 25ms 확대 — 점 하나 = 숫자 하나(400개)"); ax[1].set_xlabel("밀리초")
plt.tight_layout(); plt.show()
play(x)
''')

b.md('''
### 셀 4 · STFT 를 손으로
- 파형을 **25ms(400개) 창**으로 자르고 **10ms(160개)씩 밀며** 겹쳐 자른다 → 초당 100조각
- 조각마다 푸리에 변환(FFT) → 주파수별 세기. 설비 진동 분석에서 진동 센서 신호를 FFT 로 보는 것과 같은 계산을, **짧은 구간마다 반복**한 것(Short-Time)
- 결과 = 스펙트로그램 [주파수 201칸 × 시간 조각]. 가로 = 시간, 세로 = 주파수, 밝기 = 세기
- 손으로 계산한 값과 `torch.stft` 값이 같은지 최대 차이로 확인
''')
b.code('''
# 셀 4 · STFT 를 손으로
import torch
N_FFT, HOP = 400, 160
win = np.hanning(N_FFT + 1)[:-1].astype(np.float32)               # 창 끝을 부드럽게(주기형 hann)
xp = np.pad(x, N_FFT // 2, mode="reflect")                         # 앞뒤를 반 창만큼 채워 첫 조각이 0초에 중심
frames = np.stack([xp[i:i + N_FFT] * win for i in range(0, len(xp) - N_FFT + 1, HOP)])   # [조각 수 × 400]
spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2                    # [조각 수 × 201] 세기(제곱)
ref_spec = torch.stft(torch.tensor(x), N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).abs() ** 2
print("조각", frames.shape, "→ 스펙트로그램", spec.T.shape, "(주파수 × 시간)")
print("초당 조각 수 =", SR // HOP, "· 이 문장", spec.shape[0], "조각")
print("손 계산 vs torch.stft 최대 상대 차이 :", float(np.abs(spec.T - ref_spec.numpy()).max() / ref_spec.max()))
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.imshow(np.log10(spec.T + 1e-10), origin="lower", aspect="auto", extent=[0, len(x) / SR, 0, SR / 2])
ax.set_xlabel("초"); ax.set_ylabel("Hz"); ax.set_title("스펙트로그램(로그 세기) — 주파수 201칸 · 0~8,000Hz 를 고르게"); plt.show()
''')

b.md('''
### 셀 5 · 멜 필터 128개
- 사람 귀는 낮은 소리의 차이는 잘 듣고 높은 소리의 차이는 둔하다 → 주파수 축을 **낮은 쪽은 촘촘히, 높은 쪽은 성기게** 묶는다(멜 척도)
- 묶는 방법 = 삼각형 필터 128개. 201칸 세기에 필터를 곱해 더함 → 128칸. 행렬 곱 한 번 : [128×201] @ [201×시간]
- Whisper large-v3 · turbo 는 128개(v2 까지 80개)
''')
b.code('''
# 셀 5 · 멜 필터 128개
N_MELS = 128
fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)        # [128 × 201] — Whisper 가 쓰는 것과 같은 함수
mel = fb @ spec.T                                                   # [128 × 시간]
print("필터 묶음", fb.shape, "@ 스펙트로그램", spec.T.shape, "= 멜", mel.shape)
hz = np.linspace(0, SR / 2, fb.shape[1])
ctr = librosa.mel_frequencies(N_MELS + 2, fmin=0, fmax=SR / 2)[1:-1]      # 필터 128개의 가운데 주파수
print(f"필터 가운데 간격 : 1~2번 {ctr[1] - ctr[0]:.0f}Hz · 64~65번 {ctr[64] - ctr[63]:.0f}Hz · 127~128번 {ctr[-1] - ctr[-2]:.0f}Hz")
print(f"1,000Hz 아래에 필터 {int((ctr < 1000).sum())}개 · 4,000Hz 위에 {int((ctr > 4000).sum())}개")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
for r in fb[::8]: ax[0].plot(hz, r)
ax[0].set_title("멜 필터(8개마다 하나만 그림)"); ax[0].set_xlabel("Hz")
ax[1].imshow(np.log10(mel + 1e-10), origin="lower", aspect="auto", extent=[0, len(x) / SR, 0, N_MELS])
ax[1].set_title("멜 스펙트로그램 [128 × 시간]"); ax[1].set_xlabel("초"); plt.tight_layout(); plt.show()
''')

b.md('''
### 셀 6 · 로그 · 정규화 · 30초 채우기 → Whisper 입력
- 로그 : 소리 세기는 범위가 수십억 배 → log10 으로 눌러 작은 소리와 큰 소리를 함께 다룸
- Whisper 의 정규화 두 줄 : 가장 큰 값에서 8 이상 아래는 잘라 냄(log10 으로 1억 배 아래 = 무시) · (값 + 4) ÷ 4
- **30초 채우기** : Whisper 는 언제나 30초 = 480,000개를 받는다. 짧으면 0 으로 채움 → 3,000줄
- 손 계산과 `WhisperFeatureExtractor` 값의 최대 차이
''')
b.code('''
# 셀 6 · 로그 · 정규화 · 30초 채우기 → Whisper 입력
from transformers import WhisperFeatureExtractor
def whisper_logmel(x):
    x30 = np.zeros(30 * SR, np.float32); x30[:min(len(x), 30 * SR)] = x[:30 * SR]
    s = torch.stft(torch.tensor(x30), N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).abs() ** 2
    m = torch.tensor(fb) @ s[:, :-1]                                # 마지막 조각 하나는 버림 → 정확히 3,000
    lg = torch.clamp(m, min=1e-10).log10()
    lg = torch.maximum(lg, lg.max() - 8.0)
    return ((lg + 4.0) / 4.0).numpy()
mine = whisper_logmel(x)
lib = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")(x, sampling_rate=SR, return_tensors="np").input_features[0]
print("손 계산", mine.shape, "· 라이브러리", lib.shape, "· 최대 차이", float(np.abs(mine - lib).max()))
fig, ax = plt.subplots(figsize=(11, 3))
ax.imshow(mine, origin="lower", aspect="auto", extent=[0, 30, 0, 128]); ax.axvline(len(x) / SR, color="w", ls="--")
ax.set_title(f"Whisper 입력 [128 × 3000] — 흰 선 뒤({len(x) / SR:.1f}초~30초)는 0 으로 채운 부분"); ax.set_xlabel("초"); plt.show()
''')

b.md('''
### 셀 7 · 초당 몇 개 — 파형에서 토큰까지
- 강의 전체의 자 : 1초를 **몇 개의 숫자 · 벡터 · 토큰**으로 나타내는가
- 「측정」은 이 노트북 또는 표에 적힌 노트북에서 직접 측정한 값, 「설계값」은 모델 설정(config)에서 읽는 값
''')
b.code('''
# 셀 7 · 초당 몇 개
rows = [("파형 샘플", SR, "측정(셀 3)"),
        ("STFT · 멜 조각", SR // HOP, "측정(셀 4)"),
        ("Whisper 인코더 출력", 50, "측정 : 02 셀 3 (3000 → 1500줄 / 30초)"),
        ("Qwen2.5-Omni 오디오 토큰", 25, "측정 : 04 셀 3"),
        ("Qwen3-ASR · Mimi 코덱 한 줄", 12.5, "측정 : 03 셀 7 · 05 셀 5"),
        ("Qwen3.5-Omni 오디오 토큰", 6.25, "보고서 2604.15804(가중치 비공개)"),
        ("텍스트 토큰(한국어 382문장 평균)", round(float(tps.mean()), 1), "측정(셀 3)")]
T = pd.DataFrame(rows, columns=["단계", "초당 개수", "출처"])
T["텍스트 대비"] = (T["초당 개수"] / tps.mean()).round(1)
print(T.to_string(index=False))
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.barh(T["단계"][::-1], T["초당 개수"][::-1]); ax.set_xscale("log"); ax.set_xlabel("초당 개수(로그 눈금)")
for i, v in enumerate(T["초당 개수"][::-1]): ax.text(v * 1.1, i, f"{v:,}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(RESULTS / "01_per_second.png", dpi=150); plt.show()
''')

b.md('''
### 셀 8 · 한 파형에 섞인 것
- 같은 문장을 **다른 화자**가 읽은 쌍(FLEURS 에는 같은 문장을 여럿이 읽은 줄이 있다) → 멜 그림의 무늬가 다르다
- 음높이(기본 주파수) 곡선 : 화자마다 높이가 다르다 — 내용은 같은데 숫자는 다름
- 같은 소리에 **잡음**을 섞으면(신호 대 잡음 5dB) 멜 그림 전체가 흐려짐 — 환경도 같은 숫자에 섞여 들어감
- 모델은 이 섞인 숫자에서 「내용」만 골라 읽어야 한다(ASR) · 반대로 생성은 이것을 전부 만들어 내야 한다(TTS)
''')
b.code('''
# 셀 8 · 한 파형에 섞인 것
pairs = [g.index.tolist()[:2] for _, g in df.groupby("id") if len(g) >= 2 and g["gender"].nunique() == 2]
a, c = pairs[0]
xa, xc = wav_of(df, a), wav_of(df, c)
print("같은 문장 :", df.iloc[a]["transcription"])
print(f"화자 A({df.iloc[a]['gender']}) {len(xa) / SR:.1f}초 · 화자 B({df.iloc[c]['gender']}) {len(xc) / SR:.1f}초")
rng = np.random.default_rng(0); noise = rng.standard_normal(len(xa)).astype(np.float32)
noise *= np.sqrt((xa ** 2).mean() / (noise ** 2).mean() / 10 ** (5 / 10)); xn = xa + noise
def logmel(z): return np.log10(fb @ (torch.stft(torch.tensor(z), N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).abs() ** 2).numpy() + 1e-10)
def f0(z):
    f, v, _ = librosa.pyin(z, fmin=60, fmax=400, sr=SR, frame_length=1024); return np.nanmedian(f)
print(f"음높이 중앙값 : A {f0(xa):.0f}Hz · B {f0(xc):.0f}Hz")
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
for k, (z, ttl) in enumerate([(xa, f"화자 A({df.iloc[a]['gender']})"), (xc, f"화자 B({df.iloc[c]['gender']}) — 같은 문장"), (xn, "화자 A + 잡음 5dB")]):
    M = logmel(z); ax[k].imshow(M, origin="lower", aspect="auto", vmin=-8, vmax=M.max()); ax[k].set_title(ttl); ax[k].set_xlabel("조각(10ms)")
plt.tight_layout(); plt.savefig(RESULTS / "01_mixed.png", dpi=150); plt.show()
play(xa, label="화자 A"); play(xc, label="화자 B"); play(xn, label="화자 A + 잡음")
''')

b.save()
