"""6-1 실습 노트북 02 생성기 (강사용).

    python build_notebook_02.py   → ../notebooks/02_Whisper_열어보기.ipynb (출력 없이)
"""
from nb import Book

b = Book("02_Whisper_열어보기.ipynb")

b.md('''
# 실습 02 · Whisper 열어 보기 — 인코더 · 디코더 · 교차 어텐션 · 환각

현재 강의와의 연결은 아래 셀별 표에 표시. 구조 확인·직접 측정·문헌 설명의 구분.
01 에서 만든 [128×3000] 이 인코더에서 [1500×1280] 이 되고, 디코더가 그것을 교차 어텐션으로 보며 글자를 하나씩 내는 과정을 따라간다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 모델 · 부품별 파라미터 수 · turbo 와 large-v3 의 차이 | 음성 인식 ② — 로그멜,합성곱,트랜스포머 |
| 3 | 층별 텐서 모양 — conv 두 개 · 3000 → 1500 · 인코더 32층 | 음성 인식 ② — 로그멜,합성곱,트랜스포머 · Whisper 구성도 — 입력 가공 · 인코더 · 디코더 |
| 4 | 디코더의 첫 토큰들 = 지시 — 언어 감지 확률 · 특수 토큰 | Whisper 구성도 — 입력 가공 · 인코더 · 디코더 |
| 5 | 생성 = 다음 토큰 반복 — 손으로 쓴 루프 = generate | Whisper 구성도 — 입력 가공 · 인코더 · 디코더 |
| 6 | 교차 어텐션 — 어느 구간이 어느 글자인가 · 단어 시각 | Whisper 구성도 — 입력 가공 · 인코더 · 디코더 |
| 7 | 30초 채우기의 비용 — 7초도 30초도 인코더 시간은 같다 | 음성 인식 ② — 로그멜,합성곱,트랜스포머 · Whisper 다음 — Qwen3-ASR의 구조와 학습 |
| 8 | 환각 — 무음 · 잡음 · 모터 소리에 「감사합니다」 | 현장에서 갈리는 것 — 용어 목록과 환각 · 음성 모달 한계 |
| 9 | turbo vs large-v3 — 디코더 4층 vs 32층의 속도와 정확도 | 음성 인식 ② — 로그멜,합성곱,트랜스포머 |
| 10 | 학습 한 걸음 — 정답을 한 칸 밀어 넣고 다음 글자 확률의 손실 하나 · 손 계산 = 라이브러리 · 인코더까지 오는 기울기 | Whisper 구성도 — 입력 가공 · 인코더 · 디코더 |

- 실행 : GPU 권장(학습 역전파 셀 포함 시 VRAM 16GB 권장). turbo 1.6GB · large-v3 3.1GB 를 처음 한 번 받음
- **필요 자원(2026-10-19 강사 PC 실측 · RTX 4080 · 컨테이너 RAM 상한 9GB)** : 최대 VRAM 약 14.0GB. 실습 기준 PC(VRAM 8GB · RAM 16GB)보다 많이 필요하다 — 이 실습을 그대로 실현하려면 이 정도 자원이 든다
''')

b.md('''
### 셀 1 · 설정
- `N_EVAL` : 셀 9 에서 비교할 문장 수
''')
b.code('''
# 셀 1 · 설정
IDX = 0
N_EVAL = 30
TURBO = "openai/whisper-large-v3-turbo"
LARGE = "openai/whisper-large-v3"
''')

b.md('''
### 셀 2 · 모델 · 부품별 파라미터 수
- Whisper = **인코더**(소리 → 벡터열) + **디코더**(벡터열을 보며 글자 생성). 1강 Transformer 의 인코더-디코더 그대로
- turbo = large-v3 에서 디코더만 32층 → 4층으로 줄인 것(인코더는 같은 크기)
''')
b.code('''
# 셀 2 · 모델 · 부품별 파라미터 수
import sys, time; sys.path.insert(0, ".")
from voice_common import *
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoConfig
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.float16 if DEV == "cuda" else torch.float32
df = fleurs_ko()
proc = WhisperProcessor.from_pretrained(TURBO)
wm = WhisperForConditionalGeneration.from_pretrained(TURBO, torch_dtype=DT, attn_implementation="eager").to(DEV).eval()   # eager : 셀 6 에서 어텐션 행렬을 꺼내려고
n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
print(f"turbo 전체 {n(wm):.0f}M = 인코더 {n(wm.model.encoder):.0f}M + 디코더 {n(wm.model.decoder):.0f}M")
for name in (TURBO, LARGE):
    c = AutoConfig.from_pretrained(name)
    print(f"{name.split('/')[-1]:24s} 멜 {c.num_mel_bins} · 폭 {c.d_model} · 인코더 {c.encoder_layers}층 · 디코더 {c.decoder_layers}층 · 어휘 {c.vocab_size:,}")
''')

b.md('''
### 셀 3 · 층별 텐서 모양
- 입력 [128 × 3000] → **conv1**(3칸 창, 한 칸씩) [1280 × 3000] → **conv2**(3칸 창, 두 칸씩 건너뜀) [1280 × 1500]
- 합성곱 = 이미지 생성 강의의 U-Net 에서 본 「이웃 칸 × 웨이트 → 한 칸」을 시간 한 축으로 한 것. 두 칸씩 건너뛰니 줄이 절반
- 1500줄 ÷ 30초 = **초당 50줄(20ms 에 하나)** → 트랜스포머 블록 32층을 지나도 모양 그대로
''')
b.code('''
# 셀 3 · 층별 텐서 모양
x = wav_of(df, IDX); ref = df.iloc[IDX]["transcription"]
feat = proc(x, sampling_rate=SR, return_tensors="pt").input_features.to(DEV, DT)
shapes = []
def hook(name):
    return lambda m, i, o: shapes.append((name, tuple((o[0] if isinstance(o, tuple) else o).shape)))
enc = wm.model.encoder
hs = [enc.conv1.register_forward_hook(hook("conv1 (k=3, stride 1)")), enc.conv2.register_forward_hook(hook("conv2 (k=3, stride 2)")),
      enc.layers[0].register_forward_hook(hook("블록 1")), enc.layers[-1].register_forward_hook(hook(f"블록 {len(enc.layers)}")),
      enc.layer_norm.register_forward_hook(hook("인코더 출력"))]
with torch.no_grad(): eo = enc(feat).last_hidden_state
for h in hs: h.remove()
print("입력", tuple(feat.shape))
for s in shapes: print(f"{s[0]:24s} {s[1]}")
print("conv1 웨이트", tuple(enc.conv1.weight.shape), "· conv2 웨이트", tuple(enc.conv2.weight.shape), "· 위치 임베딩", tuple(enc.embed_positions.weight.shape))
print(f"초당 줄 수 = {eo.shape[1]} / 30초 = {eo.shape[1] / 30:.0f} → 줄 하나 = {30 / eo.shape[1] * 1000:.0f}ms")
''')

b.md('''
### 셀 4 · 디코더의 첫 토큰들 = 지시
- 디코더는 글자를 내기 전에 **특수 토큰**을 먼저 받는다 : 시작 · 언어 · 할 일(받아쓰기/번역) · 시각 표시 여부
- 언어를 주지 않으면 「시작」 다음 자리에서 언어 토큰 100여 개 중 확률이 가장 큰 것을 바꿈 = 언어 감지
''')
b.code('''
# 셀 4 · 디코더의 첫 토큰들 = 지시
tok = proc.tokenizer
sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
with torch.no_grad():
    logits = wm(encoder_outputs=(eo,), decoder_input_ids=torch.tensor([[sot]], device=DEV)).logits[0, -1].float()
lang_ids = [i for t, i in tok.get_vocab().items() if t.startswith("<|") and t.endswith("|>") and 2 <= len(t) - 4 <= 3 and t[2:-2].isalpha() and t[2:-2].islower()]
p = torch.softmax(logits[lang_ids], -1)
top = p.topk(5)
print("언어 감지 상위 5 :", [(tok.convert_ids_to_tokens(lang_ids[i]), round(float(v), 4)) for v, i in zip(top.values, top.indices)])
prompt = proc.get_decoder_prompt_ids(language="korean", task="transcribe")
print("디코더 지시 토큰 :", ["<|startoftranscript|>"] + [tok.convert_ids_to_tokens(i) for _, i in prompt])
''')

b.md('''
### 셀 5 · 생성 = 다음 토큰 반복
- 디코더는 지금까지 낸 토큰을 보고 **다음 토큰의 확률 분포**를 낸다 → 가장 큰 것을 선택해 붙이고 반복(앞서 배운 디코딩)
- 손으로 쓴 루프의 결과가 `generate` 결과와 같은지 확인
- Whisper 어휘는 바이트 단위로 쪼갠 토큰 → 한글 한 글자가 토큰 둘 이상이 되기도 한다(걸음마다 글이 반 글자씩 늘어날 수 있음)
''')
b.code('''
# 셀 5 · 생성 = 다음 토큰 반복
ids = [sot] + [i for _, i in prompt]
steps = []
with torch.no_grad():
    for _ in range(120):
        lg = wm(encoder_outputs=(eo,), decoder_input_ids=torch.tensor([ids], device=DEV)).logits[0, -1].float()
        nxt = int(lg.argmax()); pr = float(torch.softmax(lg, -1)[nxt])
        ids.append(nxt); steps.append((tok.decode(ids[4:], skip_special_tokens=True), round(pr, 3)))
        if nxt == tok.eos_token_id: break
mine = tok.decode(ids, skip_special_tokens=True)
with torch.no_grad(): gen = wm.generate(feat, language="korean", task="transcribe")
lib = tok.decode(gen[0], skip_special_tokens=True)
print("손 루프 :", mine); print("generate:", lib); print("같은가 :", mine.strip() == lib.strip())
print("정답    :", ref, "· CER", round(cer(lib, ref), 3))
print("처음 8걸음 — (지금까지 낸 글, 이번 토큰의 확률) :")
for s_, p_ in steps[:8]: print(f"   {p_:.3f}  {s_}")
n_char = len(lib.replace(" ", "")); print(f"토큰 {len(ids) - 5}개로 글자 {n_char}개 → 한글은 바이트 단위로 쪼개져 한 글자에 토큰 여러 개가 들기도 함")
''')

b.md('''
### 셀 6 · 교차 어텐션 — 어느 구간이 어느 글자인가
- 디코더의 교차 어텐션 = 「지금 낼 토큰」이 인코더 1500줄 중 어디를 보는가(이미지 생성 강의의 교차 어텐션과 같은 계산 : 질의 = 글 쪽, 키 · 값 = 소리 쪽)
- Whisper 는 그 가운데 정렬을 잘 하는 머리 몇 개를 선택해 두었다(`alignment_heads`) → 단어마다 시작 · 끝 시각
- 누가 「구간 = 글자」를 가르쳐 준 적은 없다 — 받아쓰기 학습 중 생긴 것(장표 「Whisper 구성도 — 입력 가공 · 인코더 · 디코더」의 학습 설명)
''')
b.code('''
# 셀 6 · 교차 어텐션
heads = wm.generation_config.alignment_heads
with torch.no_grad():
    out = wm(encoder_outputs=(eo,), decoder_input_ids=torch.tensor([ids], device=DEV), output_attentions=True)
A = torch.stack([out.cross_attentions[l][0, h] for l, h in heads]).float().mean(0).cpu().numpy()   # [토큰 × 1500]
L = int(len(x) / SR * 50) + 5
fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(A[4:-1, :L], aspect="auto", origin="upper", extent=[0, L / 50, A.shape[0] - 5, 0])
ax.set_xlabel("초(인코더 줄 ÷ 50)"); ax.set_ylabel("생성한 토큰 순서"); ax.set_title(f"정렬 머리 {len(heads)}개 평균 교차 어텐션 — 대각선 = 글자가 시간 순서로 소리 구간을 따라감")
plt.tight_layout(); plt.savefig(RESULTS / "02_cross_attention.png", dpi=150); plt.show()
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model=wm, tokenizer=tok, feature_extractor=proc.feature_extractor, device=DEV, torch_dtype=DT)
w = asr(x, return_timestamps="word", generate_kwargs={"language": "korean"})
print([(c["text"], tuple(round(t, 2) for t in c["timestamp"])) for c in w["chunks"][:10]])
''')

b.md('''
### 셀 7 · 30초 채우기의 비용
- Whisper 는 언제나 30초짜리 입력을 받으므로 **인코더 계산량이 소리 길이와 상관없이 같다**
- 짧은 명령어(1~2초)를 많이 받는 곳에서는 낭비 — 스트리밍 ASR 이 따로 있는 이유(장표 Whisper 다음 — Qwen3-ASR의 구조와 학습)
''')
b.code('''
# 셀 7 · 30초 채우기의 비용
def enc_ms(sec):
    z = np.zeros(int(sec * SR), np.float32) if sec else x
    f = proc(z if sec else x, sampling_rate=SR, return_tensors="pt").input_features.to(DEV, DT)
    with torch.no_grad():
        enc(f); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10): enc(f)
        torch.cuda.synchronize()
    return (time.time() - t0) / 10 * 1000
for s in (1, 7, 15, 30):
    print(f"{s:2d}초 입력 → 인코더 {enc_ms(s):.1f}ms (입력 모양은 언제나 [128×3000])")
''')

b.md('''
### 셀 8 · 환각 — 말이 없는데 글을 낸다
- 무음 · 백색잡음 · 모터 소리(60Hz 와 고조파)를 넣어 본다
- Whisper 학습 데이터 = 인터넷 영상의 소리와 자막 수백만 시간. 영상 끝의 말 없는 구간에 「시청해 주셔서 감사합니다」 같은 자막이 붙어 있던 것을 그대로 배움 — **앞서 배운 대로: 출력 분포 = 학습 데이터의 통계**
- 대책 : 말소리가 있는 구간만 잘라 넣기(VAD — 음성 구간 검출). 여기서는 세기로 자르는 가장 단순한 방법
''')
b.code('''
# 셀 8 · 환각
t = np.arange(10 * SR) / SR; rng = np.random.default_rng(0)
tests = {"무음 10초": np.zeros(10 * SR, np.float32),
         "백색잡음 10초": (0.05 * rng.standard_normal(10 * SR)).astype(np.float32),
         "모터 소리 60Hz+고조파": (0.1 * sum(np.sin(2 * np.pi * 60 * k * t) / k for k in range(1, 8))).astype(np.float32)}
def tr(z):
    f = proc(z, sampling_rate=SR, return_tensors="pt").input_features.to(DEV, DT)
    with torch.no_grad(): return tok.decode(wm.generate(f, language="korean", task="transcribe")[0], skip_special_tokens=True)
for k, z in tests.items(): print(f"{k:22s} → {tr(z)!r}")
def has_speech(z, frame=400, hop=160, db=-35):          # 25ms 조각의 세기가 가장 큰 조각보다 35dB 이상 작으면 말 없음
    e = np.array([np.sqrt((z[i:i + frame] ** 2).mean() + 1e-12) for i in range(0, len(z) - frame, hop)])
    e = 20 * np.log10(e / (e.max() + 1e-12) + 1e-12)
    return (e > db).mean() > 0.2 and np.std(z) > 1e-3 and np.ptp(e) > 20
for k, z in list(tests.items()) + [("FLEURS 문장", x)]:
    print(f"{k:22s} 말소리 있음? {has_speech(z)}")
play(tests["모터 소리 60Hz+고조파"], label="모터 소리")
''')

b.md('''
### 셀 9 · turbo vs large-v3
- 같은 인코더 크기, 디코더만 4층 vs 32층. 글자 하나 낼 때마다 디코더를 통째로 한 번 돌리므로 **디코더 층 수가 속도를 좌우**
- 한국어 FLEURS `N_EVAL` 문장의 CER(글자 오류율)과 시간
''')
b.code('''
# 셀 9 · turbo vs large-v3
def run(model_name):
    gpu_free(); torch.cuda.reset_peak_memory_stats()
    m = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=DT).to(DEV).eval()
    errs = []; torch.cuda.synchronize(); t0 = time.time()
    for i in range(N_EVAL):
        f = proc(wav_of(df, i), sampling_rate=SR, return_tensors="pt").input_features.to(DEV, DT)
        with torch.no_grad(): h = tok.decode(m.generate(f, language="korean", task="transcribe")[0], skip_special_tokens=True)
        errs.append(cer(strip_punct(h), strip_punct(df.iloc[i]["transcription"])))
    torch.cuda.synchronize(); sec = time.time() - t0
    del m; return np.mean(errs), sec, gpu_peak_gb()
del asr, wm; gpu_free()
rows = [(name.split("/")[-1],) + tuple(run(name)) for name in (TURBO, LARGE)]
R = pd.DataFrame(rows, columns=["모델", "CER(구두점 제외)", f"{N_EVAL}문장 시간(초)", "최대 VRAM(GB)"]).round(3)
print(R.to_string(index=False)); R.to_csv(RESULTS / "02_turbo_vs_large.csv", index=False)
''')

b.md('''
### 셀 10 · 학습 한 걸음 — 손실이 하나뿐임을 확인
- 학습 = 음성-글 쌍 한 개로 **앞으로 계산 한 번 + 손실 하나 + 역전파**. 소리 쪽 벡터와 글 쪽 벡터를 서로 맞추는 항은 없다(CLIP 의 대조학습과 다름)
- **교사 강요(teacher forcing)** : 정답 글을 한 칸 밀어 디코더 입력으로 넣고, 자리마다 "다음 글자"를 맞히게 함 → 정답 길이가 L 이면 한 번에 L 개 자리의 손실이 나옴
  - 입력 `[SOT, ko, transcribe, notimestamps, 다, 리, …]` → 정답 `[ko, transcribe, notimestamps, 다, 리, …, EOT]`
- 손실 = 정답 글자의 확률에 −log 를 씌워 평균(교차 엔트로피). 어휘 51,866개 중 정답 하나의 확률만 봄
- 확인 세 가지 : ① 손으로 계산한 손실 = 라이브러리 손실 ② 교차 어텐션 표의 크기 [헤드 20 × L × 1500] ③ 역전파하면 **인코더 맨 앞(conv1)까지** 기울기가 도착
- 모델은 float32 로 따로 올린다(역전파용, 약 3GB)
''')
b.code('''
# 셀 10 · 학습 한 걸음 — 손실이 하나뿐임을 확인
import torch.nn.functional as F
gpu_free()
tm = WhisperForConditionalGeneration.from_pretrained(TURBO, torch_dtype=torch.float32, attn_implementation="eager").to(DEV).train()
tok.set_prefix_tokens(language="korean", task="transcribe", predict_timestamps=False)
ids = tok(ref).input_ids                                   # [SOT, ko, transcribe, notimestamps, 글자…, EOT]
dec_in, labels = torch.tensor([ids[:-1]], device=DEV), torch.tensor([ids[1:]], device=DEV)
f32 = feat.float()
out = tm(input_features=f32, decoder_input_ids=dec_in, output_attentions=True)
logits = out.logits                                        # [1, L, 51866] — 자리마다 다음 글자의 확률 분포
nll = -torch.log_softmax(logits[0], -1).gather(1, labels[0][:, None])[:, 0]
mine = float(nll.mean()); lib = float(tm(input_features=f32, labels=labels).loss)
print(f"정답 길이 L = {labels.shape[1]} · 인코더 출력 {tuple(out.encoder_last_hidden_state.shape)} · 디코더 출력 {tuple(logits.shape)}")
print(f"손실 : 손 계산 {mine:.6f} · 라이브러리 {lib:.6f} · 차이 {abs(mine - lib):.2e}")
print("글자 자리마다 −log p(정답) :", [(tok.decode([t]).strip() or "·", round(float(v), 2)) for t, v in zip(labels[0][3:11].tolist(), nll[3:11])])
print("교차 어텐션 표(층 1) :", tuple(out.cross_attentions[0].shape), "= [배치, 헤드, 글 L, 소리 1500]")
tm.zero_grad(); F.cross_entropy(logits[0], labels[0]).backward()
g = lambda p: float(p.grad.norm())
print("기울기 크기 — 인코더 conv1 %.3e · 인코더 마지막 블록 q_proj %.3e · 디코더 교차 어텐션 k_proj(소리 쪽) %.3e · 디코더 임베딩 %.3e" % (
    g(tm.model.encoder.conv1.weight), g(tm.model.encoder.layers[-1].self_attn.q_proj.weight),
    g(tm.model.decoder.layers[0].encoder_attn.k_proj.weight), g(tm.model.decoder.embed_tokens.weight)))
del tm, out, logits; gpu_free()
''')

b.save()
