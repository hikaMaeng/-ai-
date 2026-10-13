"""6-1 실습 노트북 06 생성기 (강사용).

    python build_notebook_06.py   → ../notebooks/06_TTS_목소리_만들기.ipynb (출력 없이)
"""
from nb import Book

b = Book("06_TTS_목소리_만들기.ipynb")

b.md('''
# 실습 06 · TTS — 코덱 LM 으로 목소리 복제, flow matching 으로 소리 만들기, 그리고 측정하는 법

현재 강의와의 연결은 아래 셀별 표에 표시. 구조 확인·직접 측정·문헌 설명의 구분.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 — 읽힐 문장 · 참고 화자 | – |
| 2 | Qwen3-TTS 부품 — LM · 코드 예측기(MTP) · 12Hz 토크나이저 | 2026 방식 — 걸음을 줄이고 프레임 안에서 분업 |
| 3 | 목소리 복제 — 참고 음성 몇 초 + 글 → 새 소리 | 코덱 토큰 TTS — 소리를 이어 쓰는 LLM · 참고 음성 — 목소리는 어떻게 복제되나 |
| 4 | 받아쓰기로 측정 — 재전사 CER · 틀린 곳 | TTS를 측정하는 법 — 내용 · 목소리 · 귀 |
| 5 | 화자 유사도 — 같은 사람처럼 들리나(WavLM 화자 벡터) | 오디오 LLM 텐서 흐름 — 소리가 들어가는 자리 · TTS를 측정하는 법 — 내용 · 목소리 · 귀 |
| 6 | 참고 음성 길이 — 1초 · 3초 · 전체 | 참고 음성 — 목소리는 어떻게 복제되나 · TTS를 측정하는 법 — 내용 · 목소리 · 귀 |
| 7 | 하이브리드 — Qwen2.5-Omni 의 Talker 토큰 → DiT flow matching → 소리, 시각 점 수 · CFG | 다른 길 — 번호 없이 만드는 Flow Matching · 하이브리드 — 뼈대는 LM이, 소리 결은 flow가 |
| 8 | 워터마크 — 사람 귀에 안 들리는 표시를 넣고 찾기 | 음성 모달 한계 |
| 9 | 내 목소리로 — 녹음 파일을 넣으면 | 참고 음성 — 목소리는 어떻게 복제되나 |

- 실행 : GPU 필수(VRAM 12GB 이상, 셀 7). Qwen3-TTS 1.7B · Qwen3-ASR · WavLM 화자 모델 · Qwen2.5-Omni-3B · AudioSeal
- **목소리 복제는 본인 또는 동의한 사람의 목소리로만.** 기본 예시는 공개 데이터(FLEURS, CC BY 4.0) 화자
''')

b.md('''
### 셀 1 · 설정
''')
b.code('''
# 셀 1 · 설정
TEXTS = ["반응기 3번의 출구 온도가 설정값보다 12도 높습니다. 냉각수 밸브 개도를 확인하세요.",
         "질소 퍼지가 끝나기 전에는 맨홀을 열지 마십시오.",
         "작업 허가서에 화기 작업 시간과 가스 측정 결과를 적으세요."]
REFS = [1, 3, 7]                 # 목소리를 빌릴 FLEURS 문장 번호(화자 셋)
TERMS = "반응기, 출구 온도, 설정값, 냉각수 밸브 개도, 질소 퍼지, 맨홀, 작업 허가서, 화기 작업, 가스 측정"
''')

b.md('''
### 셀 2 · Qwen3-TTS 부품
- **토크나이저(12Hz)** : 소리 ↔ 정수. 초당 12.5 프레임 × 코드북 16개(05 의 Mimi 와 같은 계열)
- **LM(talker)** : 글 + 참고 음성 코드를 보고 프레임마다 **0번 코드북**을 다음 토큰처럼 예측(앞서 배운 디코딩)
- **코드 예측기(MTP, multi-token prediction)** : 그 프레임의 나머지 15개 코드북을 작은 모델이 채움 → VALL-E 의 「AR 1개 + NAR 나머지」 분업의 2026년 판
''')
b.code('''
# 셀 2 · Qwen3-TTS 부품
import sys, time; sys.path.insert(0, ".")
from voice_common import *
import torch
from qwen_tts import Qwen3TTSModel
df = fleurs_ko()
tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
core = tts.model
print("최상위 부품 :", {k: f"{n(v):,.0f}M" for k, v in core.named_children()})
for k, v in core.named_children():
    sub = {kk: f"{n(vv):,.0f}M" for kk, vv in v.named_children() if n(vv) > 1}
    if sub: print(f"  {k} 안 :", sub)
def find_keys(obj, keys, path="", out=None, depth=0):      # 설정 객체 안을 뒤져 이름에 keys 가 든 값만
    out = {} if out is None else out
    if depth > 4 or obj is None: return out
    for k, v in (vars(obj).items() if hasattr(obj, "__dict__") else []):
        if k.startswith("_"): continue
        if any(x in k for x in keys) and isinstance(v, (int, float, str, tuple, list)): out[path + k] = v
        elif hasattr(v, "__dict__") and "config" in k.lower(): find_keys(v, keys, path + k + ".", out, depth + 1)
    return out
st = core.speech_tokenizer
print("토크나이저 :", type(st).__name__, find_keys(getattr(st, "config", None) or getattr(getattr(st, "model", None), "config", None), ("quantizer", "codebook_size", "sample_rate", "upsample", "rate")))
print("LM 쪽 :", find_keys(core.config, ("num_code_groups", "hidden_size", "num_hidden_layers")))
del core, st                                   # 참조를 남기면 del tts 를 해도 GPU 메모리가 풀리지 않음
''')

b.md('''
### 셀 3 · 목소리 복제
- 입력 : 새로 읽힐 **글** + 참고 음성(몇 초) + 참고 음성의 **받아쓴 글**
- 참고 음성은 같은 토크나이저로 정수가 되어 LM 앞에 놓인다(장표 「참고 음성」 2번 acoustic prompt)
- 생성 시간 ÷ 소리 길이 = RTF(1보다 작으면 실시간보다 빠름)
''')
b.code('''
# 셀 3 · 목소리 복제
gen = {}
for r in REFS:
    rx, rt = wav_of(df, r), df.iloc[r]["transcription"]
    torch.cuda.synchronize(); t0 = time.time()
    wavs, sr = tts.generate_voice_clone(text=TEXTS, language=["Korean"] * len(TEXTS), ref_audio=[(rx, SR)] * len(TEXTS), ref_text=[rt] * len(TEXTS))
    torch.cuda.synchronize(); dt = time.time() - t0
    dur = sum(len(w) for w in wavs) / sr
    gen[r] = [librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=SR) for w in wavs]
    print(f"화자 {r}({df.iloc[r]['gender']}) : 소리 {dur:.1f}초 · 시간 {dt:.1f}초 · RTF {dt / dur:.2f}")
    for k, w in enumerate(gen[r]): sf.write(DATA / f"06_clone_ref{r}_{k}.wav", w, SR)
play(wav_of(df, REFS[0]), label="참고 음성(FLEURS)"); play(gen[REFS[0]][0], label="복제 : 문장 1")
''')

b.md('''
### 셀 4 · 받아쓰기로 측정
- TTS 의 「내용이 맞나」를 사람 대신 ASR 로 측정한다 : 만든 소리를 Qwen3-ASR 로 받아 적어 원래 글과 CER
- 주의 : 이 숫자에는 **ASR 의 실수도 섞인다** → 용어 목록(03 셀 6)을 주고 다시 측정해서 둘을 나눠 본다
''')
b.code('''
# 셀 4 · 받아쓰기로 측정
from qwen_asr import Qwen3ASRModel
del tts; gpu_free()
asr = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0")
rows = []
for r in REFS:
    h0 = [o.text for o in asr.transcribe(audio=[(w, SR) for w in gen[r]], language="Korean")]
    h1 = [o.text for o in asr.transcribe(audio=[(w, SR) for w in gen[r]], language="Korean", context=TERMS)]
    for k, (a, c) in enumerate(zip(h0, h1)):
        rows.append((r, k + 1, round(cer(strip_punct(a), strip_punct(TEXTS[k])), 3), round(cer(strip_punct(c), strip_punct(TEXTS[k])), 3), a))
R4 = pd.DataFrame(rows, columns=["화자", "문장", "CER", "CER(용어 목록)", "받아쓰기"])
print(R4.drop(columns="받아쓰기").groupby("화자")[["CER", "CER(용어 목록)"]].mean().round(3))
for _, r in R4.iterrows():
    if r["CER"] > 0: print(f"화자 {r['화자']} 문장 {r['문장']} : {r['받아쓰기']}")
del asr; gpu_free()
''')

b.md('''
### 셀 5 · 화자 유사도
- WavLM 화자 검증 모델 : 소리 → 화자 벡터(512차원). 두 벡터의 코사인이 1 에 가까울수록 이 모델의 화자 표현이 가까움(앞서 배운 코사인)
- 비교 기준 : 복제 ↔ 자기 참고 음성 / 복제 ↔ 다른 화자 / FLEURS 같은 성별끼리 · 다른 성별끼리
''')
b.code('''
# 셀 5 · 화자 유사도
from transformers import AutoFeatureExtractor, WavLMForXVector
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()
def emb(z):
    with torch.no_grad(): e = sv(**fe(z, sampling_rate=SR, return_tensors="pt").to("cuda")).embeddings
    return torch.nn.functional.normalize(e, dim=-1)[0]
E_ref = {r: emb(wav_of(df, r)) for r in REFS}
rows = []
for r in REFS:
    for w in gen[r]:
        e = emb(w)
        rows.append((r, float(e @ E_ref[r]), np.mean([float(e @ E_ref[o]) for o in REFS if o != r])))
R5 = pd.DataFrame(rows, columns=["화자", "복제↔자기 참고", "복제↔다른 화자"]).groupby("화자").mean().round(3)
print(R5)
E = [emb(wav_of(df, i)) for i in range(20)]; g = df["gender"].iloc[:20].tolist()
same = [float(E[i] @ E[j]) for i in range(20) for j in range(i + 1, 20) if g[i] == g[j]]
diff = [float(E[i] @ E[j]) for i in range(20) for j in range(i + 1, 20) if g[i] != g[j]]
print(f"기준 — FLEURS 20문장 : 같은 성별끼리 {np.mean(same):.3f} · 다른 성별끼리 {np.mean(diff):.3f}")
# 성별 라벨은 화자 ID가 아님. 동일인 정답 대신 각 녹음 쌍과 유사도를 공개
pair_rows = [(i, j, g[i], g[j], float(E[i] @ E[j])) for i in range(20) for j in range(i + 1, 20)]
pair_table = pd.DataFrame(pair_rows, columns=["녹음 A", "녹음 B", "성별 A", "성별 B", "코사인"])
pair_table.to_csv(RESULTS / "06_speaker_pairs.csv", index=False)
male = df.index[df["gender"] == "남"].tolist(); female = df.index[df["gender"] == "여"].tolist()
checks = [(male[k], female[k]) for k in range(6)]
print("오디오 LLM 실험과 같은 남·여 6쌍 — 기준 0.8은 설명용이며 검증된 동일인 임계값이 아님")
for i, j in checks:
    score = float(emb(wav_of(df, i)) @ emb(wav_of(df, j)))
    print(f"녹음 {i} ↔ {j}: 코사인 {score:.3f} · 0.8 미만 {score < 0.8}")
print("같은 성별의 높은 유사도만으로 동일인을 판정할 수 없음: 이 데이터에는 화자 ID 대조가 없음")

''')

b.md('''
### 셀 6 · 참고 음성 길이
- 한 화자·한 문장의 비교. 3초가 모든 화자에게 충분하다는 결론은 낼 수 없음
- 참고 음성을 1초 · 3초 · 전체로 잘라 같은 글을 복제 → 화자 유사도 · CER
- 짧을수록 목소리 정보가 적다. 「3초 복제」(VALL-E · Qwen3-TTS 의 홍보 문구)가 어느 정도인지 직접
''')
b.code('''
# 셀 6 · 참고 음성 길이
tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
r = REFS[0]; rx = wav_of(df, r)
outs = {}
for sec in (1, 3, None):
    clip = rx[: int(sec * SR)] if sec else rx
    w, sr = tts.generate_voice_clone(text=TEXTS[0], language="Korean", ref_audio=(clip, SR), x_vector_only_mode=True)
    outs[sec or "전체"] = librosa.resample(np.asarray(w[0], np.float32), orig_sr=sr, target_sr=SR)
del tts; gpu_free()
print("참고 길이 → 복제↔참고(전체) 화자 유사도")
for k, w in outs.items(): print(f"  {k}초 : {float(emb(w) @ E_ref[r]):.3f}") if k != "전체" else print(f"  전체({len(rx) / SR:.1f}초) : {float(emb(w) @ E_ref[r]):.3f}")
for k, w in outs.items(): play(w, label=f"참고 {k}초")
del sv; gpu_free()          # 셀 7 의 Omni(약 12GB)를 올릴 자리 확보 — 화자 모델은 셀 9 에서 다시 올림
''')

b.md('''
### 셀 7 · 하이브리드 — 토큰 → flow matching → 소리
- Qwen2.5-Omni 의 음성 출력부 : **Talker(LM)** 가 음성 토큰을 내고 → **Token2Wav** 의 DiT 가 그 토큰을 조건으로 노이즈에서 멜을 만들고(flow matching) → 보코더가 파형
- 이미지 생성과 같은 적분: 시각 점 수(적분 구간의 양 끝을 포함한 점 수) · CFG(조건 있는 예측과 없는 예측의 차이를 키움) · sway(초반 스텝에 촘촘히)
- 이 구현의 num_steps = 0~1 사이 시각 점의 개수. RK4(Runge–Kutta 4차 적분)는 구간마다 DiT를 4회 호출. CFG(Classifier-Free Guidance, 조건 유도)는 조건·무조건 예측을 배치 2개로 묶음 → 호출 4 × (점 − 1)회, 예측 개수는 그 2배. 점 1개는 적분 구간 0개
- CFG 식 : 조건 예측 + (조건 예측 − 조건 없는 예측) × g → 이미지 생성에서 쓴 w 로 쓰면 w = 1 + g(기본 g 0.5 = w 1.5)
- 같은 Talker 토큰으로 시각 점 수 · CFG 만 바꿔 다시 만들고 Whisper 로 받아 적어 WER(Thinker 가 낸 글 기준) · 시간. 이 모델의 음성 출력은 영어 · 중국어만 → 영어 문장
''')
b.code('''
# 셀 7 · 하이브리드
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, pipeline
OM = "Qwen/Qwen2.5-Omni-3B"
op = Qwen2_5OmniProcessor.from_pretrained(OM)
om = Qwen2_5OmniForConditionalGeneration.from_pretrained(OM, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
cap = {}; orig = om.token2wav.forward
def rec(code, conditioning, reference_mel, **kw):
    cap.update(code=code, cond=conditioning, ref=reference_mel); return orig(code, conditioning, reference_mel, **kw)
om.token2wav.forward = rec
SAY = "The outlet temperature of reactor three is twelve degrees above the set point."
SYS = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]}, {"role": "user", "content": [{"type": "text", "text": f"Say exactly this sentence and nothing else: {SAY}"}]}]
inp = op(text=op.apply_chat_template(conv, add_generation_prompt=True, tokenize=False), return_tensors="pt").to("cuda")
with torch.no_grad(): ids, wav = om.generate(**inp, speaker="Ethan", thinker_max_new_tokens=64)
dur = wav.numel() / 24000
SPOKEN = op.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print("Thinker 글(Talker 가 읽은 글) :", SPOKEN)
print(f"Talker 토큰 {tuple(cap['code'].shape)} · 소리 {dur:.2f}초 → 초당 {cap['code'].shape[-1] / dur:.1f}개 · 화자 조건 {tuple(cap['cond'].shape)} · 참고 멜 {tuple(cap['ref'].shape)}")
om.thinker.cpu(); om.talker.cpu(); gpu_free()      # 이제부터는 Token2Wav 만 GPU 에
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device="cuda")
rows = []; souts = {}
for steps, gs in [(1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (10, 0.5), (32, 0.5), (10, 1e-4), (10, 2.0)]:   # CFG 0 은 라이브러리 경로 오류 → 0.0001(사실상 끔)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): w = orig(cap["code"], cap["cond"].float(), cap["ref"].float(), num_steps=steps, guidance_scale=gs)
    torch.cuda.synchronize(); dt = time.time() - t0
    y = w.reshape(-1).float().cpu().numpy(); souts[(steps, gs)] = y
    h = asr(librosa.resample(y, orig_sr=24000, target_sr=SR), generate_kwargs={"language": "english"})["text"]
    calls = 4 * (steps - 1)
    predictions = calls * (2 if gs >= 1e-5 else 1)
    rows.append((steps, steps - 1, gs, calls, predictions, round(dt * 1000), round(wer(h, SPOKEN), 3), h))
R7 = pd.DataFrame(rows, columns=["시각 점", "적분 구간", "CFG", "DiT 호출 수(소스 계산)", "조건·무조건 예측 개수", "Token2Wav 시간(ms)", "WER", "받아쓰기"])
print(R7.to_string(index=False)); R7.to_csv(RESULTS / "06_token2wav_steps.csv", index=False)
for k in [(1, 0.5), (2, 0.5), (10, 0.5), (10, 1e-4)]: play(souts[k], 24000, label=f"시각 점 {k[0]} · CFG {k[1]}")
del om, asr; gpu_free()
''')

b.md('''
### 셀 8 · 워터마크
- AudioSeal(Meta) : 사람 귀에 거의 안 들리는 작은 신호를 더해 「AI 가 만든 소리」 표시를 넣고, 탐지기가 구간마다 찾는다
- 넣기 전 · 넣은 뒤 · 8kHz 로 낮췄다 올린 뒤(전화 음질) · 잡음을 섞은 뒤 탐지 확률
''')
b.code('''
# 셀 8 · 워터마크
from audioseal import AudioSeal
wg = AudioSeal.load_generator("audioseal_wm_16bits"); wd = AudioSeal.load_detector("audioseal_detector_16bits")
x = torch.tensor(gen[REFS[0]][0])[None, None]
with torch.no_grad():
    mark = wg.get_watermark(x, SR); y = x + mark
    y8 = torch.tensor(librosa.resample(librosa.resample(y[0, 0].numpy(), orig_sr=SR, target_sr=8000), orig_sr=8000, target_sr=SR))[None, None]
    det = lambda z: round(float(wd.detect_watermark(z, SR)[0]), 3)
    snr = 10 * np.log10(float((x ** 2).sum() / (mark ** 2).sum()))
    print(f"표시 신호 크기 : 원래 소리보다 {snr:.0f}dB 작음")
    print("탐지 확률 — 넣기 전", det(x), "· 넣은 뒤", det(y), "· 전화 음질", det(y8), "· 잡음 섞음", det(y + 0.01 * torch.randn_like(y)))
play(y[0, 0].numpy(), label="워터마크를 넣은 복제 음성")
''')

b.md('''
### 셀 9 · 내 목소리로
- `6-1/data/my_voice/ref.wav`(5~15초, 조용한 곳) 와 `ref.txt`(그 녹음에서 말한 글)를 넣고 이 셀을 실행
- 결과는 `data/my_voice/clone_*.wav` — 저장소에는 올라가지 않음(.gitignore)
''')
b.code('''
# 셀 9 · 내 목소리로
my = DATA / "my_voice"; my.mkdir(exist_ok=True)
if (my / "ref.wav").exists() and (my / "ref.txt").exists():
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
    rx = load_wav(my / "ref.wav"); rt = (my / "ref.txt").read_text(encoding="utf-8").strip()
    wavs, sr = tts.generate_voice_clone(text=TEXTS, language=["Korean"] * len(TEXTS), ref_audio=[(rx, SR)] * len(TEXTS), ref_text=[rt] * len(TEXTS))
    for k, w in enumerate(wavs): sf.write(my / f"clone_{k + 1}.wav", w, sr); play(w, sr, label=f"내 목소리 복제 {k + 1}")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()
    print("화자 유사도(복제 1 ↔ 내 녹음) :", round(float(emb(librosa.resample(np.asarray(wavs[0], np.float32), orig_sr=sr, target_sr=SR)) @ emb(rx)), 3))
else:
    print("data/my_voice/ref.wav · ref.txt 가 없어 건너뜀")
''')

b.save()
