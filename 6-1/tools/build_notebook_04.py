"""6-1 실습 노트북 04 생성기 (강사용).

    python build_notebook_04.py   → ../notebooks/04_오디오_LLM.ipynb (출력 없이)
"""
from nb import Book

b = Book("04_오디오_LLM.ipynb")

b.md('''
# 실습 04 · 오디오 LLM — 소리를 LLM 의 입력 토큰 자리에 넣기

장표 「음성 인식에서 음성 이해로」 · 「음성 이해의 상세 단계」 · 「오디오 LLM 텐서 흐름」을 Qwen2.5-Omni-3B 로 확인한다.
5-1 의 VLM(그림 → 비전 인코더 → 커넥터 → LLM)과 **같은 자리표 방식**인지, 전사문만 준 LLM 이 놓치는 것이 무엇인지 본다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 | – |
| 2 | 모델 · 부품별 파라미터 수(오디오 인코더 · LLM · Talker · Token2Wav) | 음성 이해의 상세 단계 |
| 3 | 오디오 토큰 자리표 — 초당 몇 개가 LLM 입력에 들어가나 | 오디오 LLM 텐서 흐름 · 초당 몇 개 |
| 4 | 커넥터 — 오디오 인코더 폭 → LLM 폭 | 음성 이해의 상세 단계 3번 |
| 5 | 전사문만 vs 소리 — 화자 성별을 맞히나 | 음성 인식에서 음성 이해로 1번 |
| 6 | 몇 명이 말하나 — 두 화자를 이어 붙인 소리 | 음성 이해의 상세 단계 1번 |
| 7 | 길이의 비용 — 1분 · 10분 · 1시간 소리는 토큰 몇 개 | 초당 몇 개 |

- 실행 : GPU 필수(VRAM 12GB 이상). Qwen2.5-Omni-3B(약 11GB)를 처음 한 번 받음
- 이 노트북은 **텍스트 출력만** 쓴다(음성 출력부 Talker 는 06 셀 7 에서)
''')

b.md('''
### 셀 1 · 설정
- `N_EVAL` : 셀 5 에서 성별을 물을 문장 수 · `N_PAIR` : 셀 6 에서 만들 두 화자 쌍 수
''')
b.code('''
# 셀 1 · 설정
N_EVAL = 20
N_PAIR = 6
MODEL = "Qwen/Qwen2.5-Omni-3B"
''')

b.md('''
### 셀 2 · 모델 · 부품별 파라미터 수
- Thinker = 오디오 인코더 + 비전 인코더 + LLM(3B) · Talker = 음성 토큰 생성 LM · Token2Wav = 토큰 → 소리(06 에서)
- 오디오 인코더는 Whisper-large-v3 로 시작해 학습한 것(Qwen2.5-Omni 보고서) — 02 의 그 인코더 구조
''')
b.code('''
# 셀 2 · 모델 · 부품별 파라미터 수
import sys, time; sys.path.insert(0, ".")
from voice_common import *
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, WhisperForConditionalGeneration, WhisperProcessor
df = fleurs_ko()
proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)
m = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
th = m.thinker
print(f"전체 {n(m):,.0f}M")
print(f"  Thinker : 오디오 인코더 {n(th.audio_tower):,.0f}M · 비전 인코더 {n(th.visual):,.0f}M · LLM {n(th.model):,.0f}M")
print(f"  Talker {n(m.talker):,.0f}M · Token2Wav {n(m.token2wav):,.0f}M")
print("오디오 인코더 층 수", len(th.audio_tower.layers), "· 폭", th.audio_tower.config.d_model, "· LLM 폭", th.config.text_config.hidden_size)
m.disable_talker(); gpu_free()           # 이 노트북은 텍스트 출력만 → 음성 출력부를 내려 메모리 확보
SYS = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."   # 모델 기본 시스템 글
def ask(audios, question, max_new=120):
    content = [{"type": "audio", "audio": a} for a in audios] + [{"type": "text", "text": question}]
    conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]}, {"role": "user", "content": content}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=audios or None, return_tensors="pt", padding=True).to("cuda").to(torch.bfloat16)
    with torch.no_grad(): g = m.generate(**inp, max_new_tokens=max_new, do_sample=False, return_audio=False)
    return proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(), inp
''')

b.md('''
### 셀 3 · 오디오 토큰 자리표
- 프롬프트 안의 소리 하나는 **자리표 토큰 여러 개**로 펼쳐진다 → 그 자리에 오디오 인코더 출력 벡터가 들어감(5-1 VLM 의 그림 자리표와 같은 방식)
- 자리표 개수 ÷ 소리 길이 = 초당 오디오 토큰 수
- 인코더 안 : 멜(초당 100) → 합성곱 stride 2(초당 50, Whisper 와 같음) → 평균 풀링 2(초당 25)
''')
b.code('''
# 셀 3 · 오디오 토큰 자리표
aid = th.config.audio_token_id if hasattr(th.config, "audio_token_id") else th.config.audio_token_index
rows = []
for i in range(4):
    z = wav_of(df, i)
    _, inp = ask([z], "이 소리는 무엇인가요?", max_new=1)
    k = int((inp["input_ids"] == aid).sum())
    rows.append((round(len(z) / SR, 2), int(inp["input_ids"].shape[1]), k, round(k / (len(z) / SR), 1)))
print(pd.DataFrame(rows, columns=["소리 길이(초)", "입력 토큰 전체", "오디오 자리표", "초당 오디오 토큰"]).to_string(index=False))
ids = inp["input_ids"][0].tolist(); a0 = ids.index(aid)
print("자리표 앞뒤 :", proc.tokenizer.convert_ids_to_tokens(ids[a0 - 3:a0 + 2]), "...", proc.tokenizer.convert_ids_to_tokens(ids[a0 + k - 1:a0 + k + 3]))
''')

b.md('''
### 셀 4 · 커넥터
- 오디오 인코더가 낸 벡터(폭 1280)를 LLM 의 토큰 벡터(폭 2048)와 같은 폭으로 바꾸는 선형층 하나 = 커넥터(투영)
- 훅으로 인코더 출력 · 투영 출력 모양을 찍는다
''')
b.code('''
# 셀 4 · 커넥터
shp = []
at = th.audio_tower
hs = [at.proj.register_forward_hook(lambda mod, i, o: shp.append(("투영(커넥터) 입력 → 출력", tuple(i[0].shape), tuple(o.shape))))]
z = wav_of(df, 0); ask([z], "이 소리는 무엇인가요?", max_new=1)
for h in hs: h.remove()
print("투영 웨이트", tuple(at.proj.weight.shape), "(나가는 폭 × 들어오는 폭)")
for s in shp: print(s)
''')

b.md('''
### 셀 5 · 전사문만 vs 소리 — 화자 성별
- FLEURS 에는 화자 성별 라벨이 있다 → 정답과 비교 가능
- ① 소리를 그대로 주고 묻기 — **질문 방식 세 가지** ② 같은 소리를 Whisper 로 받아 적은 **글만** 주고 묻기
- 전사문에는 목소리가 남지 않는다 → 「전사문만 LLM 에 넣으면 비언어적 정보가 사라짐」(장표)을 숫자로
- 덤 : 같은 모델 · 같은 소리인데 **질문 방식**에 따라 정답률이 크게 바뀐다 — 2강 「앞에 둔 글이 다음 토큰 분포를 고른다」
''')
b.code('''
# 셀 5 · 전사문만 vs 소리
QS = {"한국어 · 한 단어만, 모르면 '모름'": "화자가 남성인지 여성인지 '남성' 또는 '여성' 한 단어로만 답하세요. 판단할 수 없으면 '모름'.",
      "한국어 · 받아 적고 성별도": "이 음성을 받아 적고, 화자의 성별을 말해 주세요.",
      "영어 · male or female": "Is the speaker male or female? Answer with one word."}
def label(a):
    a = a.lower()
    return "여" if ("여" in a or "female" in a or "woman" in a) else "남" if ("남" in a or "male" in a or "man" in a) else "모름"
idx = list(range(N_EVAL)); gold = df["gender"].iloc[:N_EVAL].tolist()
rows = []
for name, q in QS.items():
    L = [label(ask([wav_of(df, i)], q, max_new=40)[0]) for i in idx]
    rows.append(("소리", name, np.mean([l == g for l, g in zip(L, gold)]), pd.Series(L).value_counts().to_dict()))
wp = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", torch_dtype=torch.float16).cuda().eval()
def whisper_text(z):
    f = wp(z, sampling_rate=SR, return_tensors="pt").input_features.cuda().half()
    with torch.no_grad(): return wp.tokenizer.decode(wm.generate(f, language="korean")[0], skip_special_tokens=True)
texts = [whisper_text(wav_of(df, i)) for i in idx]; del wm; gpu_free()
L = [label(ask([], f"Here is a transcript of what a person said:\\n\\n{t}\\n\\n{QS['영어 · male or female']}", max_new=40)[0]) for t in texts]
rows.append(("전사문만", "영어 · male or female", np.mean([l == g for l, g in zip(L, gold)]), pd.Series(L).value_counts().to_dict()))
R5 = pd.DataFrame(rows, columns=["넣은 것", "질문 방식", "정답률", "답 분포"])
print(R5.to_string(index=False)); print("정답 분포 :", pd.Series(gold).value_counts().to_dict())
R5.to_csv(RESULTS / "04_gender.csv", index=False)
''')

b.md('''
### 셀 6 · 몇 명이 말하나
- 두 명 : 남 · 여 화자의 문장을 0.5초 쉼을 두고 이어 붙임
- 한 명 : 한 화자의 문장을 반으로 잘라 0.5초 쉼을 두고 이어 붙임(같은 사람이 확실한 경우)
''')
b.code('''
# 셀 6 · 몇 명이 말하나
male = df.index[df["gender"] == "남"].tolist(); female = df.index[df["gender"] == "여"].tolist()
Q2 = "How many different speakers are in this audio? Answer with a number only."
gap = np.zeros(SR // 2, np.float32); rows = []
for k in range(N_PAIR):
    two = np.concatenate([wav_of(df, male[k]), gap, wav_of(df, female[k])])
    z = wav_of(df, female[k + N_PAIR]); one = np.concatenate([z[: len(z) // 2], gap, z[len(z) // 2:]])
    rows.append(("2명(남+여)", ask([two], Q2, max_new=4)[0])); rows.append(("1명(한 문장을 둘로)", ask([one], Q2, max_new=4)[0]))
R6 = pd.DataFrame(rows, columns=["실제", "모델 답"])
print(R6.groupby("실제")["모델 답"].agg(lambda s: s.value_counts().to_dict()))
''')

b.md('''
### 셀 7 · 길이의 비용
- LLM 이 한 번에 볼 수 있는 토큰 수는 정해져 있다(이 모델 32,768). 초당 토큰이 적을수록 긴 소리를 넣을 수 있다
- 2025~26 모델이 초당 25 → 12.5 → 6.25 로 줄여 온 이유
''')
b.code('''
# 셀 7 · 길이의 비용
rows = []
for name, r in [("Qwen2-Audio · 2.5-Omni(측정, 셀 3)", 25), ("Qwen3-Omni · Qwen3-ASR (AuT)", 12.5), ("Qwen3.5-Omni(보고서)", 6.25)]:
    rows.append((name, r, int(60 * r), int(600 * r), int(3600 * r), round(32768 / r / 60, 1)))
print(pd.DataFrame(rows, columns=["모델", "초당 토큰", "1분", "10분", "1시간", "32k 토큰에 들어가는 분"]).to_string(index=False))
''')

b.save()
