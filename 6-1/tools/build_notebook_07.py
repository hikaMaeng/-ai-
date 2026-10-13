"""6-1 실습 노트북 07 생성기 (강사용).

    python build_notebook_07.py   → ../notebooks/07_실시간_지연.ipynb (출력 없이)
"""
from nb import Book

b = Book("07_실시간_지연.ipynb")

b.md('''
# 실습 07 · 실시간 지연 — 듣고 · 생각하고 · 말하기를 이으면 몇 초 걸리나

현재 강의와의 연결은 아래 셀별 표에 표시. 구조 확인·직접 측정·문헌 설명의 구분.
ASR → 텍스트 LLM → TTS 를 이어 붙인 캐스케이드를 한 GPU 에서 돌려 **단계마다 시간을 측정하고**, 문장 단위로 TTS를 시작한다고 가정한 첫 재생 가능 시각을 계산한다.

| 셀 | 하는 일 | 장표 |
|---|---|---|
| 1 | 설정 — 질문 소리 | – |
| 2 | 세 모델 올리기 — Qwen3-ASR · Qwen3-1.7B · Qwen3-TTS | 음성 이해와 생성을 하나로 |
| 3 | 단계별 시간 — 받아쓰기 · 답 첫 텍스트 조각 · 첫 문장 · 답 전체 · 소리 | 실시간 — 지연은 기다리는 단위에서 생김 |
| 4 | 순차 vs 문장 단위 시작(계산) — 첫 소리까지 | 실시간 — 지연은 기다리는 단위에서 생김 · 지연 도해 — 첫 소리가 나오기까지 |
| 5 | 초당 프레임과 지연 — 1초 소리를 만드는 데 걸리는 시간 | 실시간 — 지연은 기다리는 단위에서 생김 |

- 실행 : GPU 필수(VRAM 14GB). 03 · 06 에서 받은 모델 + Qwen3-1.7B(약 4GB)
- 범위: 일괄 ASR → 텍스트 LLM → 문장/전체 TTS. 스트리밍 인코더·옴니 통합·전이중은 실행하지 않음
- 측정값은 이 PC(GPU · 드라이버 · 다른 프로그램)에 따라 달라진다 — 순서와 비율을 본다
''')

b.md('''
### 셀 1 · 설정
- 질문 소리 : `data/my_voice/question.wav` 가 있으면 그것, 없으면 03 셀 5 가 만든 현장 문장 1(또는 FLEURS 문장)
''')
b.code('''
# 셀 1 · 설정
QUESTION_TEXT_HINT = "운전원이 먼저 확인할 것을 번호 없이 두 문장으로 답하세요."
REF = 1                    # 답 목소리를 빌릴 FLEURS 문장 번호
''')

b.md('''
### 셀 2 · 세 모델 올리기
- 실제 서비스처럼 세 모델을 **미리 올려 둔 상태**에서 측정한다(올리는 시간은 빼고)
''')
b.code('''
# 셀 2 · 세 모델 올리기
import sys, time, threading, re; sys.path.insert(0, ".")
from voice_common import *
import torch
from qwen_asr import Qwen3ASRModel
from qwen_tts import Qwen3TTSModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
print("측정 환경:", torch.cuda.get_device_name(0), "· torch", torch.__version__, "· transformers", __import__("transformers").__version__)
print("초기 GPU 메모리(사용 가능/전체 bytes):", torch.cuda.mem_get_info())
df = fleurs_ko()
q = DATA / "my_voice" / "question.wav"; d1 = DATA / "domain_tts" / "domain_01.wav"
qx = load_wav(q) if q.exists() else load_wav(d1) if d1.exists() else wav_of(df, 0)
print("질문 소리", round(len(qx) / SR, 1), "초 ·", "내 녹음" if q.exists() else "03 합성 문장 1" if d1.exists() else "FLEURS 문장 0")
asr = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0")
tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"); lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, device_map="cuda:0")
tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
rx, rt = wav_of(df, REF), df.iloc[REF]["transcription"]
prompt = tts.create_voice_clone_prompt(ref_audio=(rx, SR), ref_text=rt)   # 참고 음성 처리를 미리 해 둠
print("VRAM 사용", round(torch.cuda.memory_allocated() / 2**30, 1), "GB")
# 데우기(첫 호출의 준비 시간 제거)
_ = asr.transcribe(audio=[(qx, SR)], language="Korean"); _ = tts.generate_voice_clone(text="준비", language="Korean", voice_clone_prompt=prompt)
''')

b.md('''
### 셀 3 · 단계별 시간
- 시계는 **사용자가 말을 끝낸 순간**부터
- LLM 은 토큰이 나오는 대로 받는다(스트리밍) → 첫 텍스트 조각 · 첫 문장 끝 · 전체 끝 시각을 따로 기록. TextIteratorStreamer의 첫 텍스트 조각은 첫 토큰 생성 시각과 다를 수 있음
''')
b.code('''
# 셀 3 · 단계별 시간
def sync(): torch.cuda.synchronize()
T = {}; sync(); t0 = time.time()
text = asr.transcribe(audio=[(qx, SR)], language="Korean")[0].text; sync(); T["받아쓰기 끝"] = time.time() - t0
msgs = [{"role": "user", "content": f"{text}\\n{QUESTION_TEXT_HINT}"}]
ids = tk.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to("cuda")
st = TextIteratorStreamer(tk, skip_prompt=True, skip_special_tokens=True)
th = threading.Thread(target=lambda: lm.generate(ids, max_new_tokens=150, do_sample=False, streamer=st)); th.start()
def sentences(txt):                       # 번호(1. 2.)를 떼고, 8자 이상인 것만 문장으로
    txt = re.sub(r"^\\s*\\d+[.)]\\s*", "", txt, flags=re.M)
    return [x.strip() for x in re.split(r"(?<=[.!?])\\s+|\\n+", txt) if len(x.strip()) >= 8]
reply = ""
for piece in st:
    if "답 첫 텍스트 조각" not in T and piece.strip(): T["답 첫 텍스트 조각"] = time.time() - t0
    reply += piece
    if "답 첫 문장 끝" not in T and sentences(reply) and re.search(r"[.!?]\\s*$", reply.strip()): T["답 첫 문장 끝"] = time.time() - t0
th.join(); T["답 전체 끝"] = time.time() - t0
T.setdefault("답 첫 문장 끝", T["답 전체 끝"])
sents = sentences(reply) or [reply.strip()]
print("받아쓴 질문 :", text); print("답 :", reply.strip())
sync(); t1 = time.time(); w1, sr = tts.generate_voice_clone(text=sents[0], language="Korean", voice_clone_prompt=prompt); sync(); tts_first = time.time() - t1
sync(); t1 = time.time(); wa, sr = tts.generate_voice_clone(text=reply.strip(), language="Korean", voice_clone_prompt=prompt); sync(); tts_all = time.time() - t1
dur1, dura = len(w1[0]) / sr, len(wa[0]) / sr
print({k: round(v, 2) for k, v in T.items()})
print(f"TTS : 첫 문장 {dur1:.1f}초 소리를 {tts_first:.2f}초에 · 답 전체 {dura:.1f}초 소리를 {tts_all:.2f}초에")
play(wa[0], sr, label="답 소리")
''')

b.md('''
### 셀 4 · 순차 vs 문장 단위 시작(계산)
- 순차 : 받아쓰기 → 답 **전체** → 소리 **전체** → 그제야 재생
- 문장 단위 시작의 가정: 답의 **첫 문장**이 나오면 그 문장만 소리로 만들어 재생 시작
- 둘 다 셀 3의 측정값을 더한 계산. 실제 동시 실행·오디오 스트리밍·재생 장치 지연은 측정하지 않음
- 같은 GPU에서 LLM과 TTS를 동시에 실행하면 자원 경쟁으로 달라질 수 있음
''')
b.code('''
# 셀 4 · 순차 vs 문장 단위 시작(계산)
seq = T["답 전체 끝"] + tts_all
ovl = T["답 첫 문장 끝"] + tts_first
R = pd.DataFrame([("받아쓰기", 0, T["받아쓰기 끝"]), ("LLM 답(전체)", T["받아쓰기 끝"], T["답 전체 끝"]), ("TTS(전체)", T["답 전체 끝"], seq)], columns=["단계", "시작", "끝"])
R2 = pd.DataFrame([("받아쓰기", 0, T["받아쓰기 끝"]), ("LLM 첫 문장", T["받아쓰기 끝"], T["답 첫 문장 끝"]), ("TTS 첫 문장", T["답 첫 문장 끝"], ovl)], columns=["단계", "시작", "끝"])
fig, ax = plt.subplots(2, 1, figsize=(10, 3.6), sharex=True)
for a, Rr, ttl, end in ((ax[0], R, "순차", seq), (ax[1], R2, "문장 단위 시작(계산)", ovl)):
    for i, r in Rr.iterrows(): a.barh(i, r["끝"] - r["시작"], left=r["시작"]); a.text(r["끝"] + 0.02, i, f"{r['단계']} {r['끝']:.2f}초", va="center", fontsize=8)
    a.set_yticks([]); a.set_title(f"{ttl} — 첫 재생 가능 시각 {end:.2f}초")
ax[1].set_xlabel("말을 끝낸 뒤 초"); plt.tight_layout(); plt.savefig(RESULTS / "07_latency.png", dpi=150); plt.show()
print(f"첫 재생 가능 시각(측정값을 더한 계산) : 순차 {seq:.2f}초 → 문장 단위 {ovl:.2f}초")
''')

b.md('''
### 셀 5 · 초당 프레임과 지연
- 셀 3은 완성된 파형을 반환하는 호출. 프레임당 값은 전체 생성 시간 ÷ 추정 프레임 수인 평균이며 개별 프레임 도착 간격이 아님
- 목소리 복제 실습은 TTS 단독, 여기서는 ASR·LLM·TTS 동시 상주. 문장·배치·참고 처리도 달라 RTF 차이의 원인을 하나로 단정할 수 없음
- Qwen3-TTS 는 1초 소리 = 12.5 프레임. 프레임마다 LM 이 0번 코드북 하나 + 코드 예측기가 나머지 15개
- 실시간이 되려면 **프레임 하나를 80ms(= 1초 ÷ 12.5) 안에** 만들어야 한다
- 프레임 레이트가 높은 코덱(EnCodec 75Hz)이라면 같은 1초에 프레임 75개 → 프레임당 13ms 안에 만들어야 함(장표 「실시간 — 지연은 기다리는 단위에서 생김」의 프레임 예산)
''')
b.code('''
# 셀 5 · 초당 프레임과 지연
frames = dura * 12.5
print(f"답 소리 {dura:.1f}초 = 추정 프레임 {frames:.0f}개 · TTS {tts_all:.2f}초 → 평균 프레임당 {tts_all / frames * 1000:.0f}ms (예산 80ms) · RTF {tts_all / dura:.2f}")
for name, fps in (("Qwen3-TTS · Mimi", 12.5), ("Qwen2.5-Omni Talker(06 셀 7)", 50), ("EnCodec", 75)):
    print(f"{name:28s} 초당 {fps:5.1f} 프레임 → 프레임 하나 {1000 / fps:5.1f}ms 안에")
del asr, lm, tts; gpu_free()
''')

b.save()
