"""8-1 실습 노트북 02 생성기 (강사용).

    python build_notebook_02.py   → ../notebooks/02_워크플로우_세_패턴과_에이전트.ipynb (출력 없이)
"""
from nb import Book

b = Book("02_워크플로우_세_패턴과_에이전트.ipynb")

b.md('''
# 실습 02 · 워크플로우 세 패턴과 에이전트 — 같은 문제를 두 방식으로 푼다

**직접 해 보는 것** : 경로가 코드에 박혀 있는 **워크플로우**의 세 패턴(프롬프트 체이닝 · 라우팅 · 병렬화)을 직접 구현하고 그림으로 본다. 그다음 **같은 문제를 에이전트(모델 + 도구 + 루프)로도** 풀어, 성공률 · LLM 호출 수 · 토큰 · 지연을 측정해 나란히 놓는다. 마지막 과제는 일부러 **경로를 미리 그릴 수 없는 문제**로, 워크플로우가 어디서 막히는지 본다.

**강의의 어느 개념과 이어지는가**
- 제어 흐름의 위치가 유일한 구분선 : 경로가 코드에 고정이면 **워크플로우**, 런타임에 모델이 결정하면 **에이전트**
- 워크플로우 패턴 : 프롬프트 체이닝(A의 출력 → B의 입력) · 라우팅(입력을 분류해 미리 정한 분기로) · 병렬화(독립 하위작업을 동시에 처리 후 코드가 취합)
- 에이전트 패턴 : 모델이 상태를 보고 도구를 선택하고, 관찰을 보고 다음을 정함 · 종료도 모델이 판단 · 같은 입력이어도 궤적이 달라질 수 있음(비결정성)
- 실무 원칙 : 고정 절차로 풀리면 워크플로우가 더 싸고 안정적, 경로가 사전에 예측 불가일 때만 에이전트의 자율성이 값을 함 — **자율성은 성능 옵션이 아니라 비용 · 위험을 지불하는 선택**
- 판정 : 경로를 코드로 그릴 수 있으면 워크플로우, 없으면 에이전트

**과제 네 개** (모두 노트북이 만든 합성 데이터 · 정답은 규칙 코드가 채점)

| 과제 | 내용 | 맞는 패턴 | 경로를 코드로 그릴 수 있는가 |
|---|---|---|---|
| A | 공지문 → 핵심 사실 추출 → 개조식 요약 → 영어 번역 | 프롬프트 체이닝 | 예 |
| B | 고객 문의 → 환불 / 기술지원 / 일반 분류 → 분기별 응답 | 라우팅 | 예 |
| C | 회의록 네 구간을 동시에 요약 → 코드가 합침 | 병렬화 | 예 |
| D | 가상 주문 DB · 환불 정책으로 푸는 질문 — 질문마다 필요한 조회 순서가 다름 | (고정 경로로는 어려움) | 아니오 |

| 셀 | 하는 일 |
|---|---|
| 1 | 설정 |
| 2 | 과제 네 개 살펴보기 |
| 3 | 패턴 ① 프롬프트 체이닝 — 그림 · 구현 · 한 건 실행 |
| 4 | 패턴 ② 라우팅 |
| 5 | 패턴 ③ 병렬화 — 직렬 합 vs 겹쳐 돈 시간 |
| 6 | 워크플로우로 A · B · C 전체 측정 |
| 7 | 같은 과제 A · B · C 를 에이전트로 — 측정 · 비교 |
| 8 | 과제 D 워크플로우 — 고정 경로가 막히는 곳 |
| 9 | 과제 D 에이전트 — 유형별 성공률 비교 |
| 10 | 같은 입력 · 다른 궤적 — 비결정성 |
| 11 | 판정표 · 결과 저장 · 확인 질문 |

- 실행 : LM Studio 서버가 켜져 있고 도구 호출이 되는 모델이 로드되어 있어야 한다(없으면 `AGENT_MODE=replay` 로 강사 실행 결과를 재생). 설정은 모두 출력에 찍는다
- **결과가 기대와 달라도 그대로 남긴다.** 에이전트가 더 낫게 나오는 곳도, 워크플로우가 더 낫게 나오는 곳도 있고, 작은 모델에서는 둘 다 흔들릴 수 있다
''')

b.md('''
### 셀 1 · 설정
- 아래 `CFG` 의 개수를 줄이면 서버를 덜 부른다(처음 호출한 응답은 `cache/` 에 저장되어 다시 돌릴 때 재사용)
''')
b.code('''
# 셀 1 · 설정
import sys, json, time; sys.path.insert(0, ".")
from agent_lab import *
from agent_tasks import *

CFG = dict(n_A=8, n_B=28, n_C=6, max_steps=8, max_tokens=300, 비결정성_횟수=10, 온도=0.7, 병렬_작업자=4)
llm = LLM()          # 주소 · 모델은 환경변수(LMSTUDIO_URL · AGENT_MODEL)에서 읽는다
DOCS = {d["id"]: d for d in make_docs_A(CFG["n_A"])}
INQ = make_inquiries_B()[: CFG["n_B"]]
MEET = make_meetings_C(CFG["n_C"])
orders, cust, QD = make_questions_D(0)
approx = CFG["n_A"] * (3 + 3) + CFG["n_B"] * (2 + 3) + CFG["n_C"] * (4 + 6) + len(QD) * (2 + 5) + CFG["비결정성_횟수"] * (2 + 5)
print(llm, "| 설정 :", CFG)
print(f"이 노트북은 LLM 을 약 {approx}번 부른다(캐시에 있으면 서버를 부르지 않는다).")
''')

b.md('''
### 셀 2 · 과제 네 개 살펴보기
입력 예와 **채점 기준**을 본다. 채점은 모두 규칙 코드다(사람이 보지 않음).
- A : 영어로 썼는가(영문자 중 ASCII 비율) + 핵심 숫자 네 개(행사일 · 참가비 · 신청 마감일 · 정원)가 출력에 있는가
- B : 분기별 응답이 반드시 담아야 할 말(환불 → 「영업일」 + 주문번호 / 기술지원 → 「버전」 / 일반 → 「9시」) + 분류가 맞는가
- C : 회의록 네 구간에 심어 둔 사실 네 개(예산 · 담당자 · 마감일 · 위험)가 합친 요약에 모두 있는가
- D : 정책을 코드로 옮겨 계산한 판정 · 금액이 답에 있는가
''')
b.code('''
# 셀 2 · 과제 네 개
d0 = next(iter(DOCS.values()))
print("[A] 입력 :", d0["text"]); print("    정답 사실 :", d0["facts"])
print("\\n[B] 입력 :", INQ[0]["text"], "| 정답 분류 :", INQ[0]["label"], "| 응답에 꼭 담을 것 :", REQUIRE_B[INQ[0]["label"]])
print("\\n[C] 구간 1 :", MEET[0]["sections"][0]); print("    심어 둔 사실 :", MEET[0]["facts"])
print("\\n[D] 질문 유형 :", pd.Series([q["유형"] for q in QD]).value_counts().to_dict())
print("    예 :", QD[0]["q"], "→ 정답", QD[0]["expect"])
print("    정책 :", POLICY_TEXT)
''')

b.md('''
### 셀 3 · 패턴 ① 프롬프트 체이닝
- **A 의 출력이 B 의 입력**이 되도록 단계를 고정 연결한다. 분기 · 반복이 없는 한 줄 경로다
- 공지문 → ⓐ 핵심 추출 → ⓑ 개조식 요약 → ⓒ 영어 번역. 각 단계는 LLM 한 번이고, 단계의 순서를 정한 것은 **코드**다
- 그림 : 먼저 경로만, 한 건을 실행한 뒤에는 각 단계가 걸린 시간 · 토큰을 노드에 표시한다(밝을수록 오래 걸림)
''')
b.code('''
# 셀 3 · 프롬프트 체이닝
def ask(tr, name, prompt, max_tokens=CFG["max_tokens"], stage=None, lane=0, sample=0, temperature=0.0):
    r = tr.llm(llm, name, [{"role": "user", "content": prompt}], stage=stage, lane=lane, max_tokens=max_tokens, sample=sample, temperature=temperature)
    return r.content.strip()

def wf_A(doc, tr=None):
    tr = tr or Trace()
    facts = ask(tr, "핵심 추출", f"다음 공지에서 핵심 사실(행사일 · 참가비 · 신청 마감일 · 정원)을 한 줄씩 추출해라.\\n\\n공지: {doc['text']}")
    brief = ask(tr, "개조식 요약", f"아래 원문을 개조식(짧은 항목 3~4개, 각 항목은 '- ' 로 시작)으로 요약해라. 숫자는 그대로 쓴다.\\n\\n원문: {facts}")
    eng = ask(tr, "영어 번역", f"아래 원문을 영어로 번역해라. 숫자는 그대로 쓴다. 번역문만 출력하고 설명 · 선택지 · 한국어 문장은 덧붙이지 않는다.\\n\\n원문: {brief}")
    return eng, tr

SPEC_A = spec_chain(["핵심 추출", "개조식 요약", "영어 번역"])
out, tr = wf_A(d0)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 4.6))
draw_workflow(SPEC_A, title="경로(코드가 고정)", ax=a1); draw_workflow(SPEC_A, trace=tr, title="한 건을 실행한 뒤(칸마다 걸린 시간 · 토큰)", ax=a2)
plt.tight_layout(); plt.show()
print("출력 :", out[:300]); print("채점 :", score_A(d0, out)); print("호출 통계 :", tr.stats())
''')

b.md('''
### 셀 4 · 패턴 ② 라우팅
- 입력을 **분류**해 미리 정한 분기로 보낸다(`if / switch`). 분기의 개수 · 이름 · 각 분기의 처리는 코드가 정한다
- 고객 문의 → ⓐ 분류(LLM, 한 단어) → ⓑ 분기별 응답(LLM). 실행한 분기에만 시간이 표시된다
- 분류가 틀리면 그 뒤 경로가 통째로 틀린 분기로 간다 — 경로가 고정이라서 생기는 약점도 함께 본다
''')
b.code('''
# 셀 4 · 라우팅
def wf_B(item, tr=None):
    tr = tr or Trace()
    raw = ask(tr, "분류", f"고객 문의를 환불 / 기술지원 / 일반 중 하나로 분류해라. 한 단어만 답한다.\\n\\n문의: {item['text']}", max_tokens=20)
    cat = next((c for c in CATS if c in raw), "일반")                              # 분류 결과를 분기 이름으로 바꾸는 것도 코드
    ans = ask(tr, f"{cat} 응답", f"고객 문의에 응답해라. 응대 규칙 : {REQUIRE_B[cat]}\\n\\n문의: {item['text']}")
    return ans, cat, tr

SPEC_B = spec_route("분류", ["환불 응답", "기술지원 응답", "일반 응답"])
item = INQ[0]
ans, cat, tr = wf_B(item)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6))
draw_workflow(SPEC_B, title="경로(코드가 고정)", ax=a1); draw_workflow(SPEC_B, trace=tr, title="한 건을 실행한 뒤(실행한 분기만 표시)", ax=a2)
plt.tight_layout(); plt.show()
print("문의 :", item["text"], "| 분류 :", cat, "| 정답 :", item["label"])
print("응답 :", ans[:200]); print("채점 :", score_B(item, ans, routed=cat))
''')

b.md('''
### 셀 5 · 패턴 ③ 병렬화
- **독립 하위작업**을 동시에 처리하고, 결과를 **코드가** 취합한다. 회의록 네 구간을 각각 요약(LLM 네 번)하고 코드가 이어 붙인다
- 이 실습의 지연은 「단계별로 가장 긴 호출의 합」으로 계산한다(같은 단계에서 동시에 돈 호출은 겹쳐 센다). 서버가 동시 요청을 하나씩 처리하면 실제 시간은 더 길 수 있다
- 아래에서 **직렬로 돌았다면의 합**과 **겹쳐 돈 시간**을 나란히 본다
''')
b.code('''
# 셀 5 · 병렬화
def wf_C(m, tr=None):
    tr = tr or Trace()
    secs = m["sections"]
    def one(i, s, stage):
        return ask(tr, f"요약 {i + 1}", f"회의록 구간 {i + 1}을 두 문장으로 요약해라. 숫자 · 이름 · 날짜는 빠뜨리지 않는다.\\n\\n구간 원문: {s}", stage=stage, lane=i)
    outs = parallel_map(tr, "요약", one, secs, max_workers=CFG["병렬_작업자"])        # 같은 단계 = 동시에 돈 것
    merged = tr.code("합치기", lambda: "\\n".join(f"[{i + 1}] {o}" for i, o in enumerate(outs)))   # 취합은 코드
    return merged, tr

SPEC_C = spec_parallel("구간 나누기", ["요약 1", "요약 2", "요약 3", "요약 4"], "합치기")
m0 = MEET[0]
out, tr = wf_C(m0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2))
draw_workflow(SPEC_C, title="경로(코드가 고정)", ax=a1); draw_workflow(SPEC_C, trace=tr, title="한 건을 실행한 뒤", ax=a2)
plt.tight_layout(); plt.show()
serial = sum(s.dur for s in tr.spans if s.kind == "llm")
print(f"직렬로 돌았다면 LLM 시간의 합 {serial:.1f}초 | 겹쳐 돈 시간(단계별 최댓값의 합) {tr.latency:.1f}초 | LLM 호출 {tr.llm_calls}번 · 토큰 {tr.tokens}")
print("채점 :", score_C(m0, out))
''')

b.md('''
### 셀 6 · 워크플로우로 A · B · C 전체를 측정한다
- 같은 코드를 모든 입력에 돌려 성공률 · LLM 호출 수 · 토큰 · 지연의 평균을 모은다
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
- 워크플로우는 입력이 달라도 **호출 수가 같다**(A 3 · B 2 · C 4). 경로가 고정이기 때문이다
''')
b.code('''
# 셀 6 · 워크플로우로 A · B · C 전체
ROWS = []
def add(과제, 방식, 성공, tr, **ex):
    ROWS.append({"과제": 과제, "방식": 방식, "성공": bool(성공), "호출": tr.llm_calls, "토큰": tr.tokens, "지연": tr.latency, **ex})

# 입력마다 서로 독립이므로 pmap 으로 겹쳐 돌리고, 결과는 입력 순서대로 표에 쌓는다
for d, (o, tr) in zip(DOCS.values(), pmap(wf_A, list(DOCS.values()))):
    add("A 체이닝", "워크플로우", score_A(d, o)["성공"], tr)
for it, (o, cat, tr) in zip(INQ, pmap(wf_B, INQ)):
    add("B 라우팅", "워크플로우", score_B(it, o, routed=cat)["성공"], tr, 분류=(cat == it["label"]))
for m, (o, tr) in zip(MEET, pmap(wf_C, MEET)):
    add("C 병렬화", "워크플로우", score_C(m, o)["성공"], tr)
df = pd.DataFrame(ROWS)
print(df.groupby(["과제", "방식"]).agg(성공률=("성공", "mean"), 호출=("호출", "mean"), 토큰=("토큰", "mean"), 지연=("지연", "mean"), 건수=("성공", "size")).round(2).to_string())
''')

b.md('''
### 셀 7 · 같은 과제 A · B · C 를 에이전트로 푼다
- **같은 모델 · 같은 입력 · 같은 채점**이다. 다른 것은 경로를 누가 정하는가뿐이다
- 에이전트에게는 단계를 알려 주지 않고 **목표와 도구**만 준다. A 는 문서를 읽는 도구, B 는 응대 규칙을 알려 주는 도구 세 개, C 는 회의록 구간을 읽는 도구
- 에이전트가 **어떤 도구를 몇 번 부르는지는 모델이 정한다**. 워크플로우의 고정된 호출 수와 비교한다
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
''')
b.code('''
# 셀 7 · 같은 과제를 에이전트로
SYS_A = "공지를 읽고 핵심 사실을 개조식으로 요약한 뒤 영어로 번역해 최종 답으로 낸다. 숫자는 그대로 쓴다. 공지 원문은 get_document 도구로 읽는다."
SYS_B = "고객 문의에 응대한다. 문의 종류에 맞는 응대 규칙을 도구로 먼저 확인하고, 그 규칙대로 답한다."
SYS_C = "회의록을 읽고 핵심(예산 · 담당자 · 마감일 · 위험)을 빠짐없이 정리해 답한다. 회의록 구간은 도구로 읽는다."
TOOL_A = [Tool("get_document", "공지 문서의 원문을 돌려준다.", {"properties": {"doc_id": {"type": "string", "description": "예 : A01"}}, "required": ["doc_id"]}, lambda doc_id: DOCS[str(doc_id).strip()]["text"])]

def ag(system, user, tools, tr=None, **kw):
    return run_agent(llm, system, user, tools, max_steps=CFG["max_steps"], max_tokens=CFG["max_tokens"], trace=tr, **kw)

# 입력마다 서로 독립 → pmap 으로 겹쳐 돌리고 결과는 입력 순서대로 쌓는다
for d, t in zip(DOCS.values(), pmap(lambda d: ag(SYS_A, f"문서 {d['id']} 를 처리해라.", TOOL_A), list(DOCS.values()))):
    add("A 체이닝", "에이전트", score_A(d, t.final)["성공"], t.trace, 호출도구=len(t.calls))
for it, t in zip(INQ, pmap(lambda it: ag(SYS_B, f"문의: {it['text']}", tools_B()), INQ)):
    add("B 라우팅", "에이전트", score_B(it, t.final)["성공"], t.trace, 분류=(t.calls[:1] == [B_TOOL_FOR[it["label"]]]), 호출도구=len(t.calls))
for m, t in zip(MEET, pmap(lambda m: ag(SYS_C, f"회의록 {m['id']} 를 정리해라.", tools_C(m)), MEET)):
    add("C 병렬화", "에이전트", score_C(m, t.final)["성공"], t.trace, 호출도구=len(t.calls))
df = pd.DataFrame(ROWS)
g = df.groupby(["과제", "방식"]).agg(성공률=("성공", "mean"), 호출=("호출", "mean"), 토큰=("토큰", "mean"), 지연=("지연", "mean"), 건수=("성공", "size")).round(2)
print(g.to_string())
tasks = ["A 체이닝", "B 라우팅", "C 병렬화"]
fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
for a, (col, ttl) in zip(ax, [("성공률", "성공률"), ("호출", "LLM 호출 수"), ("토큰", "토큰"), ("지연", "지연(초)")]):
    x = np.arange(len(tasks))
    for k, (meth, c) in enumerate([("워크플로우", COLOR["code"]), ("에이전트", COLOR["llm"])]):
        v = [g.loc[(t_, meth), col] for t_ in tasks]
        a.bar(x + (k - 0.5) * 0.38, v, 0.38, color=c, ec="#444", lw=0.8, label=meth)
    a.set_xticks(x); a.set_xticklabels([t_[0] for t_ in tasks]); a.set_title(ttl, fontsize=10, loc="left")
ax[0].legend(fontsize=8, frameon=False); plt.tight_layout(); plt.show()
''')

b.md('''
### 셀 8 · 과제 D — 고정 경로가 막히는 곳
- 질문마다 **필요한 조회 순서가 다르다** : 주문번호가 주어지면 「조회 → 정책 → 계산」이면 되지만, 이름과 상품만 주어지면 「고객 검색 → 주문 조회 → …」, 한 고객의 모든 주문을 합하라면 조회 횟수 자체가 달라진다
- 워크플로우로 푼다면 : ⓐ 질문에서 주문번호 추출(LLM) → ⓑ 주문 조회(코드) → ⓒ 정책에 따라 판정(LLM). 주문번호가 없는 질문은 이 경로로 **갈 수 없다**
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
- 이 경로를 확장해 이름 · 합계 질문도 처리하려면 분기를 더 설계해야 한다(확인 질문)
''')
b.code('''
# 셀 8 · 과제 D 워크플로우
lookup = {t.name: t for t in tools_D(orders, cust)}["lookup_order"].fn

def wf_D(q, tr=None, sample=0, temperature=0.0):
    tr = tr or Trace()
    raw = ask(tr, "추출", f"질문에서 주문번호를 추출해라. 주문번호는 A와 숫자 네 자리다. 없으면 없음이라고만 답한다.\\n\\n질문 : {q['q']}", max_tokens=30, sample=sample, temperature=temperature)
    m = re.search(r"A\\d{4}", raw)
    if not m:
        return "질문에 주문번호가 없어 확인할 수 없습니다.", tr, ("추출", "끝(주문번호 없음)")
    rec = tr.code("조회", lookup, m.group(0))
    if rec.startswith("오류"):
        return "해당 주문을 찾을 수 없습니다.", tr, ("추출", "조회", "끝(주문 없음)")
    ans = ask(tr, "판정", f"다음 정책에 따라 환불 가능 여부와 금액을 답해라. 금액은 원 단위 정수로 쓴다.\\n\\n{POLICY_TEXT}\\n\\n주문 정보: {rec}\\n\\n질문 : {q['q']}", sample=sample, temperature=temperature)
    return ans, tr, ("추출", "조회", "판정", "끝")

RD = []
for q, (a, tr, path) in zip(QD, pmap(wf_D, QD)):          # 질문 18개는 서로 독립 → 겹쳐 돌린다
    RD.append({"유형": q["유형"], "방식": "워크플로우", "성공": score_D(q, a)["성공"], "호출": tr.llm_calls, "토큰": tr.tokens, "지연": tr.latency})
    add("D 개방형", "워크플로우", score_D(q, a)["성공"], tr)
dw = pd.DataFrame(RD)
print(dw.groupby("유형").agg(성공률=("성공", "mean"), 건수=("성공", "size"), 호출=("호출", "mean")).round(2).to_string())
print(f"\\n워크플로우 전체 성공률 {dw['성공'].mean():.2f} — 경로에 올라타는 질문(주문번호 유형)만 풀린다")
''')

b.md('''
### 셀 9 · 과제 D 를 에이전트로 — 유형별 성공률 비교
- 같은 18개 질문을 도구 네 개(`lookup_order` · `search_customer` · `get_policy` · `calc`)와 루프로 푼다
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
- 질문마다 모델이 **스스로** 도구와 순서를 정한다. 아래 표의 유형별 성공률을 워크플로우와 나란히 본다
''')
b.code('''
# 셀 9 · 과제 D 를 에이전트로
TD = tools_D(orders, cust)
TRAJ_D = []
for q, t in zip(QD, pmap(lambda q: ag(SYSTEM_D, q["q"], TD), QD)):     # 질문 18개는 서로 독립 → 겹쳐 돌린다
    TRAJ_D.append((q, t))
    RD.append({"유형": q["유형"], "방식": "에이전트", "성공": score_D(q, t.final)["성공"], "호출": t.trace.llm_calls, "토큰": t.trace.tokens, "지연": t.trace.latency})
    add("D 개방형", "에이전트", score_D(q, t.final)["성공"], t.trace, 호출도구=len(t.calls))
dd = pd.DataFrame(RD)
piv = dd.pivot_table(index="유형", columns="방식", values="성공", aggfunc="mean").round(2)
cost = dd.groupby("방식").agg(성공률=("성공", "mean"), 호출=("호출", "mean"), 토큰=("토큰", "mean"), 지연=("지연", "mean")).round(2)
print(piv.to_string()); print(); print(cost.to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
x = np.arange(len(piv));
for k, (meth, c) in enumerate([("워크플로우", COLOR["code"]), ("에이전트", COLOR["llm"])]):
    ax[0].bar(x + (k - 0.5) * 0.38, piv[meth].values, 0.38, color=c, ec="#444", lw=0.8, label=meth)
ax[0].set_xticks(x); ax[0].set_xticklabels(piv.index, fontsize=8, rotation=15); ax[0].set_title("과제 D — 질문 유형별 성공률", fontsize=10, loc="left"); ax[0].legend(fontsize=8, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.22))
bars(ax[1], ["워크플로우", "에이전트"], [cost.loc["워크플로우", "토큰"], cost.loc["에이전트", "토큰"]], colors=[COLOR["code"], COLOR["llm"]], title="질문당 토큰", fmt="{:.0f}")
plt.tight_layout(); plt.show()
q_show, t_show = next((q, t) for q, t in TRAJ_D if q["유형"] == "이름+상품")
draw_trajectory(t_show, title="에이전트 궤적 한 건 : " + q_show["q"][:30], show_obs=False); plt.show()
''')

b.md('''
### 셀 10 · 같은 입력, 다른 궤적 — 비결정성
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
- 질문 하나를 **온도 0.7 로 여러 번** 돌린다(`sample` 번호를 달리 해 표본마다 따로 뽑음)
- 에이전트는 **도구 순서 · 호출 횟수가 갈라질 수 있다**. 워크플로우는 같은 입력이면 **같은 경로**를 간다(분기가 없다)
- 가로 = 몇 번째 행동, 세로 = 어떤 도구(단계). 선이 굵을수록 그 길로 간 횟수가 많다
''')
b.code('''
# 셀 10 · 같은 입력 · 다른 궤적
qn = next(q for q in QD if q["유형"] == "이름+상품")
N = CFG["비결정성_횟수"]
ag_traj = pmap(lambda i: ag(SYSTEM_D, qn["q"], TD, temperature=CFG["온도"], sample=i), range(N))          # 표본 번호별로 서로 독립
wf_paths = [r[2] for r in pmap(lambda i: wf_D(qn, temperature=CFG["온도"], sample=i), range(N))]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2))
draw_trajectory_flow(wf_paths, ax=a1, title="워크플로우")
draw_trajectory_flow(ag_traj, ax=a2, title="에이전트")
plt.tight_layout(); plt.show()
sig = pd.Series([t.signature for t in ag_traj]).value_counts()
print(f"에이전트의 서로 다른 궤적 {len(sig)}가지 / {N}회, 성공 {sum(score_D(qn, t.final)['성공'] for t in ag_traj)}회 — 워크플로우의 서로 다른 경로 {len(set(wf_paths))}가지")
print(sig.head(5).to_string())
''')

b.md('''
### 셀 11 · 판정표 · 결과 저장
- 과제마다 **경로를 코드로 그릴 수 있는가**의 답과, 실제로 측정한 성공률 · 비용을 나란히 놓는다
- 「실측이 판정과 맞는가」 열은 강의의 판정 기준(그릴 수 있으면 워크플로우가 싸고 안정적 · 없으면 에이전트)이 이번 측정에서 성립했는지를 **규칙으로** 적은 것이다. 어긋나면 어긋난 그대로 보인다
''')
b.code('''
# 셀 11 · 판정표 · 결과 저장
df = pd.DataFrame(ROWS)
g = df.groupby(["과제", "방식"]).agg(성공률=("성공", "mean"), 호출=("호출", "mean"), 토큰=("토큰", "mean"), 지연=("지연", "mean")).round(3)
rows = []
for t_, drawable in [("A 체이닝", "예"), ("B 라우팅", "예"), ("C 병렬화", "예"), ("D 개방형", "아니오")]:
    w, a = g.loc[(t_, "워크플로우")], g.loc[(t_, "에이전트")]
    ok = (w["성공률"] >= a["성공률"] - 0.05 and w["토큰"] < a["토큰"]) if drawable == "예" else (a["성공률"] > w["성공률"])
    rows.append({"과제": t_, "경로를 코드로 그릴 수 있나": drawable, "워크플로우 성공률": w["성공률"], "에이전트 성공률": a["성공률"],
                 "토큰 비(에이전트/워크플로우)": round(a["토큰"] / max(w["토큰"], 1), 2), "호출 비": round(a["호출"] / max(w["호출"], 1), 2),
                 "강의의 판정": "워크플로우" if drawable == "예" else "에이전트", "실측이 판정과 맞는가": "맞음" if ok else "어긋남"})
J = pd.DataFrame(rows)
print(J.to_string(index=False))
RES = {"설정": {"모델": llm.model, **CFG}, "과제별 요약": {f"{k[0]}|{k[1]}": {c: float(v) for c, v in r.items()} for k, r in g.iterrows()},
       "판정표": J.to_dict("records"), "D 유형별 성공률": piv.to_dict(), "비결정성": {"에이전트 서로 다른 궤적 수": int(len(sig)), "워크플로우 서로 다른 경로 수": len(set(wf_paths)), "횟수": N}}
save_measured("02", RES)
''')

b.md('''
### 확인 질문
1. 셀 5 에서 「직렬로 돌았다면의 합」과 「겹쳐 돈 시간」은 얼마나 달랐는가? 병렬화가 지연을 줄이는 조건은 무엇이고, 이 실습의 지연 계산이 놓치는 것(서버의 동시 처리)은 무엇인가?
2. 셀 7 에서 과제 A · B · C 는 워크플로우와 에이전트 중 어느 쪽이 더 싸고(토큰 · 호출) 더 안정적이었는가? 에이전트가 같은 일을 하는 데 호출을 더 쓴다면 그 추가 호출은 무엇을 위한 것이었는가?
3. 셀 8 · 9 에서 워크플로우가 풀지 못한 질문 유형을 풀려면 코드에 어떤 분기를 더 만들어야 하는가? 질문 유형이 열 가지로 늘어나면 워크플로우와 에이전트 중 어느 쪽의 설계 비용이 더 빨리 커지는가? 셀 11 의 「어긋남」이 있다면 그 이유를 설명해 보자.
''')

b.save()
