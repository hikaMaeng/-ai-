"""LM Studio 서버 점검(강사용) — 실습을 돌리기 전에 서버 · 모델 · 도구 호출 · seed 재현 · logprobs 지원을 확인한다.

    python tools/check_lmstudio.py                                   # 기본 http://localhost:1234/v1, 로드된 첫 모델
    python tools/check_lmstudio.py --url http://localhost:12345/v1 --model qwen3.5-4b-mtp

표준 라이브러리만 쓴다(호스트 파이썬에 requests 가 없어도 돈다). 모델을 불러오거나 서버를 켜지 않는다 — 켜진 서버에 요청만 보낸다.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:1234/v1")
ap.add_argument("--model", default=None)
a = ap.parse_args()
URL = a.url.rstrip("/")


def post(path, body, timeout=300):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")), time.perf_counter() - t0


def get(path):
    with urllib.request.urlopen(URL + path, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


ok_all = True


def report(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


try:
    models = [m["id"] for m in get("/models")["data"]]
except Exception as e:
    print(f"[실패] 서버에 연결하지 못했다 : {URL} ({type(e).__name__}) — LM Studio 의 Developer 탭에서 서버를 켜고 포트를 확인한다")
    sys.exit(1)
report("서버 연결", True, f"{URL} · 로드된 모델 {len(models)}개 : {models[:6]}")
if not models:
    print("로드된 모델이 없다 — LM Studio 에서 모델을 로드한 뒤 다시 돌린다(GPU 는 LM Studio 설정에서 RTX 4080 만 쓰도록 지정)")
    sys.exit(1)
model = a.model or models[0]
report("모델 이름", model in models, f"점검 대상 {model!r}")
base = {"model": model, "max_tokens": 200, "temperature": 0.0, "seed": 7}

try:
    j, dt = post("/chat/completions", {**base, "messages": [{"role": "user", "content": "한 문장으로 인사해라."}]})
    msg = j["choices"][0]["message"]
    report("일반 대화", bool(msg.get("content")), f"{dt:.1f}초 · 토큰 {j.get('usage', {}).get('total_tokens')} · 답 {str(msg.get('content'))[:40]!r}")
    report("생각 과정 분리", True, "reasoning_content 있음" if msg.get("reasoning_content") else "reasoning_content 없음(생각을 내지 않거나 본문에 섞임 — 본문의 <think> 는 실습 코드가 걷어낸다)")
except urllib.error.HTTPError as e:
    report("일반 대화", False, f"HTTP {e.code} {e.read().decode('utf-8', 'replace')[:200]}")
    sys.exit(1)

tool = {"type": "function", "function": {"name": "get_weather", "description": "도시의 현재 날씨를 조회한다.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
try:
    j, dt = post("/chat/completions", {**base, "tools": [tool], "messages": [{"role": "user", "content": "서울의 날씨를 알려 줘."}]})
    m = j["choices"][0]["message"]
    tc = m.get("tool_calls") or []
    args = json.loads(tc[0]["function"]["arguments"]) if tc else None
    report("도구 호출 생성", bool(tc) and tc[0]["function"]["name"] == "get_weather", f"{dt:.1f}초 · tool_calls={tc[0]['function'] if tc else None}")
    report("인자가 JSON 이고 스키마에 맞음", isinstance(args, dict) and "city" in args, f"arguments={args}")
except Exception as e:
    report("도구 호출 생성", False, f"{type(e).__name__} {str(e)[:160]} — 도구 호출을 지원하는 모델 · 채팅 템플릿인지 확인")

try:
    p = {**base, "temperature": 0.9, "seed": 123, "messages": [{"role": "user", "content": "1부터 20 사이의 무작위 정수 하나만 말해라."}], "max_tokens": 12}
    x = post("/chat/completions", p)[0]["choices"][0]["message"].get("content")
    y = post("/chat/completions", p)[0]["choices"][0]["message"].get("content")
    p2 = {**p, "seed": 124}
    z = post("/chat/completions", p2)[0]["choices"][0]["message"].get("content")
    report("seed 재현(같은 seed · 온도 0.9 → 같은 답)", x == y, f"seed 123 두 번 : {x!r} / {y!r} · seed 124 : {z!r}")
except Exception as e:
    report("seed 재현", False, f"{type(e).__name__} {str(e)[:120]}")

try:
    j, _ = post("/chat/completions", {**base, "logprobs": True, "top_logprobs": 3, "max_tokens": 5, "messages": [{"role": "user", "content": "하나만 말해라 : 예 또는 아니오"}]})
    lp = j["choices"][0].get("logprobs")
    report("logprobs 지원", bool(lp), "응답에 logprobs 있음 — 첫 토큰 확률을 직접 볼 수 있다" if lp else "응답에 logprobs 가 없다 — 03 번은 표본 빈도로 분포를 추정한다")
except Exception as e:
    report("logprobs 지원", False, f"{type(e).__name__} {str(e)[:120]} — 03 번은 표본 빈도로 분포를 추정한다")

print("\n전체 :", "통과" if ok_all else "실패 항목이 있다(위 설명 참고)")
