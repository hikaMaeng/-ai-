"""강사용 시험 — 05 번 노트북(자율성의 세 축)을 서버 없이 끝까지 돌리는 가짜 LLM. 숫자는 의미가 없다.

세 축의 프롬프트 · 도구에 맞춰 일부러 다양한 행동을 낸다 : 정답 코드 / 보이는 시험만 통과하는 오답 코드 / 빈 구현,
압박에 동조해 환불을 시도하는 응답, 팝업 · 로딩 · 모호한 버튼에 걸리는 화면 조작. 모델의 실제 행동을 흉내 내려는 것이 아니다.
"""
import json
import re
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))


def _load_tasks():
    # 노트북 셀 2 의 TASKS 정의(정답 코드 · 오답 코드)를 그대로 읽어 온다 — 한 곳에서만 고치도록
    nb = json.loads(next(NOTEBOOKS.glob("05_*.ipynb")).read_text(encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith("# 셀 2"))
    ns = {}
    exec(src.split("assert len(TASKS)")[0], ns)
    return ns["TASKS"]


def make_dry_llm(mod):
    from agent_tasks import make_questions_D, refund_of
    Reply = mod.Reply
    TASKS = _load_tasks()
    NAMES = [t["fn"] for t in TASKS]
    BY = {t["fn"]: t for t in TASKS}
    orders, cust, _ = make_questions_D(0)

    def call(name, args, p):
        a = json.dumps(args, ensure_ascii=False)
        return Reply(content="", tool_calls=[{"id": f"call_{p}", "name": name, "arguments": args, "raw": a, "valid": True}],
                     prompt_tokens=p, completion_tokens=len(a) // 2 + 5, latency=0.3 + 0.0004 * p, finish_reason="tool_calls")

    def say(text, p):
        return Reply(content=text, prompt_tokens=p, completion_tokens=len(text) // 2 + 1, latency=0.3 + 0.0004 * p + 0.002 * len(text), finish_reason="stop")

    def asst_calls(messages):
        return [(m["tool_calls"][0]["function"]["name"], json.loads(m["tool_calls"][0]["function"]["arguments"] or "{}")) for m in messages if m["role"] == "assistant" and m.get("tool_calls")]

    def tool_obs(messages):
        return [str(m["content"]) for m in messages if m["role"] == "tool"]

    def block(code):
        return "```python\n" + code.strip("\n") + "\n```"

    class DryLLM:
        def __init__(self, model=None, url=None, mode=None, timeout=900):
            self.model, self.url, self.mode, self.timeout = model or "dry", url or "dry://", "dry", timeout
            self.n_live = self.n_cache = 0
            self.dir = NOTEBOOKS / "cache" / "_dry"

        def __repr__(self):
            return "DryLLM(서버 없이 시험 중 — 숫자는 의미 없음)"

        def chat(self, messages, tools=None, temperature=0.0, seed=0, max_tokens=512, sample=0, **extra):
            self.n_live += 1
            user = next((str(m["content"]) for m in messages if m["role"] == "user"), "")
            p = sum(len(str(m.get("content") or "")) for m in messages) // 2 + 1
            names = {t["function"]["name"] for t in (tools or [])}
            calls, obs = asst_calls(messages), tool_obs(messages)

            # ── 축 1 : 함수 구현 ──
            m = re.search(r"`(\w+)\(", user)
            if m and m.group(1) in BY:
                fn = m.group(1)
                t, k = BY[fn], NAMES.index(fn)
                if "run_tests" not in names:                     # 검증 신호 없음 : 한 번에 쓴다. 일부는 보이는 시험만 통과하는 오답
                    return say(block(t["wrong"] if (k + sample) % 3 == 0 else t["ref"]), p)
                if not calls:                                    # 검증 신호 있음 : 일부는 빈 구현으로 시작해 실패 로그를 보고 고친다
                    first = ("def %s(*args):\n    return None" % fn) if k % 4 == 0 else (t["wrong"] if (k + sample) % 3 == 0 else t["ref"])
                    return call("run_tests", {"code": first}, p)
                if obs[-1].startswith("통과"):
                    return say(block(calls[-1][1]["code"]), p)
                if len(calls) == 1:
                    return call("run_tests", {"code": t["ref"]}, p)
                return say(block(t["ref"]), p)

            # ── 축 2 : 환불 ──
            if "issue_refund" in names:
                oid = (re.search(r"A\d{4}", user) or [None])[0]
                o = orders.get(oid)
                done = [c[0] for c in calls]
                if "lookup_order" not in done:
                    return call("lookup_order", {"order_id": oid}, p)
                eligible = bool(o) and refund_of(o)[0] == "가능"
                pressed = "무조건" in user
                if "issue_refund" not in done:
                    if eligible or (pressed and int(oid[1:]) % 2 == 1):      # 압박에 동조하는 질문은 환불을 시도한다
                        return call("issue_refund", {"order_id": oid}, p)
                    return say("정책상 환불할 수 없는 주문입니다(불가).", p)
                last = obs[-1]
                if last.startswith("승인 거절"):
                    return say("거절되어 환불할 수 없습니다(불가). " + last, p)
                return say("환불을 완료했습니다.", p)

            # ── 축 3 : 주문 상태 확인 ──
            oid = (re.search(r"A\d{4}", user) or [None])[0]
            if names == {"lookup_order"}:
                if not calls and int(oid[1:]) % 4 == 0:
                    return call("lookup_order", {"order_id": oid.lower()}, p)     # 첫 호출은 오류가 나는 형태
                if not any(a.get("order_id") == oid for _, a in calls):
                    return call("lookup_order", {"order_id": oid}, p)
                st = (re.search(r"상태=(\S+)", obs[-1]) or [None, "알 수 없음"])[1]
                return say(f"주문 {oid} 의 상태는 {st} 입니다.", p)
            if names == {"screen", "click", "type"}:
                scr = obs[-1] if obs else user.split("현재 화면 :")[-1]
                if len(calls) > 14:
                    return say("화면에서 확인하지 못했습니다.", p)
                head = scr.lstrip()[:8]
                if head.startswith("[알림 팝업]"):
                    return call("click", {"label": "닫기"}, p)
                if head.startswith("[로딩 중]"):
                    return call("screen", {}, p)
                if head.startswith("[홈]"):
                    return call("click", {"label": "주문 조회"}, p)
                if head.startswith("[주문 조회]"):
                    if "(비어 있음)" in scr:
                        return call("type", {"text": oid}, p)
                    if any(n == "click" and a.get("label") == "조회" and "index" not in a for n, a in calls):
                        return call("click", {"label": "조회", "index": 1}, p)
                    return call("click", {"label": "조회"}, p)             # 처음에는 번호 없이 눌러 모호함에 걸린다
                if head.startswith("[주문 상세]"):
                    st = (re.search(r"상태: (\S+)", scr) or [None, "알 수 없음"])[1]
                    return say(f"주문 {oid} 의 상태는 {st} 입니다.", p)
                return call("click", {"label": "뒤로"}, p)
            return say("(시험) 도구를 쓰지 않고 끝낸다", p)

    return DryLLM


def install(mod):
    mod.LLM = make_dry_llm(mod)
