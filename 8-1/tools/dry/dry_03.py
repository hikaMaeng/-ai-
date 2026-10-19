"""실습 03(행동 분포 들여다보기) 전용 가짜 LLM — 서버 없이 노트북을 끝까지 돌려 코드 경로와 그림만 확인한다. 숫자는 의미가 없다.

첫 행동은 (질문 · 도구 설명 · 시스템지시 · temperature) 에 따라 정해 둔 분포에서 뽑는다. 뽑는 난수는 (질문 · 도구 · temperature · seed · sample · 스텝)의
해시로 정해져서, 같은 입력 · 같은 seed 면 같은 결과가 나온다(= seed 를 지키는 서버를 흉내 냄). temperature 0 은 최빈 행동 하나만 낸다.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

NONE_ = "(글 답)"


def make_dry_llm(mod):
    from agent_tasks import make_questions_D, plan_D, refund_of
    Reply = mod.Reply
    orders, cust, QS = make_questions_D(0)

    def asst_calls(messages):
        return [m["tool_calls"][0]["function"] for m in messages if m["role"] == "assistant" and m.get("tool_calls")]

    def progress(messages, steps):
        k = 0
        for c in asst_calls(messages):
            if k < len(steps) and c["name"] == steps[k][0] and json.loads(c["arguments"] or "{}") == steps[k][1]:
                k += 1
        return k

    def call(name, args, p):
        a = json.dumps(args, ensure_ascii=False)
        return Reply(content="", tool_calls=[{"id": f"call_{p}", "name": name, "arguments": args, "raw": a, "valid": True}],
                     prompt_tokens=p, completion_tokens=len(a) // 2 + 5, latency=0.3 + 0.0004 * p, finish_reason="tool_calls")

    def say(text, p):
        return Reply(content=text, prompt_tokens=p, completion_tokens=len(text) // 2 + 1, latency=0.3 + 0.0004 * p + 0.002 * len(text), finish_reason="stop")

    def pick(rng, base, T):
        """base 는 temperature 0.7 에서의 분포로 둔다. T 가 낮으면 뾰족하게, 높으면 평평하게. T=0 은 최빈 하나"""
        if T <= 0.01:
            return max(base, key=base.get)
        w = {k: v ** (0.7 / T) for k, v in base.items() if v > 0}
        return rng.choices(list(w), weights=list(w.values()))[0]

    class DryLLM:
        def __init__(self, model=None, url=None, mode=None, timeout=900):
            self.model, self.url, self.mode, self.timeout = model or "dry", url or "dry://", "dry", timeout
            self.n_live = self.n_cache = 0
            self.dir = NOTEBOOKS / "cache" / "_dry"

        def __repr__(self):
            return "DryLLM(서버 없이 시험 중 — 숫자는 의미 없음)"

        def chat(self, messages, tools=None, temperature=0.0, seed=0, max_tokens=512, sample=0, **extra):
            self.n_live += 1
            text = "\n".join(str(m.get("content") or "") for m in messages)
            user = next((str(m["content"]) for m in messages if m["role"] == "user"), "")
            system = next((str(m["content"]) for m in messages if m["role"] == "system"), "")
            p = len(text) // 2 + 1
            if not tools:
                return say("(시험) 도구 없이 답한다", p)
            names = [t["function"]["name"] for t in tools]
            desc = {t["function"]["name"]: t["function"].get("description", "") for t in tools}
            prev = asst_calls(messages)
            key = json.dumps([user, system, desc, temperature, seed, sample, len(prev)], ensure_ascii=False, sort_keys=True)
            rng = random.Random(int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16))

            # ── 환불 요청(위험 도구가 있는 질문) ──
            m = re.search(r"주문 (A\d+) 환불해 주세요", user)
            if m and "issue_refund" in names:
                o = orders.get(m.group(1))
                ok = bool(o) and refund_of(o)[0] == "가능"
                if prev:
                    return say("확인한 결과를 안내드립니다.", p)
                ban = "issue_refund 를 부르지 않는다" in system
                pi = 0.08 if ban else (0.5 if ok else 0.42)
                rest = 1 - pi
                base = {"issue_refund": pi, "lookup_order": 0.6 * rest, "get_policy": 0.2 * rest, "search_customer": 0.05 * rest, "calc": 0.05 * rest, NONE_: 0.1 * rest}
                c = pick(rng, base, temperature)
                if c == NONE_:
                    return say("환불 처리를 도와드리겠습니다.", p)
                return call(c, {"order_id": m.group(1)} if c in ("lookup_order", "issue_refund") else ({"name": "김영희"} if c == "search_customer" else ({"expression": "0"} if c == "calc" else {})), p)

            # ── 환불 가능 여부 질문(과제 D) ──
            q = next((x for x in QS if x["q"] == user), None)
            if q is None:
                return say("알 수 없음", p)
            steps, final = plan_D(q, orders, cust)
            first = steps[0][0]
            if not prev:
                other = "search_customer" if first == "lookup_order" else "lookup_order"
                same = desc.get("lookup_order") == desc.get("search_customer")
                vague = "데이터를 가져온다" in desc.get("lookup_order", "")
                if vague:
                    b = {first: 0.42, other: 0.28, "get_policy": 0.14, "calc": 0.05, NONE_: 0.11}
                elif same:
                    b = {first: 0.47, other: 0.44, "get_policy": 0.05, "calc": 0.01, NONE_: 0.03}
                else:
                    b = {first: 0.80, other: 0.08, "get_policy": 0.08, "calc": 0.01, NONE_: 0.03}
                b = {k: v for k, v in b.items() if k == NONE_ or k in names}
                c = pick(rng, b, temperature)
                if c == NONE_:
                    return say("주문 정보를 확인할 수 없어 알 수 없습니다.", p)
                if c == first:
                    return call(*steps[0], p)
                if c == "lookup_order":
                    return call("lookup_order", {"order_id": q["oid"] or "A1001"}, p)
                if c == "search_customer":
                    return call("search_customer", {"name": orders[q["oid"]]["고객"] if q["oid"] in orders else "홍길동"}, p)
                return call(c, {"expression": "0"} if c == "calc" else {}, p)
            # 두 번째 스텝부터 : 정답 경로를 따르되 temperature 가 높을수록 가끔 엉뚱한 호출
            k = progress(messages, steps)
            if rng.random() < 0.03 + 0.08 * temperature and k < len(steps):
                return call("get_policy", {}, p)
            return call(*steps[k], p) if k < len(steps) else say(final, p)

    return DryLLM


def install(mod):
    mod.LLM = make_dry_llm(mod)
