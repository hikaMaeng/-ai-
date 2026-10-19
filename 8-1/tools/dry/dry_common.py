"""강사용 시험 — 서버 없이 노트북의 코드 경로를 끝까지 돌리는 가짜 LLM(AGENT_DRY=NN). 성공률 · 토큰 · 지연 숫자는 의미가 없다.

노트북이 쓰는 프롬프트의 말투(「분류」「추출」「개조식」「영어로」…)를 보고 그럴듯한 답을 낸다. 진짜 모델의 행동을 흉내 내려는 것이 아니라,
셀이 오류 없이 끝나고 그림이 그려지는지만 확인한다.
"""
import json
import re
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))


def make_dry_llm(mod):
    from agent_tasks import (B_TOOL_FOR, REQUIRE_B, _KEY_B, make_docs_A, make_inquiries_B, make_meetings_C, make_questions_D, plan_D, refund_of)
    Reply = mod.Reply
    docs, inqs, meets = make_docs_A(8), make_inquiries_B(), make_meetings_C()
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
        return Reply(content="", tool_calls=[{"id": f"call_{p}", "name": name, "arguments": args, "raw": a, "valid": True}], prompt_tokens=p, completion_tokens=len(a) // 2 + 5, latency=0.3 + 0.0004 * p, finish_reason="tool_calls")

    def say(text, p):
        return Reply(content=text, prompt_tokens=p, completion_tokens=len(text) // 2 + 1, latency=0.3 + 0.0004 * p + 0.002 * len(text), finish_reason="stop")

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
            p = len(text) // 2 + 1
            if tools:
                names = {t["function"]["name"] for t in tools}
                if "lookup_order" in names:
                    q = next((q for q in QS if q["q"] == user), None)
                    if q is None:
                        return say("알 수 없음", p)
                    steps, final = plan_D(q, orders, cust)
                    k = progress(messages, steps)
                    return call(*steps[k], p) if k < len(steps) else say(final, p)
                if "read_section" in names:
                    m = re.search(r"C\d+", user)
                    mt = next(x for x in meets if x["id"] == m.group(0)) if m else meets[0]
                    steps = [("section_count", {})] + [("read_section", {"n": i + 1}) for i in range(4)]
                    k = progress(messages, steps)
                    return call(*steps[k], p) if k < len(steps) else say(" ".join(mt["sections"]), p)
                if "refund_policy" in names:
                    it = next((x for x in inqs if x["text"] in user), inqs[0])
                    steps = [(B_TOOL_FOR[it["label"]], {})]
                    k = progress(messages, steps)
                    return call(*steps[k], p) if k < len(steps) else say(f"안내드립니다. {_KEY_B[it['label']]} {it.get('oid') or ''}", p)
                if "get_document" in names:
                    m = re.search(r"A\d+", user)
                    d = next((x for x in docs if x["id"] == m.group(0)), docs[0]) if m else docs[0]
                    steps = [("get_document", {"doc_id": d["id"]})]
                    k = progress(messages, steps)
                    return call(*steps[k], p) if k < len(steps) else say("English summary: " + " ".join(str(v) for v in d["facts"].values()) + " won", p)
                return say("(시험) 도구를 쓰지 않고 끝낸다", p)
            # 도구 없는 호출(워크플로우의 노드 · 단발 호출)
            if "분류" in user:
                it = next((x for x in inqs if x["text"] in user), None)
                return say(it["label"] if it else "일반", p)
            if "주문번호를 추출" in user:
                m = re.search(r"A\d{4}", user.split("질문 :")[-1])
                return say(m.group(0) if m else "없음", p)
            if "정책에 따라" in user:
                m = re.search(r"id=(A\d+)", user)
                o = orders.get(m.group(1)) if m else None
                if not o:
                    return say("확인할 수 없음", p)
                j, amt = refund_of(o)
                return say("환불 불가" if j == "불가" else f"환불 가능, 환불 금액 {amt:,}원", p)
            if "구간" in user and "요약" in user:
                return say(user.split("구간 원문:")[-1].strip(), p)
            if "핵심" in user and "추출" in user:
                return say(user.split("공지:")[-1].strip(), p)
            if "개조식" in user:
                return say("- " + user.split("원문:")[-1].strip(), p)
            if "영어" in user:
                return say("Summary: " + " ".join(re.findall(r"[\d,]+", user.split("원문:")[-1])) + " won, apply by deadline", p)
            if "응답" in user or "답변" in user:
                it = next((x for x in inqs if x["text"] in user), None)
                return say(f"안내드립니다. {_KEY_B[it['label']] if it else ''} {(it or {}).get('oid') or ''}", p)
            return say("주문 정보를 확인할 수 없어 알 수 없습니다.", p)

    return DryLLM
