"""06 번 시험용 가짜 LLM — 코딩 에이전트(run_tests · grep · read_file · edit_file). 서버 없이 코드 경로만 확인한다(숫자는 의미 없음).

샌드박스 · 도구 · 시험 실행 · 숨은 확인은 노트북의 진짜 코드가 돈다. 가짜인 것은 모델의 선택뿐이다.
궤적마다(질문 · 표본 번호 · 도구 세트로 정해지는 난수) 다음 중 하나의 길을 간다 :
  correct   정답 패치           unrelated 엉뚱한 곳을 고침(시험이 깨짐 → 되돌리고 정답으로 고침 = 반복 경로)
  symptom   증상만 가린 수정(보이는 시험은 통과, 숨은 확인은 부분)   big  정답 패치 + 불필요한 주석(바꾼 줄이 커짐)
여기에 확률로 old 를 틀리게 쓴 시도(오류 관찰), 관련 없는 파일 읽기가 끼어든다. 온도 0 의 한 궤적은 항상 깔끔한 정답 경로다.
"""
import hashlib
import json
import random

from dry_common import make_dry_llm

# 노트북 셀 2 의 BUGS 와 같은 글이어야 한다(다르면 edit_file 이 오류를 돌려줘서 바로 드러난다)
PATCH = {
    "게스트": dict(file="payment/checkout.py", grep="get_user(",
                 bad="    uid = user.id\n    grade = user.grade\n",
                 good="    uid = user.id if user else None\n    grade = user.grade if user else \"guest\"\n",
                 symptom="    uid = user.id if user else 0\n    grade = user.grade if user else \"guest\"\n",
                 note="게스트(user 가 None)일 때 id 와 등급을 기본값으로 처리"),
    "경계값": dict(file="payment/discount.py", grep="bulk_rate",
                 bad="    return 3 if qty > BULK_MIN else 0\n", good="    return 3 if qty >= BULK_MIN else 0\n", symptom=None,
                 note="수량 할인 비교를 > 에서 >= 로 바꿈"),
    "반올림": dict(file="payment/discount.py", grep="apply_discount",
                 bad="    return amount * (100 - rate) // 100\n", good="    return (amount * (100 - rate) + 50) // 100\n",
                 symptom="    return -((-amount * (100 - rate)) // 100)\n", note="내림을 반올림으로 바꿈"),
}
UNRELATED = ("payment/discount.py", '"silver": 5', '"silver": 6')
BIG_COMMENT = "".join(f"    # (시험) 수정 메모 {i}\n" for i in range(1, 6))
EXTRA_READ = "payment/users.py"


def install(mod):
    Base = make_dry_llm(mod)
    Reply = mod.Reply

    def bug_of(user):
        if "AttributeError" in user:
            return "게스트"
        if "반올림" in user:
            return "반올림"
        return "경계값"

    def call(name, args, thought, p):
        a = json.dumps(args, ensure_ascii=False)
        return Reply(content=thought, tool_calls=[{"id": f"call_{p}", "name": name, "arguments": args, "raw": a, "valid": True}],
                     prompt_tokens=p, completion_tokens=len(a) // 2 + 20, latency=0.4 + 0.0004 * p, finish_reason="tool_calls")

    def say(text, p):
        return Reply(content=text, prompt_tokens=p, completion_tokens=len(text) // 2 + 1, latency=0.3 + 0.0004 * p + 0.002 * len(text), finish_reason="stop")

    class Dry06(Base):
        def __repr__(self):
            return "Dry06(서버 없이 시험 중 — 숫자는 의미 없음)"

        def chat(self, messages, tools=None, temperature=0.0, seed=0, max_tokens=512, sample=0, **extra):
            names = {t["function"]["name"] for t in (tools or [])}
            if not {"edit_file", "read_file"} <= names:
                return super().chat(messages, tools=tools, temperature=temperature, seed=seed, max_tokens=max_tokens, sample=sample, **extra)
            self.n_live += 1
            tests = "run_tests" in names
            user = next(str(m["content"]) for m in messages if m["role"] == "user")
            p = sum(len(str(m.get("content") or "")) for m in messages) // 2 + 1
            bug = bug_of(user)
            P = PATCH[bug]
            rng = random.Random(int(hashlib.sha1(f"{user}|{sample}|{tests}|{temperature}".encode("utf-8")).hexdigest(), 16))
            if temperature == 0:
                kind, typo, extra_read = "correct", False, False
            else:
                r = rng.random()
                kind = "correct" if r < 0.45 else "unrelated" if r < 0.70 else "symptom" if r < 0.88 else "big"
                typo, extra_read = rng.random() < 0.25, rng.random() < 0.4
                if kind == "symptom" and not P["symptom"]:
                    kind = "correct"

            calls = [(m["tool_calls"][0]["function"]["name"], json.loads(m["tool_calls"][0]["function"]["arguments"] or "{}")) for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
            obs = [m["content"] for m in messages if m["role"] == "tool"]
            k = len(calls)

            def edit(old, new, thought, path=None):
                return ("edit_file", {"path": path or P["file"], "old": old, "new": new}, thought)

            seq = []
            if tests:
                seq.append(("run_tests", {}, "재현부터. 어떤 시험이 깨지는지 확인한다."))
            seq.append(("grep", {"pattern": P["grep"]}, "관련 코드를 찾는다."))
            seq.append(("read_file", {"path": P["file"]}, "해당 파일을 읽고 원인을 확인한다."))
            if extra_read:
                seq.append(("read_file", {"path": EXTRA_READ}, "연관된 파일도 확인한다."))
            fix_thought = "원인에 해당하는 줄만 최소한으로 고친다."
            if kind == "unrelated":
                seq.append(edit(UNRELATED[1], UNRELATED[2], "원인이 이 근처라고 보고 수정한다.", UNRELATED[0]))
            else:
                if typo:
                    seq.append(edit(P["bad"].replace("user", "usr", 1), P["good"], fix_thought))      # 글자를 틀리게 쓴 시도 → 오류 관찰(파일은 그대로)
                new = {"correct": P["good"], "symptom": P["symptom"], "big": BIG_COMMENT + P["good"]}[kind]
                seq.append(edit(P["bad"], new, fix_thought))
            if tests:
                seq.append(("run_tests", {}, "고친 뒤 시험으로 확인한다."))
            if k < len(seq):
                return call(*seq[k], p)

            done = "고쳤다. " + P["note"] + "했다." + (" 시험이 통과했다." if tests else "")
            if not tests:
                return say(done, p)
            runs = [o for (n, _), o in zip(calls, obs) if n == "run_tests"]
            if runs and runs[-1].startswith("PASS"):
                return say(done, p)
            # 시험이 깨졌다 → 방금 고친 곳을 다시 읽고, 엉뚱한 수정은 되돌리고, 정답으로 고친 뒤 다시 시험한다
            rep = [("read_file", {"path": P["file"]}, "시험이 아직 깨진다. 실패 로그를 다시 읽고 파일을 확인한다.")]
            if kind == "unrelated":
                rep.append(edit(UNRELATED[2], UNRELATED[1], "엉뚱한 곳을 건드렸다. 되돌린다.", UNRELATED[0]))
            rep += [edit(P["bad"], P["good"], fix_thought), ("run_tests", {}, "다시 시험한다.")]
            j = k - len(seq)
            if j < len(rep):
                return call(*rep[j], p)
            return say("더 고치지 못했다.", p)

    mod.LLM = Dry06
