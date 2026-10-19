"""04 번 시험용 가짜 LLM — 연쇄 조회(get 한 개) · 과제 D 첫 행동 · 모델 이름별 오류율. 서버 없이 코드 경로만 확인한다(숫자는 의미 없음).

- 모델 이름마다 스텝 오류율이 다르다(작은 모델일수록 큼) : 깨진 인자 · 없는 키 · 틀린 도구 이름 · 이른 종료
- qwen3.5-9b-mtp 는 replay 모드에서 CacheMiss 를 내서 「캐시가 없는 모델은 표에서 빠진다」 경로를 시험한다
"""
import hashlib
import random
import re

from dry_common import make_dry_llm

EPS = {"qwen3.5-0.8b-mtp": 0.25, "qwen3.5-2b-mtp": 0.12, "qwen3.5-4b-mtp": 0.05, "qwen3.5-9b-mtp": 0.02}


def install(mod):
    Base = make_dry_llm(mod)
    Reply = mod.Reply

    class Dry04(Base):
        def __init__(self, model=None, url=None, mode=None, timeout=900):
            super().__init__(model or "qwen3.5-4b-mtp", url, mode, timeout)
            self.req_mode = mode or "auto"

        def __repr__(self):
            return f"Dry04(model={self.model!r}, 서버 없이 시험 중 — 숫자는 의미 없음)"

        def chat(self, messages, tools=None, temperature=0.0, seed=0, max_tokens=512, sample=0, **extra):
            if self.req_mode == "replay" and self.model == "qwen3.5-9b-mtp":
                raise mod.CacheMiss("(시험) 이 모델의 캐시가 없다")
            names = {t["function"]["name"] for t in (tools or [])}
            text = "\n".join(str(m.get("content") or "") for m in messages)
            rng = random.Random(int(hashlib.sha1(f"{self.model}|{text}|{sample}|{len(messages)}".encode("utf-8")).hexdigest(), 16))
            eps = EPS.get(self.model, 0.1)
            self.n_live += 1
            p = len(text) // 2 + 1
            if "get" in names:
                user = next((str(m["content"]) for m in messages if m["role"] == "user"), "")
                n = int(re.search(r"모두 ([0-9]+)번", user).group(1))
                obs = [m["content"] for m in messages if m["role"] == "tool"]
                if len(obs) >= n:
                    last = obs[-1]
                    return Reply(content=last.split(": ")[1] if last.startswith("값 :") else "모르겠다", prompt_tokens=p, completion_tokens=6, latency=0.3, finish_reason="stop")
                last = obs[-1] if obs else None
                key = "K0" if last is None else (last.split(": ")[1] if last.startswith("값 :") else "?")
                if rng.random() < eps:
                    kind = rng.choice(["깨진", "없는 키", "이른 종료"])
                    if kind == "깨진":
                        return Reply(tool_calls=[{"id": "c", "name": "get", "arguments": {}, "raw": "{key: ", "valid": False}], prompt_tokens=p, completion_tokens=8, latency=0.3, finish_reason="tool_calls")
                    if kind == "이른 종료":
                        return Reply(content="K0", prompt_tokens=p, completion_tokens=3, latency=0.3, finish_reason="stop")
                    key = "K" + str(rng.randint(100, 999))
                return Reply(tool_calls=[{"id": f"c{len(obs)}", "name": "get", "arguments": {"key": key}, "raw": '{"key": "%s"}' % key, "valid": True}],
                             prompt_tokens=p, completion_tokens=12, latency=0.3 + 0.0004 * p, finish_reason="tool_calls")
            r = super().chat(messages, tools=tools, temperature=temperature, seed=seed, max_tokens=max_tokens, sample=sample, **extra)
            if r.tool_calls and rng.random() < eps:
                c = dict(r.tool_calls[0])
                kind = rng.choice(["깨진", "이름", "인자 빠짐"])
                if kind == "깨진":
                    c.update(arguments={}, raw="{order_id: ", valid=False)
                elif kind == "이름":
                    c.update(name=c["name"] + "s")
                else:
                    c.update(arguments={}, raw="{}")
                r.tool_calls = [c]
            return r

    mod.LLM = Dry04
