"""8-1 에이전트 실습 공통 모듈 — LM Studio 호출(응답 캐시) · 도구 루프 · 추적 · 그림 · 서버 없이 시험하는 가짜 정책.

노트북은 첫 셀에서 `from agent_lab import *` 로 읽는다. 이 파일은 「에이전트가 무엇인가」를 보여 주는 쪽이 아니라
호출 · 기록 · 그리기를 맡는 도구 상자다. 루프 자체(장표의 의사코드)는 01 번 노트북에서 직접 만들고, 이 파일의 run_agent 는 그 완성형이다.
"""
import hashlib
import json
import logging
import math
import os
import random
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"        # LM Studio 응답 캐시(강사 실행 결과를 함께 둔다 — 서버 없이 재생할 때 쓴다)
RESULT_DIR = HERE / "results"     # 노트북이 쓴 핵심 숫자 · 그림(.gitignore)

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic") if f in _have] or ["DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

# 색 : 종류별로 고정(그림이 바뀌어도 같은 뜻은 같은 색)
COLOR = {"io": "#E5E7EB", "code": "#DDE6F2", "llm": "#F6C9A0", "tool": "#D8EBD3", "agent": "#E8DFF3", "bad": "#F3C9C9"}
EDGE = {"io": "#6B7280", "code": "#3F6694", "llm": "#D9822B", "tool": "#3E7A3A", "agent": "#6A4C9C", "bad": "#B91C1C"}


# ───────────────────────── 공통 도구 ─────────────────────────
def seed_all(s):
    random.seed(s)
    np.random.seed(s)


def ms(xs, d=2):
    """평균±표준편차 문자열"""
    xs = np.asarray(list(xs), dtype=float)
    return f"{xs.mean():.{d}f}±{xs.std():.{d}f}"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def save_measured(name, d):
    """핵심 숫자를 results/measured_<name>.json 에 쓴다 — 실습 장표의 숫자는 여기서 옮긴다"""
    RESULT_DIR.mkdir(exist_ok=True)
    p = RESULT_DIR / f"measured_{name}.json"
    p.write_text(json.dumps(_jsonable(d), ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장 :", p)


# ───────────────────────── LLM 호출 ─────────────────────────
class LLMUnavailable(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


@dataclass
class Reply:
    """모델 한 번 호출의 결과. tool_calls 는 [{"id","name","arguments"(dict),"raw"(문자열),"valid"}]"""
    content: str = ""
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0           # 초. 캐시에서 읽었으면 처음 호출했을 때 잰 값(강사 PC)
    finish_reason: str = ""
    cached: bool = False
    reasoning: str = ""

    @property
    def tokens(self):
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self):
        """응답이 max_tokens 에서 잘렸는가 — 「생각」을 먼저 쓰는 모델은 생각에 토큰을 다 써서 답이 빈 채로 잘릴 수 있다"""
        return self.finish_reason == "length"


_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLM:
    """OpenAI 호환 chat/completions 호출 + 응답 캐시.

    mode : auto  = 캐시에 있으면 캐시, 없으면 서버를 부르고 캐시에 씀(기본)
           live  = 항상 서버를 부르고 캐시를 덮어씀(다시 측정할 때)
           replay= 캐시만 씀(서버가 없는 PC 에서 강사 실행 결과를 그대로 재생). 없으면 CacheMiss
    캐시 키 = (모델, 메시지, 도구, 온도, 시드, 최대 토큰, sample). sample 은 서버로 보내지 않고 키에만 들어간다 —
    같은 질문을 온도 > 0 으로 여러 번 뽑을 때 표본마다 다른 캐시 칸을 쓰게 한다.
    """

    def __init__(self, model=None, url=None, mode=None, timeout=900, min_tokens=None):
        self.url = (url or os.environ.get("LMSTUDIO_URL") or "http://host.docker.internal:1234/v1").rstrip("/")
        self.model = model or os.environ.get("AGENT_MODEL") or "qwen3.5-4b-mtp"
        self.mode = mode or os.environ.get("AGENT_MODE") or "auto"
        self.timeout = timeout
        # 생각(reasoning)을 먼저 쓰는 모델은 생각에 토큰을 다 써서 답이 빈 채로 잘릴 수 있다. 그런 모델이면 AGENT_MIN_TOKENS=4096 처럼 하한을 준다(느려진다)
        self.min_tokens = int(min_tokens if min_tokens is not None else (os.environ.get("AGENT_MIN_TOKENS") or 0))
        self._warned = False
        self.n_live = self.n_cache = 0
        self.dir = CACHE_DIR / re.sub(r"[^\w.\-]+", "_", self.model)

    def __repr__(self):
        return f"LLM(model={self.model!r}, url={self.url!r}, mode={self.mode!r})"

    def _path(self, payload, sample):
        key = json.dumps([payload, sample], sort_keys=True, ensure_ascii=False)
        return self.dir / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:20] + ".json")

    def _check(self, r):
        if r.truncated and not self._warned:
            self._warned = True
            print(f"[경고] 응답이 max_tokens 에서 잘렸다(모델 {self.model}) — 글 답이 비었거나 도구 호출이 끊겼을 수 있다. "
                  f"생각을 먼저 쓰는 모델이면 AGENT_MIN_TOKENS=4096 으로 하한을 주거나 생각을 쓰지 않는 모델을 쓴다(점검 : python tools/check_lmstudio.py)")
        return r

    def chat(self, messages, tools=None, temperature=0.0, seed=0, max_tokens=512, sample=0, **extra):
        max_tokens = max(max_tokens, self.min_tokens)
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "seed": seed, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
        payload.update(extra)
        path = self._path(payload, sample)
        if self.mode != "live" and path.exists():
            self.n_cache += 1
            r = Reply(**json.loads(path.read_text(encoding="utf-8"))["reply"])
            r.cached = True
            return self._check(r)
        if self.mode == "replay":
            raise CacheMiss(f"캐시에 없는 호출이다(replay 모드) : 모델 {self.model} · 마지막 메시지 {str(messages[-1].get('content'))[:60]!r}\n"
                            "  LM Studio 서버를 켜고 AGENT_MODE=auto 로 바꾸거나, 강사 캐시(notebooks/cache)를 같은 모델 이름으로 받는다")
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{self.url}/chat/completions", json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            raise LLMUnavailable(f"LM Studio 서버에 연결하지 못했다 : {self.url}\n"
                                 "  ① LM Studio 의 Developer 탭에서 서버를 켠다  ② 포트가 1234 가 아니면 루트 .env 의 LMSTUDIO_URL 을 맞춰 docker compose up -d 로 컨테이너를 다시 올린다\n"
                                 "  ③ 서버를 못 켜면 AGENT_MODE=replay 로 강사 실행 결과를 재생한다") from None
        dt = time.perf_counter() - t0
        if resp.status_code != 200:
            raise LLMUnavailable(f"LM Studio 가 오류를 돌려줬다({resp.status_code}) : {resp.text[:300]}\n  모델 {self.model!r} 이 로드되어 있는지, 도구 호출을 지원하는 모델인지 확인한다")
        j = resp.json()
        msg = j["choices"][0]["message"]
        raw_content = msg.get("content") or ""
        reasoning = (msg.get("reasoning_content") or "") + "".join(_THINK.findall(raw_content))
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            rawargs = fn.get("arguments", "")
            try:
                args = json.loads(rawargs) if isinstance(rawargs, str) and rawargs.strip() else (rawargs or {})
                valid = isinstance(args, dict)
            except Exception:
                args, valid = {}, False
            calls.append({"id": c.get("id") or f"call_{len(calls)}", "name": fn.get("name", ""), "arguments": args if valid else {}, "raw": rawargs if isinstance(rawargs, str) else json.dumps(rawargs, ensure_ascii=False), "valid": valid})
        u = j.get("usage") or {}
        r = Reply(content=_THINK.sub("", raw_content).strip(), tool_calls=calls, prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                  latency=dt, finish_reason=j["choices"][0].get("finish_reason", ""), cached=False, reasoning=reasoning)
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"payload": payload, "sample": sample, "reply": r.__dict__}, ensure_ascii=False), encoding="utf-8")
        self.n_live += 1
        return self._check(r)


# ───────────────────────── 서버 없이 시험하는 가짜 정책 ─────────────────────────
class ScriptedPolicy:
    """LLM 과 같은 chat() 모양의 가짜 정책 — 서버 없이 코드를 시험하고, 「행동이 확률적이면 긴 궤적에서 어떻게 되는가」를 이론과 맞춰 보는 데 쓴다.

    script(messages, tools, rng) 가 (「call」, 이름, 인자) 또는 (「final」, 글) 을 돌려준다(= 오류 없는 기준 정책).
    각 호출에서 확률 eps 로 corrupt(messages, tools, rng) 가 만든 틀린 행동으로 바뀐다(corrupt 가 없으면 아무 도구에 빈 인자).
    토큰 수 · 지연은 글자 수에서 정한 값이다(진짜 측정이 아님 — 그림 · 집계 코드를 시험하는 용도).
    """

    def __init__(self, script, eps=0.0, corrupt=None, seed=0, model="scripted"):
        self.script, self.eps, self.corrupt, self.model = script, eps, corrupt, model
        self.rng = random.Random(seed)
        self.mode = "scripted"
        self.n_live = self.n_cache = 0

    def __repr__(self):
        return f"ScriptedPolicy(eps={self.eps})"

    def chat(self, messages, tools=None, temperature=0.0, seed=0, max_tokens=512, sample=0, **extra):
        act = self.script(messages, tools or [], self.rng)
        if act[0] == "call" and self.rng.random() < self.eps:
            if self.corrupt:
                act = self.corrupt(messages, tools or [], self.rng)
            else:
                names = [t["function"]["name"] for t in (tools or [])]
                act = ("call", self.rng.choice(names), {}) if names else ("final", "")
        p = sum(len(str(m.get("content") or "")) for m in messages) // 2 + 1
        self.n_live += 1
        if act[0] == "call":
            args = json.dumps(act[2], ensure_ascii=False)
            return Reply(content="", tool_calls=[{"id": f"call_{self.n_live}", "name": act[1], "arguments": act[2], "raw": args, "valid": True}],
                         prompt_tokens=p, completion_tokens=len(args) // 2 + 5, latency=0.2 + 0.0005 * p, finish_reason="tool_calls")
        return Reply(content=act[1], prompt_tokens=p, completion_tokens=len(act[1]) // 2 + 1, latency=0.2 + 0.0005 * p + 0.002 * len(act[1]), finish_reason="stop")


# ───────────────────────── 추적 ─────────────────────────
@dataclass
class Span:
    name: str
    kind: str              # llm | tool | code
    stage: int             # 같은 단계의 구간은 동시에 돈 것(병렬). 단계가 다르면 차례로 돈 것
    lane: int = 0
    dur: float = 0.0       # 초(LLM 은 Reply.latency — 캐시면 처음 잰 값)
    tok_in: int = 0
    tok_out: int = 0
    meta: dict = field(default_factory=dict)


class Trace:
    """한 번의 실행에서 일어난 호출을 모은다. 지연 = 단계별 가장 긴 구간의 합(병렬이면 겹쳐 돈 것으로 침)"""

    def __init__(self):
        self.spans = []
        self._stage = -1

    def new_stage(self):
        self._stage += 1
        return self._stage

    def llm(self, client, name, messages, stage=None, lane=0, **kw):
        r = client.chat(messages, **kw)
        s = self.new_stage() if stage is None else stage
        self.spans.append(Span(name, "llm", s, lane, r.latency, r.prompt_tokens, r.completion_tokens, {"cached": r.cached}))
        return r

    def add(self, name, kind, dur=0.0, stage=None, lane=0, **meta):
        s = self.new_stage() if stage is None else stage
        self.spans.append(Span(name, kind, s, lane, dur, 0, 0, meta))

    def code(self, name, fn, *a, stage=None, lane=0, **kw):
        """코드 한 단계를 실행하고 시간을 기록"""
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.add(name, "code", time.perf_counter() - t0, stage, lane)
        return out

    @property
    def llm_calls(self):
        return sum(1 for s in self.spans if s.kind == "llm")

    @property
    def tokens(self):
        return sum(s.tok_in + s.tok_out for s in self.spans)

    @property
    def latency(self):
        by = {}
        for s in self.spans:
            by[s.stage] = max(by.get(s.stage, 0.0), s.dur)
        return sum(by.values())

    def stats(self):
        return {"LLM 호출": self.llm_calls, "토큰": self.tokens, "지연(초)": round(self.latency, 2),
                "입력 토큰": sum(s.tok_in for s in self.spans), "출력 토큰": sum(s.tok_out for s in self.spans)}


def parallel_map(trace, name, fn, items, max_workers=4):
    """items 를 동시에 처리한다(한 단계로 기록). fn(i, item) -> 결과. 서버가 동시 요청을 직렬로 처리하면 실제 시간은 더 길 수 있다"""
    stage = trace.new_stage()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda ia: fn(ia[0], ia[1], stage), enumerate(items)))


# ───────────────────────── 도구 루프 ─────────────────────────
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON Schema(properties · required)
    fn: object                 # fn(**인자) -> 문자열

    def schema(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": {"type": "object", **self.parameters}}}


@dataclass
class Trajectory:
    steps: list
    final: str
    stopped: str               # final | max_steps | error
    trace: Trace
    messages: list

    @property
    def calls(self):
        return [c["name"] for st in self.steps for c in st["calls"]]

    @property
    def n_steps(self):
        return len(self.steps)

    @property
    def signature(self):
        return tuple(self.calls) + (("끝" if self.stopped == "final" else "중단"),)


def run_agent(client, system, user, tools, max_steps=8, temperature=0.0, seed=0, max_tokens=512, trace=None, sample=0, max_obs_chars=1500):
    """도구 호출 루프의 완성형 — 01 번 노트북에서 직접 만든 루프와 같은 구조에 기록과 안전장치(없는 도구 · 깨진 인자 · 최대 스텝)를 더했다"""
    trace = trace or Trace()
    by = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    steps, final, stopped = [], "", "max_steps"
    for i in range(max_steps):
        r = trace.llm(client, f"LLM {i + 1}", msgs, tools=schemas, temperature=temperature, seed=seed, max_tokens=max_tokens, sample=sample)
        st = {"i": i + 1, "thought": r.content, "calls": [], "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "dur": r.latency,
              "ctx_msgs": len(msgs), "ctx_chars": sum(len(str(m.get("content") or "")) for m in msgs)}
        steps.append(st)
        if not r.tool_calls:
            final, stopped = r.content, "final"
            msgs.append({"role": "assistant", "content": r.content})
            break
        msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["raw"] or "{}"}} for c in r.tool_calls]})
        for c in r.tool_calls:
            t0 = time.perf_counter()
            err = ""
            if c["name"] not in by:
                obs, err = f"오류: 없는 도구 {c['name']!r}. 쓸 수 있는 도구 : {', '.join(by)}", "없는 도구"
            elif not c["valid"]:
                obs, err = "오류: 인자를 JSON 으로 해석하지 못했다", "깨진 인자"
            else:
                try:
                    obs = str(by[c["name"]].fn(**c["arguments"]))
                except Exception as e:  # 도구 안의 오류도 관찰로 돌려준다(모델이 고칠 기회)
                    obs, err = f"오류: {type(e).__name__}: {e}", "도구 오류"
            obs = obs[:max_obs_chars]
            trace.add(f"도구 {c['name']}", "tool", time.perf_counter() - t0, stage=trace.new_stage())
            st["calls"].append({"name": c["name"], "args": c["arguments"], "obs": obs, "error": err})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
    return Trajectory(steps, final, stopped, trace, msgs)


# ───────────────────────── 그림 ─────────────────────────
def _box(ax, x, y, w, h, text, kind="code", fc=None, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc or COLOR[kind], ec=EDGE[kind], lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)


def _arrow(ax, x0, y0, x1, y1, color="#555", lw=1.3, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def spec_chain(labels, kinds=None):
    """프롬프트 체이닝 — 입력 → 단계 1 → 단계 2 … → 출력 (노드 id = 라벨 그대로)"""
    ids = ["입력"] + list(labels) + ["출력"]
    nodes = {n: {"label": n, "x": i * 1.9, "y": 0, "kind": ("io" if n in ("입력", "출력") else (kinds or {}).get(n, "llm"))} for i, n in enumerate(ids)}
    return {"nodes": nodes, "edges": [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]}


def spec_route(classifier, branches, kinds=None):
    """라우팅 — 입력 → 분류 → (분기별 처리) → 출력"""
    n = len(branches)
    nodes = {"입력": {"label": "입력", "x": 0, "y": 0, "kind": "io"}, classifier: {"label": classifier, "x": 1.9, "y": 0, "kind": "llm"}, "출력": {"label": "출력", "x": 5.7, "y": 0, "kind": "io"}}
    edges = [("입력", classifier)]
    for i, b in enumerate(branches):
        y = (n - 1) / 2 * 1.1 - i * 1.1
        nodes[b] = {"label": b, "x": 3.8, "y": y, "kind": (kinds or {}).get(b, "llm")}
        edges += [(classifier, b), (b, "출력")]
    return {"nodes": nodes, "edges": edges}


def spec_parallel(split, workers, merge, worker_kind="llm"):
    """병렬화 — 입력 → 나누기 → (동시에 처리) → 합치기 → 출력"""
    n = len(workers)
    nodes = {"입력": {"label": "입력", "x": 0, "y": 0, "kind": "io"}, split: {"label": split, "x": 1.9, "y": 0, "kind": "code"}, merge: {"label": merge, "x": 5.7, "y": 0, "kind": "code"}, "출력": {"label": "출력", "x": 7.6, "y": 0, "kind": "io"}}
    edges = [("입력", split)]
    for i, w in enumerate(workers):
        y = (n - 1) / 2 * 1.1 - i * 1.1
        nodes[w] = {"label": w, "x": 3.8, "y": y, "kind": worker_kind}
        edges += [(split, w), (w, merge)]
    edges.append((merge, "출력"))
    return {"nodes": nodes, "edges": edges}


def draw_workflow(spec, trace=None, title=None, ax=None, figsize=None):
    """고정 경로 그림. trace 를 주면 LLM 노드를 걸린 시간(밝기)과 토큰으로 표시한다(노드 id == Span.name 인 구간을 찾음)"""
    nodes = spec["nodes"]
    xs = [v["x"] for v in nodes.values()]
    ys = [v["y"] for v in nodes.values()]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (max(7, (max(xs) + 2) * 1.05), max(2.2, (max(ys) - min(ys) + 1.6) * 0.95)))
    spans = {}
    if trace:
        for s in trace.spans:
            spans.setdefault(s.name, []).append(s)
    mx = max([sum(x.dur for x in v) for v in spans.values()] or [1]) or 1
    for nid, v in nodes.items():
        txt, fc = v["label"], None
        if nid in spans:
            d = sum(x.dur for x in spans[nid])
            tk = sum(x.tok_in + x.tok_out for x in spans[nid])
            txt += f"\n{d:.1f}초" + (f" · {tk}토큰" if tk else "")
            if v["kind"] == "llm":
                a = 0.25 + 0.75 * d / mx
                fc = (1 - (1 - 0.965) * a, 1 - (1 - 0.788) * a, 1 - (1 - 0.627) * a)
        _box(ax, v["x"], v["y"], 1.55, 0.62, txt, v["kind"], fc=fc)
    for a, b in spec["edges"]:
        A, B = nodes[a], nodes[b]
        dx = 0.78
        _arrow(ax, A["x"] + dx, A["y"], B["x"] - dx, B["y"])
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    return ax


def draw_trajectory(traj, title=None, ax=None, show_obs=True, wrap=34):
    """에이전트 궤적 — 위에서 아래로 「생각 → 도구 호출 → 관찰」을 이어 그린다. 오류가 난 도구 호출은 붉은 테두리"""
    rows = []
    for st in traj.steps:
        rows.append(("llm", f"LLM {st['i']}" + (f" · {st['thought'][:40]}" if st["thought"] else "")))
        for c in st["calls"]:
            a = ", ".join(f"{k}={str(v)[:18]}" for k, v in c["args"].items())
            rows.append(("bad" if c["error"] else "tool", f"{c['name']}({a})"))
            if show_obs:
                o = c["obs"].replace("\n", " ")
                rows.append(("io", (o[:wrap] + "…") if len(o) > wrap else o))
    rows.append(("agent" if traj.stopped == "final" else "bad", ("최종 답 : " + traj.final.replace("\n", " ")[:wrap]) if traj.stopped == "final" else f"중단({traj.stopped})"))
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 0.6))
    for i, (k, t) in enumerate(rows):
        y = -i
        _box(ax, 0, y, 5.6, 0.46, t, k, fs=8)
        if i:
            _arrow(ax, 0, y + 0.52, 0, y + 0.25, lw=1.0)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-len(rows) + 0.4, 0.7)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    return ax


def draw_context_growth(traj, ax=None, title=None):
    """스텝마다 컨텍스트(메시지 열)가 얼마나 커졌는가 — 입력 토큰이 곱해지는 모습"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    xs = [s["i"] for s in traj.steps]
    ax.bar(xs, [s["prompt_tokens"] for s in traj.steps], color=COLOR["llm"], ec=EDGE["llm"], label="입력 토큰(모델이 읽은 양)")
    ax.bar(xs, [s["completion_tokens"] for s in traj.steps], bottom=[s["prompt_tokens"] for s in traj.steps], color=COLOR["agent"], ec=EDGE["agent"], label="출력 토큰")
    ax.set_xlabel("스텝")
    ax.set_ylabel("토큰")
    ax.set_xticks(xs)
    ax.legend(fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    return ax


def draw_trajectory_flow(trajs, ax=None, title=None, max_len=10):
    """같은 입력을 여러 번 돌린 궤적을 한 그림에 겹친다 — 가로 = 몇 번째 행동, 세로 = 어떤 도구. 선 굵기 = 그 길로 간 횟수.
    고정 경로(워크플로우)는 한 줄로 겹치고, 에이전트는 갈래가 갈라진다(같은 입력이어도 궤적이 달라짐 = 비결정성)"""
    sigs = [(t.signature if hasattr(t, "signature") else tuple(t))[:max_len] for t in trajs]
    N = len(sigs)
    L = max(len(s) for s in sigs)
    first = {}
    for s in sigs:
        for i, tok in enumerate(s):
            first.setdefault(tok, []).append(i)
    names = sorted(first, key=lambda k: (k in ("끝", "중단"), np.mean(first[k])))
    ypos = {k: -i for i, k in enumerate(names)}
    cnt, trans = {}, {}
    for s in sigs:
        for i, tok in enumerate(s):
            cnt[(i, tok)] = cnt.get((i, tok), 0) + 1
            if i + 1 < len(s):
                trans[(i, tok, s[i + 1])] = trans.get((i, tok, s[i + 1]), 0) + 1
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.5, 0.95 * L + 2.6), max(2.6, 0.55 * len(names) + 1.2)))
    for (i, a, b), n in trans.items():
        ax.plot([i, i + 1], [ypos[a], ypos[b]], color=EDGE["llm"], lw=0.8 + 9 * n / N, alpha=0.55, solid_capstyle="round", zorder=1)
    for (i, tok), n in cnt.items():
        bad = tok == "중단"
        ax.scatter([i], [ypos[tok]], s=40 + 900 * n / N, color=COLOR["bad"] if bad else (COLOR["agent"] if tok == "끝" else COLOR["code"]),
                   ec=EDGE["bad"] if bad else (EDGE["agent"] if tok == "끝" else EDGE["code"]), zorder=2)
        if n / N >= 0.06:
            ax.text(i, ypos[tok], str(n), ha="center", va="center", fontsize=7.5, zorder=3)
    ax.set_yticks([ypos[k] for k in names])
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"{i + 1}" for i in range(L)], fontsize=8.5)
    ax.set_xlabel("몇 번째 행동")
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-len(names) + 0.4, 0.6)
    ax.grid(axis="y", color="#eee", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title((title or "") + f"  (총 {N}회)", fontsize=11, loc="left")
    return ax


draw_trajectory_tree = draw_trajectory_flow      # 이전 이름 호환


def bars(ax, labels, means, stds=None, colors=None, title=None, ylabel=None, fmt="{:.1f}"):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors or COLOR["code"], ec="#444", lw=0.8)
    for xi, m in zip(x, means):
        ax.text(xi, m, fmt.format(m), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


# ───────────────────────── 강사용 시험 훅 ─────────────────────────
def _dry_id():
    # 환경변수 AGENT_DRY, 없으면 표식 파일 notebooks/.agent_dry (JupyterLab 화면에서 컨테이너를 다시 만들지 않고 시험할 때. 강사용 · gitignore 대상)
    v = os.environ.get("AGENT_DRY", "").strip()
    if not v:
        try:
            v = (HERE / ".agent_dry").read_text(encoding="utf-8").strip()
        except OSError:
            v = ""
    return v


if _dry_id():
    # 서버 없이 노트북 전체를 돌려 코드 경로를 확인한다(AGENT_DRY=01 → tools/dry/dry_01.py 의 install() 이 LLM 을 가짜로 바꿈). 숫자는 의미가 없다
    import importlib
    import sys
    sys.path.insert(0, str(HERE.parent / "tools" / "dry"))
    importlib.import_module("dry_" + _dry_id()).install(sys.modules[__name__])
