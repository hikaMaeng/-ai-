"""8-1 실습 과제 A~D — 합성 데이터 · 규칙 채점기 · 에이전트 도구. 모두 시드로 고정되어 있고, 정답은 코드가 계산한다(사람이 채점하지 않음).

  A 체이닝   : 공지문 → 핵심 사실 추출 → 개조식 요약 → 영어 번역              (경로가 코드로 그려진다)
  B 라우팅   : 고객 문의 → 환불 / 기술지원 / 일반 분류 → 분기별 응답           (경로가 코드로 그려진다)
  C 병렬화   : 회의록 네 구간을 동시에 요약 → 코드가 합침                      (경로가 코드로 그려진다)
  D 개방형   : 가상 주문 DB · 환불 정책으로 푸는 질문 — 질문마다 필요한 조회 순서가 다름 (경로를 미리 그리기 어렵다)
"""
import ast
import operator as op
import random
import re
from datetime import date

from agent_lab import Tool

TODAY = date(2026, 10, 19)


def _num(s):
    return re.sub(r"[^0-9]", "", str(s))


# ═════════════════════ A · 체이닝 ═════════════════════
_EVENTS = ["가을 안전 교육", "신입사원 환영회", "사내 해커톤", "고객 감사 행사", "분기 성과 발표회", "연말 송년회", "기술 세미나", "봉사 활동의 날"]
_PLACES = ["본관 대강당", "별관 3층 회의실", "지하 1층 강당", "옥상 정원", "연수원 대회의실", "도서관 세미나실"]


def make_docs_A(n=8, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        month, day = rng.choice([10, 11, 12]), rng.randint(3, 28)
        dm, dd = month, max(1, day - rng.randint(3, 9))
        fee, cap, ext = rng.choice([10000, 20000, 30000, 45000, 50000]), rng.choice([30, 40, 60, 80, 120]), rng.randint(2100, 2999)
        ev, pl = _EVENTS[i % len(_EVENTS)], rng.choice(_PLACES)
        text = (f"[공지] 2026년 {month}월 {day}일 {pl}에서 '{ev}'가 열립니다. 참가비는 {fee:,}원이며 {dm}월 {dd}일까지 신청해야 합니다. "
                f"정원은 {cap}명이고 선착순으로 마감합니다. 문의는 총무팀(내선 {ext})으로 주십시오.")
        out.append({"id": f"A{i + 1:02d}", "text": text, "facts": {"행사일": day, "참가비": fee, "신청 마감일": dd, "정원": cap}})
    return out


def _ascii_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def score_A(doc, out):
    """영어로 썼는가(영문자 중 ASCII 비율 ≥ 0.8) + 핵심 숫자 네 개가 출력에 있는가(쉼표 · 공백 무시)"""
    flat = re.sub(r"[,\s]", "", out)
    found = {k: (str(v) in flat) for k, v in doc["facts"].items()}
    eng = _ascii_ratio(out) >= 0.8
    return {"영어": eng, "사실": sum(found.values()) / len(found), "성공": bool(eng and all(found.values())), "빠진 사실": [k for k, v in found.items() if not v]}


# ═════════════════════ B · 라우팅 ═════════════════════
CATS = ["환불", "기술지원", "일반"]
_REF_REASONS = ["파손된 채로 도착했어요", "마음에 들지 않아요", "배송이 너무 늦었어요", "상품 설명과 달라요"]
_TECH = ["화면이 하얗게만 나와요", "로그인이 계속 실패해요", "알림이 오지 않아요", "결제 버튼이 눌리지 않아요"]
_GEN = ["영업시간이 어떻게 되나요?", "주차는 가능한가요?", "매장은 어디에 있나요?", "포장 주문도 되나요?"]


def make_inquiries_B(n=24, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        cat = CATS[i % 3]
        oid = f"B{rng.randint(1000, 9999)}"
        if cat == "환불":
            t = f"주문번호 {oid} 상품이 {rng.choice(_REF_REASONS)}. 환불해 주세요."
        elif cat == "기술지원":
            t = f"앱을 열면 {rng.choice(_TECH)}. 어떻게 해야 하나요?"
        else:
            t = rng.choice(_GEN)
        out.append({"id": f"B{i + 1:02d}", "text": t, "label": cat, "oid": oid if cat == "환불" else None})
    # 헷갈리는 문의 4개(정답은 만든 규칙으로 정해 둠)
    hard = [("결제는 됐는데 앱에서 주문이 보이지 않아요. 어떻게 하나요?", "기술지원", None), ("환불 정책이 어떻게 되나요?", "일반", None),
            (f"주문번호 B{rng.randint(1000, 9999)} 환불 신청했는데 앱이 계속 꺼져서 진행 상황을 못 봐요.", "환불", None), ("영업시간 중에 앱 점검도 하나요?", "일반", None)]
    for j, (t, c, o) in enumerate(hard):
        m = re.search(r"B\d{4}", t)
        out.append({"id": f"B{n + j + 1:02d}", "text": t, "label": c, "oid": m.group(0) if m else None, "hard": True})
    return out


# 분기별 응답이 반드시 담아야 하는 것(채점 기준) — 워크플로우의 분기 프롬프트와 에이전트의 도구 결과가 같은 요구를 한다
REQUIRE_B = {"환불": "접수 번호 안내 후 환불은 3영업일 안에 처리된다고 알린다(「영업일」 포함, 주문번호가 있으면 함께 적는다)",
             "기술지원": "먼저 앱 버전을 물어본다(「버전」 포함)",
             "일반": "영업시간은 평일 9시~18시라고 알린다(「9시」 포함)"}
_KEY_B = {"환불": "영업일", "기술지원": "버전", "일반": "9시"}


def score_B(item, answer, routed=None):
    key = _KEY_B[item["label"]]
    ok_elem = key in answer
    if item["label"] == "환불" and item.get("oid"):
        ok_elem = ok_elem and item["oid"] in answer
    return {"요소": ok_elem, "분류": (routed == item["label"]) if routed is not None else None, "성공": bool(ok_elem and (routed is None or routed == item["label"]))}


B_TOOLS_TEXT = {"refund_policy": "환불 문의 응대 규칙 : " + REQUIRE_B["환불"],
                "tech_checklist": "기술 문의 응대 규칙 : " + REQUIRE_B["기술지원"],
                "store_info": "일반 문의 응대 규칙 : " + REQUIRE_B["일반"]}
B_TOOL_FOR = {"환불": "refund_policy", "기술지원": "tech_checklist", "일반": "store_info"}


def tools_B():
    z = {"type": "object"}
    return [Tool("refund_policy", "환불 · 반품 · 취소 문의에 응대할 때 따르는 규칙을 돌려준다.", {"properties": {}}, lambda: B_TOOLS_TEXT["refund_policy"]),
            Tool("tech_checklist", "앱 오류 · 로그인 · 알림 같은 기술 문의에 응대할 때 따르는 규칙을 돌려준다.", {"properties": {}}, lambda: B_TOOLS_TEXT["tech_checklist"]),
            Tool("store_info", "영업시간 · 위치 · 주차 같은 일반 문의에 응대할 때 따르는 규칙을 돌려준다.", {"properties": {}}, lambda: B_TOOLS_TEXT["store_info"])]


# ═════════════════════ C · 병렬화 ═════════════════════
_FLUFF = ["지난 회의의 결정 사항을 간단히 확인했다.", "참석자 전원이 일정에 동의했다.", "세부 내용은 다음 주에 다시 공유하기로 했다.", "관련 부서의 의견을 추가로 수렴하기로 했다.",
          "현재까지 큰 이견은 없었다.", "회의록은 공유 폴더에 올리기로 했다.", "다음 회의 날짜는 별도로 알리기로 했다.", "담당자가 진행 상황을 정리해 두기로 했다."]
_OWNERS = ["김도윤", "이서연", "박지후", "최하준", "정유나", "한지민"]
_RISKS = ["서버 용량 부족", "협력사 납기 지연", "인력 이탈", "보안 점검 미흡", "예산 초과", "일정 중복"]


def make_meetings_C(n=6, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        budget, owner = rng.randint(2, 98), rng.choice(_OWNERS)
        dl, risk = (rng.randint(10, 12), rng.randint(2, 28)), rng.choice(_RISKS)
        facts = {"예산": f"{budget}억", "담당자": owner, "마감일": f"{dl[0]}월 {dl[1]}일", "위험": risk}
        planted = [f"올해 사업 예산은 {budget}억 원으로 확정했다.", f"이 사업의 총괄 담당자는 {owner}으로 정했다.", f"최종 제출 마감일은 {dl[0]}월 {dl[1]}일이다.", f"가장 큰 위험 요인으로 {risk}이(가) 지적되었다."]
        sections = []
        for k in range(4):
            fl = rng.sample(_FLUFF, 3)
            body = fl[:1] + [planted[k]] + fl[1:]
            sections.append(" ".join(body))
        out.append({"id": f"C{i + 1}", "sections": sections, "facts": facts})
    return out


def score_C(meeting, out):
    flat = re.sub(r"\s", "", out)
    chk = {"예산": _num(meeting["facts"]["예산"]) in flat, "담당자": meeting["facts"]["담당자"] in flat,
           "마감일": (re.sub(r"\s", "", meeting["facts"]["마감일"]) in flat) or all(p in flat for p in re.findall(r"\d+", meeting["facts"]["마감일"])),
           "위험": meeting["facts"]["위험"].replace(" ", "") in flat}
    return {"사실": sum(chk.values()) / 4, "성공": all(chk.values()), "빠진 사실": [k for k, v in chk.items() if not v]}


def tools_C(meeting):
    secs = meeting["sections"]
    return [Tool("section_count", "회의록이 몇 개 구간으로 나뉘어 있는지 돌려준다.", {"properties": {}}, lambda: str(len(secs))),
            Tool("read_section", "회의록의 n번째 구간 원문을 돌려준다(1부터).", {"properties": {"n": {"type": "integer", "description": "구간 번호(1부터)"}}, "required": ["n"]},
                 lambda n: secs[int(n) - 1] if 1 <= int(n) <= len(secs) else f"오류: 구간은 1~{len(secs)}번이다")]


# ═════════════════════ D · 개방형 ═════════════════════
POLICY_TEXT = ("환불 정책 : ① 배송완료 후 7일 이내 = 결제금액 전액 ② 배송완료 후 8~14일 = 결제금액의 70% ③ 15일 이후 = 환불 불가 "
               "④ 배송중 = 취소 가능(결제금액 전액) ⑤ 주문취소 · 환불완료 = 이미 처리되어 환불 불가 ⑥ 결제금액 = 상품가 − 쿠폰 사용액(쿠폰 사용분은 환불하지 않음)")
SYSTEM_D = (f"너는 쇼핑몰 환불 상담 도우미다. 오늘은 {TODAY.isoformat()}이다. 도구로 사실을 확인한 뒤 환불 정책에 따라 답한다. "
            "금액은 원 단위 정수로 쓰고, 확인되지 않는 것은 지어내지 않는다. 계산은 calc 도구로 한다.")

_ITEMS = ["노트북", "키보드", "무선 이어폰", "모니터", "책상 조명", "백팩", "스마트워치", "마우스", "공기청정기", "전기포트", "태블릿", "블루투스 스피커"]
_NAMES = ["김영희", "이철수", "박민수", "최지우", "정다은", "한승우"]


def make_db_D(seed=0):
    rng = random.Random(seed)
    orders, cust, k = {}, {n: [] for n in _NAMES}, 0
    statuses = ["배송완료"] * 7 + ["배송중"] * 2 + ["주문취소", "환불완료"]
    items = _ITEMS[:]
    rng.shuffle(items)
    for name in _NAMES:
        for _ in range(rng.choice([2, 2, 3])):
            k += 1
            oid, st = f"A{1000 + k}", rng.choice(statuses)
            price = rng.choice([39000, 59000, 89000, 129000, 249000, 990000])
            coupon = rng.choice([0, 0, 5000, 10000])
            days = rng.randint(1, 20) if st == "배송완료" else None
            orders[oid] = {"id": oid, "고객": name, "상품": items[k % len(items)], "상품가": price, "쿠폰": coupon, "상태": st, "배송완료 후 일수": days}
            cust[name].append(oid)
    return orders, cust


def refund_of(o):
    """정책 ①~⑥을 코드로 옮긴 정답 계산 — (판정, 금액)"""
    st, pay = o["상태"], o["상품가"] - o["쿠폰"]
    if st in ("주문취소", "환불완료"):
        return "불가", 0
    if st == "배송중":
        return "가능", pay
    d = o["배송완료 후 일수"]
    if d <= 7:
        return "가능", pay
    if d <= 14:
        return "가능", int(round(pay * 0.7))
    return "불가", 0


def make_questions_D(seed=0):
    orders, cust = make_db_D(seed)
    rng = random.Random(seed + 1)
    qs = []
    ids = list(orders)
    rng.shuffle(ids)
    for oid in ids[:8]:     # 유형 1 : 주문번호가 주어짐(경로가 짧고 정해져 있음)
        j, amt = refund_of(orders[oid])
        qs.append({"유형": "주문번호", "q": f"주문 {oid}은 환불할 수 있나요? 가능하면 환불 금액을 알려 주세요.", "expect": (j, amt), "oid": oid})
    for name in rng.sample(_NAMES, 4):    # 유형 2 : 이름 + 상품(고객 → 주문 찾기 → 조회)
        oid = rng.choice(cust[name])
        o = orders[oid]
        if sum(orders[x]["상품"] == o["상품"] for x in cust[name]) != 1:
            continue
        j, amt = refund_of(o)
        qs.append({"유형": "이름+상품", "q": f"{name}님이 주문한 {o['상품']}을(를) 환불하면 얼마를 돌려받나요? 환불이 안 되면 안 된다고 알려 주세요.", "expect": (j, amt), "oid": oid})
    for name in rng.sample(_NAMES, 4):    # 유형 3 : 고객의 모든 주문 중 환불 가능한 것의 합계(조회 횟수가 달라짐)
        tot = sum(refund_of(orders[x])[1] for x in cust[name])
        qs.append({"유형": "고객 합계", "q": f"{name}님의 주문 중 지금 환불할 수 있는 것의 환불 금액을 모두 합하면 얼마인가요?", "expect": ("합계", tot), "oid": None})
    qs.append({"유형": "없는 주문", "q": "주문 A9999는 환불할 수 있나요? 가능하면 환불 금액을 알려 주세요.", "expect": ("없음", 0), "oid": "A9999"})
    qs.append({"유형": "없는 고객", "q": "홍길동님의 주문 중 환불할 수 있는 것의 환불 금액 합계는 얼마인가요?", "expect": ("없음", 0), "oid": None})
    return orders, cust, qs


def score_D(q, answer):
    """정답 판정 · 금액이 답에 들어 있는지 규칙으로 본다. 「모르겠다 · 확인할 수 없다」만으로는 어떤 정답도 인정하지 않는다(그런 답은 불가 · 없음의 정답이 아님)"""
    j, amt = q["expect"]
    flat = re.sub(r"[,\s]", "", answer)
    if j == "없음":
        ok = bool(re.search(r"찾을수없|존재하지않|조회되지|등록되지|없는주문|없는고객|해당(주문|고객)이?없", flat))
    elif j == "불가":
        ok = bool(re.search(r"불가|환불할수없|환불이어렵|환불은어렵|환불되지않|환불이안", flat))
    elif j == "합계" and amt == 0:
        ok = bool(re.search(r"0원|환불가능한(주문|것)이?없|모두불가|전부불가", flat))
    else:
        ok = str(amt) in flat and not (re.search(r"불가|환불할수없", flat) and "가능" not in flat)
    return {"성공": ok}


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg, ast.Pow: op.pow}


def _ev(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.left), _ev(n.right))
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    raise ValueError("허용되지 않는 식")


def safe_calc(expression):
    v = _ev(ast.parse(str(expression).replace(",", "").replace("×", "*").replace("÷", "/"), mode="eval").body)
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def tools_D(orders, cust):
    def lookup_order(order_id):
        o = orders.get(str(order_id).strip())
        return "오류: 그런 주문이 없다" if not o else " · ".join(f"{k}={v}" for k, v in o.items() if v is not None)

    def search_customer(name):
        ids = cust.get(str(name).strip())
        return "오류: 그런 고객이 없다" if ids is None else "주문번호 : " + ", ".join(ids)

    return [Tool("lookup_order", "주문번호로 주문 한 건(고객 · 상품 · 상품가 · 쿠폰 · 상태 · 배송완료 후 일수)을 조회한다.", {"properties": {"order_id": {"type": "string", "description": "예 : A1003"}}, "required": ["order_id"]}, lookup_order),
            Tool("search_customer", "고객 이름으로 그 고객의 주문번호 목록을 조회한다.", {"properties": {"name": {"type": "string", "description": "고객 이름"}}, "required": ["name"]}, search_customer),
            Tool("get_policy", "환불 정책 전문을 돌려준다.", {"properties": {}}, lambda: POLICY_TEXT),
            Tool("calc", "사칙연산 식을 계산한다(예 : (89000-5000)*0.7).", {"properties": {"expression": {"type": "string"}}, "required": ["expression"]}, safe_calc)]


def plan_D(q, orders, cust):
    """정답 경로(시험용) — ScriptedPolicy 가 따라가는 완전한 에이전트. 진짜 모델이 이 경로를 찾아가는지는 별개의 문제"""
    steps = []
    j, amt = q["expect"]
    if q["유형"] in ("주문번호", "없는 주문"):
        steps.append(("lookup_order", {"order_id": q["oid"]}))
        if q["oid"] in orders:
            steps += [("get_policy", {}), ("calc", {"expression": str(amt)})]
    elif q["유형"] == "이름+상품":
        name = orders[q["oid"]]["고객"]
        steps += [("search_customer", {"name": name})] + [("lookup_order", {"order_id": x}) for x in cust[name][:cust[name].index(q["oid"]) + 1]] + [("get_policy", {}), ("calc", {"expression": str(amt)})]
    elif q["유형"] == "고객 합계":
        name = re.match(r"(\S+)님", q["q"]).group(1)
        steps += [("search_customer", {"name": name})] + [("lookup_order", {"order_id": x}) for x in cust[name]] + [("get_policy", {}), ("calc", {"expression": str(amt)})]
    elif q["유형"] == "없는 고객":
        steps.append(("search_customer", {"name": "홍길동"}))
    final = {"없음": "해당 정보를 찾을 수 없습니다.", "불가": "환불할 수 없습니다(불가).", "합계": f"환불 가능한 금액의 합계는 {amt:,}원입니다."}.get(j, f"환불 가능합니다. 환불 금액은 {amt:,}원입니다.")
    return steps, final
