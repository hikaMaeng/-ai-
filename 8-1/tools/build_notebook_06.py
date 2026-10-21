"""실습 노트북 06 생성기 (강사용).

    python build_notebook_06.py   → ../notebooks/06_코딩_에이전트의_한_궤적.ipynb (출력 없이)
"""
from nb import Book

b = Book("06_코딩_에이전트의_한_궤적.ipynb")

b.md(r'''
# 실습 06 · 코딩 에이전트의 한 궤적 — 검색 · 읽기 · 수정 · 테스트 루프

**직접 해 보는 것** : 작은 파이썬 결제 프로젝트(임시 폴더 안)에 버그를 심어 두고, 모델에게 `run_tests` · `grep` · `read_file` · `edit_file` 네 도구만 주어 고치게 한다. 한 번의 궤적을 Thought / Action / Observation 으로 읽고, 같은 과제를 여러 번 돌려 ① 실패 로그가 수정 범위를 좁히는지 ② 시험(검증 신호)을 빼면 무엇이 달라지는지 ③ 버그 종류마다 어떤지 ④ 종료를 코드가 정하는 것과 모델이 정하는 것이 어떻게 다른지 숫자로 본다. 성공은 모델이 볼 수 없는 **숨은 확인**으로 코드가 채점한다. 채점기부터 시험하고 나서 모델을 돌린다.

**강의의 어느 개념과 이어지는가**
- **ReAct** : Thought → Action → Observation 이 교대한다. 코딩에서는 시험 로그가 관찰이다
- **반성** : 실패 로그를 근거로 수정 범위를 좁힌다
- **종료** : 사람의 판단이 아니라 「시험 통과」가 종료 조건이다
- 코딩이 대표 사례인 이유 : 실행 · 시험으로 결과를 자동 검증하므로 반성과 재시도가 실제로 작동한다. **검증 신호(PASS/FAIL)가 있으면 루프가 스스로 수렴한다**

강의에서 보는 한 궤적은 다음 모양이다. 이 노트북은 이것을 실제로 돌려 본다.
```
[목표] 결제 모듈의 NullPointerException 을 고쳐라(테스트 통과까지)
Thought: 재현부터. 어떤 테스트가 깨지는지 확인한다.      Action: run_tests("payment")
Observation: FAIL — checkout 테스트에서 NPE(user.id 접근)
Thought: user 가 null 인 경로가 있다. 호출부를 찾자.     Action: grep("getUser(")
Observation: 게스트 결제 시 user 가 미할당되는 분기 발견
Thought: 게스트 분기에 null 가드 누락 → 최소 수정.       Action: edit(파일, 게스트 대체값 처리 추가) → Action: run_tests("payment")
Observation: PASS (12/12)
```

**문제** : 프로젝트는 `payment/`(users · cart · discount · checkout, 소스 4개)와 `tests/`(시험 12개)로 되어 있다. 버그는 세 종류다 — 게스트 결제에서 `user.id` 접근(파이썬에서는 `AttributeError`, 강의의 NPE 에 해당) · 수량 할인의 경계값(off-by-one) · 할인 뒤 반올림. 정답 패치와 숨은 시험은 노트북 안에 있고 모델에게는 보이지 않는다. 사람이 채점하지 않는다.

| 셀 | 하는 일 |
|---|---|
| 1 | 설정 · LLM 호출 횟수 계산 |
| 2 | 프로젝트 보기 — 파일 · 버그가 있는 줄 · 깨지는 시험 수 |
| 3 | 도구 만들기 — 네 도구와 안전장치를 손으로 불러 본다 |
| 4 | 채점기 점검 — 정답 패치 · 버그 그대로 · 엉뚱한 수정을 직접 넣어 숨은 확인이 맞게 판정하는지 |
| 5 | 한 궤적 따라가기 — Thought / Action / Observation · 궤적 그림 · 숨은 확인 |
| 6 | 실패 로그가 수정 범위를 좁히는가(반성) — 같은 과제를 N번 |
| 7 | 검증 신호가 없으면 — `run_tests` 를 뺀 도구 세트 |
| 8 | 버그 세 종류 |
| 9 | 종료 조건 — 시험 통과를 코드가 종료 조건으로 쓰는 루프 vs 모델이 스스로 끝내는 루프 |
| 10 | 정리 · 결과 저장 · 확인 질문 |

- 실행 : 셀 2 ~ 4 는 서버 없이 돈다(샌드박스 · 도구 · 채점기는 진짜로 돈다). 셀 5 ~ 9 는 LM Studio 서버가 켜져 있고 도구 호출이 되는 모델이 로드되어 있어야 한다(없으면 `AGENT_MODE=replay` 로 강사 실행 결과를 재생). 시드 · 표본 수 · 온도 같이 결과를 바꾸는 설정은 모두 출력에 찍는다
- **이 노트북은 모델이 고친 코드를 실제로 실행한다.** 컨테이너 안에서, 실행마다 새 임시 폴더를 만들고, 서브프로세스와 시간 제한으로 가둔다. 이것은 실수(무한 반복 · 경로 이탈)를 막는 장치이지 악의적인 코드를 막는 장치가 아니다. 컨테이너 밖의 내 PC 에서 직접 돌리지 않는다
- 같은 질문을 온도 > 0 으로 여러 번 돌릴 때는 표본 번호 i 마다 `seed` = 기준 시드 + i 로 서버의 표본 추출을 바꾸고(시드가 같으면 같은 답이 반복될 수 있다), `sample=i` 로 캐시 칸도 나눈다. 같은 표본 번호는 같은 응답을 쓴다
- **결과가 기대와 달라도 그대로 남긴다.** 숫자가 말하는 것은 이 모델 · 이 설정 · 이 표본 수에서 본 것뿐이다
''')

# ───────────────────────── 셀 1 ─────────────────────────
b.md(r'''
### 셀 1 · 설정
- `n_runs` : 같은 과제를 몇 번 돌리는가. `max_steps` : 한 번의 궤적이 쓸 수 있는 LLM 호출 상한(루프의 안전장치). `temperature` · `seed` : 표본 추출 설정. `test_timeout` : 시험 한 번이 쓸 수 있는 시간(초). `min_factor` : 「최소 수정」으로 인정하는 바꾼 줄 수의 상한(정답 패치 줄 수의 몇 배까지)
- 셀 5 의 한 궤적은 온도 0 으로, 셀 6 이후의 반복은 `temperature` 로 돌린다
- 아래에서 이 노트북이 LLM 을 부르는 횟수를 계산해 찍는다. 캐시에 있는 호출은 서버를 부르지 않는다
''')
b.code(r'''
# 셀 1 · 설정
import sys, json, time, re, os, shutil, tempfile, subprocess, difflib, atexit, dataclasses
try:
    import resource                    # 시험 서브프로세스의 자원 제한용. 스레드가 도는 중에 자식 프로세스 안에서 처음 가져오지 않도록 미리 가져온다
except ImportError:
    pass
from pathlib import Path
sys.path.insert(0, ".")
from agent_lab import *
from agent_tasks import *

CFG = dict(n_runs=12, max_steps=12, max_tokens=800, temperature=0.6, seed=0, trace_temperature=0.0, trace_sample=0, test_timeout=20, min_factor=2)
llm = LLM()          # 주소 · 모델은 환경변수(LMSTUDIO_URL · AGENT_MODEL)에서 읽는다
BUG_NAMES = ["게스트", "경계값", "반올림"]

N = CFG["n_runs"]
n_traj = 1 + N + N + 2 * N          # 셀 5 한 번 + 셀 6(게스트 N) + 셀 7(시험 없이 N) + 셀 8(다른 두 버그 각 N). 셀 9 의 코드 종료 루프는 같은 응답을 캐시에서 다시 쓴다
avg_steps = 7                       # 궤적 하나가 평균 이만큼 호출한다고 가정한 값(어림)
print(llm, "| 설정 :", CFG)
print("버그 종류 :", BUG_NAMES)
print(f"이 노트북이 LLM 을 부르는 횟수 : 약 {n_traj * avg_steps}번 (궤적 {n_traj}개 × 평균 {avg_steps}스텝 가정, 최대 {n_traj * CFG['max_steps']}번) — 캐시에 있으면 서버를 부르지 않는다")
print("  셀 9 의 코드 종료 루프는 모델 종료 루프와 같은 질문 · 같은 표본 번호를 쓰므로 통과 시점까지의 호출이 캐시에서 다시 쓰인다. 셀 2 ~ 4 는 LLM 을 부르지 않는다")
''')

# ───────────────────────── 셀 2 ─────────────────────────
b.md(r'''
### 셀 2 · 프로젝트 보기
- 노트북이 작은 쇼핑 결제 프로젝트를 **임시 폴더(샌드박스)** 에 만든다. 소스 4개 + 시험 3개 파일(시험 12개). 버그는 정답 프로젝트의 한 곳을 일부러 틀리게 바꿔서 만든다
- 숨은 확인 = 모델이 보지 못하는 시험 2개를 더한 것(그 버그를 겨냥한 시험). 고친 코드에 **원래 시험 파일**과 숨은 시험을 새 폴더에서 다시 돌리므로, 모델이 시험 파일을 고쳐서 통과시켜도 채점에는 소용이 없다
- 임시 폴더는 쓰고 나면 바로 지운다. 이전 실행이 중간에 멈춰 남긴 폴더도 이 셀에서 지운다(표식 파일이 있는 `sbx06_*` 폴더만)
- 이 셀은 서버 없이 돈다. 버그가 있는 프로젝트에서 시험을 한 번씩 돌려 몇 개가 깨지는지 코드로 확인한다
''')
b.code(r'''
# 셀 2 · 프로젝트 보기
FILES = {
"payment/__init__.py": """# 결제 모듈
""",
"payment/users.py": """# 세션에서 사용자를 찾는다. 게스트(로그인하지 않은 세션)는 None 이다.
class User:
    def __init__(self, id, name, grade):
        self.id = id
        self.name = name
        self.grade = grade


USERS = {
    "u1": User(1, "김영희", "gold"),
    "u2": User(2, "이철수", "silver"),
    "u3": User(3, "박민수", "basic"),
}


def get_user(session):
    return USERS.get(session.get("user_key"))
""",
"payment/cart.py": """# 장바구니 = [(상품명, 단가, 수량), ...]
def subtotal(cart):
    return sum(price * qty for _, price, qty in cart)


def total_qty(cart):
    return sum(qty for _, _, qty in cart)
""",
"payment/discount.py": """# 할인율은 % 단위 정수다.
RATES = {"basic": 0, "silver": 5, "gold": 10}
BULK_MIN = 10          # 수량 합계가 이 값 이상이면 수량 할인


def grade_rate(grade):
    return RATES.get(grade, 0)


def bulk_rate(qty):
    # 수량 합계가 BULK_MIN 개 이상이면 3% 추가 할인
    return 3 if qty >= BULK_MIN else 0


def apply_discount(amount, rate):
    # 할인 뒤 금액, 원 단위 반올림(0.5 는 올림)
    return (amount * (100 - rate) + 50) // 100
""",
"payment/checkout.py": """# 장바구니를 결제 금액과 영수증으로 만든다.
from payment.users import get_user
from payment.cart import subtotal, total_qty
from payment.discount import grade_rate, bulk_rate, apply_discount


def checkout(session, cart):
    user = get_user(session)
    uid = user.id if user else None
    grade = user.grade if user else "guest"
    rate = grade_rate(grade) + bulk_rate(total_qty(cart))
    total = apply_discount(subtotal(cart), rate)
    return {"user_id": uid, "grade": grade, "rate": rate, "total": total}
""",
}
TESTS = {   # 보이는 시험 12개 — 모델이 run_tests 로 돌려 볼 수 있다
"tests/test_users.py": """import unittest
from payment.users import get_user


class UserTests(unittest.TestCase):
    def test_logged_in_user(self):
        self.assertEqual(get_user({"user_key": "u1"}).id, 1)

    def test_guest_returns_none(self):
        self.assertIsNone(get_user({}))
""",
"tests/test_discount.py": """import unittest
from payment.discount import grade_rate, bulk_rate, apply_discount


class DiscountTests(unittest.TestCase):
    def test_grade_rate_gold(self):
        self.assertEqual(grade_rate("gold"), 10)

    def test_grade_rate_unknown_is_zero(self):
        self.assertEqual(grade_rate("vip"), 0)

    def test_bulk_rate_9(self):
        self.assertEqual(bulk_rate(9), 0)

    def test_bulk_rate_10(self):
        self.assertEqual(bulk_rate(10), 3)

    def test_apply_discount_exact(self):
        self.assertEqual(apply_discount(20000, 10), 18000)

    def test_apply_discount_half_up(self):
        self.assertEqual(apply_discount(1990, 5), 1891)
""",
"tests/test_checkout.py": """import unittest
from payment.checkout import checkout


class CheckoutTests(unittest.TestCase):
    def test_gold_single(self):
        r = checkout({"user_key": "u1"}, [("키보드", 59000, 1)])
        self.assertEqual(r["total"], 53100)

    def test_silver_bulk(self):
        r = checkout({"user_key": "u2"}, [("마우스", 10000, 12)])
        self.assertEqual(r["total"], 110400)

    def test_guest_single_total(self):
        r = checkout({}, [("키보드", 59000, 1)])
        self.assertEqual(r["total"], 59000)

    def test_guest_bulk_total(self):
        r = checkout({}, [("마우스", 10000, 12)])
        self.assertEqual(r["total"], 116400)
""",
}
# 버그마다 : 어디를 어떻게 틀리게 만들었는가(bad) · 정답(good) · 모델에게 주는 과제 · 숨은 시험
BUGS = {
"게스트": dict(
    file="payment/checkout.py",
    bad="    uid = user.id\n    grade = user.grade\n",
    good="    uid = user.id if user else None\n    grade = user.grade if user else \"guest\"\n",
    task="결제 모듈에 버그가 있다. 로그인하지 않은 게스트로 결제하면 AttributeError 가 난다. 고쳐라. 게스트 결제는 등급 할인 없이 처리하고, 영수증의 user_id 는 None, grade 는 \"guest\" 로 한다.",
    hidden="""import unittest
from payment.checkout import checkout


class HiddenGuest(unittest.TestCase):
    def test_guest_receipt(self):
        r = checkout({}, [("키보드", 59000, 1)])
        self.assertIsNone(r["user_id"])
        self.assertEqual(r["grade"], "guest")
        self.assertEqual(r["rate"], 0)

    def test_user_receipt_unchanged(self):
        r = checkout({"user_key": "u1"}, [("노트", 30000, 2)])
        self.assertEqual((r["user_id"], r["grade"], r["total"]), (1, "gold", 54000))
"""),
"경계값": dict(
    file="payment/discount.py",
    bad="    return 3 if qty > BULK_MIN else 0\n",
    good="    return 3 if qty >= BULK_MIN else 0\n",
    task="결제 모듈에 버그가 있다. 수량 합계가 정확히 10개인 주문의 결제 금액이 기대와 다르다는 보고가 있다. 수량 할인 규칙은 「수량 합계가 10개 이상이면 3% 추가 할인」이다. 원인을 찾아 고쳐라.",
    hidden="""import unittest
from payment.checkout import checkout


class HiddenBulk(unittest.TestCase):
    def test_exactly_ten(self):
        r = checkout({"user_key": "u3"}, [("펜", 1000, 10)])
        self.assertEqual((r["rate"], r["total"]), (3, 9700))

    def test_nine_and_eleven(self):
        self.assertEqual(checkout({"user_key": "u3"}, [("펜", 1000, 9)])["total"], 9000)
        self.assertEqual(checkout({"user_key": "u3"}, [("펜", 1000, 11)])["total"], 10670)
"""),
"반올림": dict(
    file="payment/discount.py",
    bad="    return amount * (100 - rate) // 100\n",
    good="    return (amount * (100 - rate) + 50) // 100\n",
    task="결제 모듈에 버그가 있다. 할인 뒤 금액의 원 단위 반올림이 기대와 다르다는 보고가 있다. 규칙은 「소수점 이하는 반올림(0.5 는 올림)」이다. 원인을 찾아 고쳐라.",
    hidden="""import unittest
from payment.checkout import checkout
from payment.discount import apply_discount


class HiddenRound(unittest.TestCase):
    def test_round_down_case(self):
        self.assertEqual(apply_discount(1012, 5), 961)

    def test_gold_half(self):
        self.assertEqual(checkout({"user_key": "u1"}, [("펜", 1255, 1)])["total"], 1130)
"""),
}


def project_files(bug=None):
    """정답 프로젝트(bug=None) 또는 그 버그가 심어진 프로젝트의 파일 모음"""
    files = {**FILES, **TESTS}
    if bug:
        B = BUGS[bug]
        assert B["good"] in files[B["file"]], bug
        files[B["file"]] = files[B["file"]].replace(B["good"], B["bad"], 1)
    return files


# ── 샌드박스 : 실행마다 새 임시 폴더 · 서브프로세스 · 시간 제한 ──
SBX = "sbx06_"
LIVE = set()


def sweep_stale():
    """표식 파일(.sbx06)이 있는 이전 실행의 임시 폴더를 지운다. 지운 개수를 돌려준다"""
    n = 0
    for d in Path(tempfile.gettempdir()).glob(SBX + "*"):
        if (d / ".sbx06").exists():
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def _limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (CFG["test_timeout"] + 5,) * 2)      # CPU 시간
        resource.setrlimit(resource.RLIMIT_FSIZE, (2_000_000,) * 2)                  # 파일 하나의 크기
    except Exception:
        pass


class Sandbox:
    def __init__(self, files):
        self.root = Path(tempfile.mkdtemp(prefix=SBX)).resolve()
        (self.root / ".sbx06").write_text("sandbox", encoding="utf-8")
        LIVE.add(self.root)
        for rel, text in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)
        LIVE.discard(self.root)

    def resolve(self, rel):
        """샌드박스 안의 경로만 허락한다(절대 경로 · .. · 심볼릭 링크로 나가는 경로 · 점으로 시작하는 파일 거부)"""
        rel = str(rel).strip()
        if not rel or rel.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", rel):
            raise ValueError("상대 경로만 쓸 수 있다(예 : payment/users.py)")
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError("샌드박스 밖의 경로는 쓸 수 없다")
        if any(part.startswith(".") for part in p.relative_to(self.root).parts):
            raise ValueError("점으로 시작하는 파일은 쓸 수 없다")
        return p

    def files(self):
        return sorted(str(p.relative_to(self.root)).replace(os.sep, "/") for p in self.root.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and p.name != ".sbx06")

    def run_unittest(self, timeout=None):
        """python -m unittest 를 서브프로세스로 돌린다. (종료 코드, 출력, 시간 초과 여부)"""
        timeout = timeout or CFG["test_timeout"]
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8"}
        try:
            r = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, text=True, encoding="utf-8", errors="replace",
                               preexec_fn=_limits if os.name == "posix" else None)
            out, code, to = r.stdout, r.returncode, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            code, to = -1, True
        out = out[-20000:].replace(str(self.root) + os.sep, "").replace(str(self.root), ".")
        return code, out, to


atexit.register(lambda: [shutil.rmtree(p, ignore_errors=True) for p in list(LIVE)])
print("지난 실행에서 남은 임시 폴더", sweep_stale(), "개를 지웠다")

LINE = re.compile(r"^(\w+) \(([\w.]+)\) \.\.\. (.+)$", re.M)


def parse(out):
    """unittest -v 출력 → ({시험 id: ok | FAIL | ERROR …}, 실행된 시험 수)"""
    res = {m.group(2): m.group(3).strip() for m in LINE.finditer(out)}
    ran = re.search(r"Ran (\d+) tests? in", out)
    return res, (int(ran.group(1)) if ran else 0)


# 프로젝트를 훑어본다
base = project_files()
src_lines = sum(len(t.splitlines()) for k, t in base.items() if k.startswith("payment/"))
test_lines = sum(len(t.splitlines()) for k, t in base.items() if k.startswith("tests/"))
print("\n파일 :")
for k, t in base.items():
    print(f"  {k:<24}{len(t.splitlines()):>3}줄")
print(f"  소스 {src_lines}줄 · 시험 {test_lines}줄")

# 정답 프로젝트 : 시험 12개가 모두 통과해야 한다
with Sandbox(base) as s0:
    code, out, to = s0.run_unittest()
res0, ran0 = parse(out)
print(f"\n정답 프로젝트 : 시험 {ran0}개 중 {sum(v == 'ok' for v in res0.values())}개 통과 (종료 코드 {code})")
N_VISIBLE = ran0

# 버그마다 : 심은 줄 · 깨지는 시험 · 숨은 확인의 전체 시험 수
TARGETS, ALL_IDS, REF_LINES = {}, {}, {}
for bug in BUG_NAMES:
    B = BUGS[bug]
    files = project_files(bug)
    ln = files[B["file"]].splitlines().index(B["bad"].splitlines()[0]) + 1
    with Sandbox(files) as s1:
        _, out1, _ = s1.run_unittest()
    res1, ran1 = parse(out1)
    TARGETS[bug] = {k for k, v in res1.items() if v != "ok"}
    with Sandbox({**base, "tests/test_hidden.py": B["hidden"]}) as s2:
        code2, out2, _ = s2.run_unittest()
    res2, ran2 = parse(out2)
    assert code2 == 0 and all(v == "ok" for v in res2.values()), (bug, out2[-500:])
    ALL_IDS[bug] = set(res2)
    REF_LINES[bug] = len(B["bad"].splitlines()) + len(B["good"].splitlines())
    print(f"\n[{bug}] 버그가 있는 줄 : {B['file']}:{ln}  →  {B['bad'].splitlines()[0].strip()!r}")
    print(f"  시험 {ran1}개 중 {len(TARGETS[bug])}개가 깨진다 : {sorted(k.split('.', 1)[1] for k in TARGETS[bug])}")
    print(f"  정답 패치 = 지울 줄 {len(B['bad'].splitlines())} + 쓸 줄 {len(B['good'].splitlines())} = {REF_LINES[bug]}줄 · 숨은 확인의 전체 시험 {len(ALL_IDS[bug])}개 (보이는 {N_VISIBLE} + 숨은 {len(ALL_IDS[bug]) - N_VISIBLE})")
print("\n게스트 과제 :", BUGS["게스트"]["task"])
''')

# ───────────────────────── 셀 3 ─────────────────────────
b.md(r'''
### 셀 3 · 도구 만들기 — 손으로 한 번씩 불러 본다
- 도구 네 개 : `run_tests()` · `grep(pattern)` · `read_file(path)` · `edit_file(path, old, new)`. 설명은 모델에게 보이는 글이므로 무엇을 돌려주는지, 어떤 경우에 오류인지를 적었다
- 안전장치 : ① 경로는 샌드박스 안만(절대 경로 · `..` · 점 파일 거부) ② `edit_file` 은 `old` 가 파일 안에 **정확히 한 번** 나올 때만 바꾼다(없거나 여러 번이면 오류를 관찰로 돌려줘 모델이 고칠 기회를 준다) ③ 시험은 서브프로세스 + 시간 제한(`test_timeout`)으로 돌린다
- 아래에서 버그가 있는 프로젝트에 이 도구들을 직접 불러 본다. 모델이 하는 일을 우리가 손으로 해 보는 것이다. 마지막에 일부러 끝나지 않는 시험을 넣어 시간 제한이 걸리는지도 본다
- 시험 로그는 깨진 시험마다 머리글과 오류 꼬리만 남겨 짧게 만든다. 경로의 임시 폴더 부분은 지운다
''')
b.code(r'''
# 셀 3 · 도구 만들기
def fail_log(out, n_blocks=4):
    """unittest 출력에서 깨진 시험마다 머리글 + 오류 꼬리(마지막 5줄)만 남긴다"""
    blocks = re.split(r"^={70}$", out, flags=re.M)[1:]
    parts = []
    for blk in blocks[:n_blocks]:
        lines = blk.strip("\n").split("\n")
        body = blk.split("-" * 70)[1] if blk.count("-" * 70) >= 1 and len(blk.split("-" * 70)) > 1 else ""
        tail = [x for x in body.split("\n") if x.strip()][-5:]
        parts.append(lines[0] + "\n" + "\n".join(tail))
    if len(blocks) > n_blocks:
        parts.append(f"… 깨진 시험 {len(blocks) - n_blocks}개 더")
    return "\n".join(parts)


def tests_text(code, out, to):
    if to:
        return f"오류: 시험이 {CFG['test_timeout']}초 안에 끝나지 않아 중단했다(끝나지 않는 반복이 있을 수 있다)"
    res, ran = parse(out)
    ok = sum(v == "ok" for v in res.values())
    if code == 0 and ran > 0:
        return f"PASS ({ok}/{ran})"
    if ran == 0:
        return "FAIL — 실행된 시험이 없다\n" + out[-400:]
    return f"FAIL ({ok}/{max(ran, len(res))} 통과)\n" + fail_log(out)


def make_tools(sb, with_tests=True):
    def run_tests():
        return tests_text(*sb.run_unittest())

    def grep(pattern):
        pat = str(pattern)
        if not pat:
            return "오류: 찾을 글자가 비어 있다"
        hits = []
        for rel in sb.files():
            if rel.endswith(".py"):
                for ln, line in enumerate((sb.root / rel).read_text(encoding="utf-8").splitlines(), 1):
                    if pat in line:
                        hits.append(f"{rel}:{ln}: {line.strip()}")
        return "\n".join(hits[:30]) if hits else f"일치하는 줄이 없다 : {pat!r}"

    def read_file(path):
        try:
            p = sb.resolve(path)
        except ValueError as e:
            return f"오류: {e}"
        if not p.is_file():
            return f"오류: 파일이 없다 : {path}"
        t = p.read_text(encoding="utf-8")
        return t if len(t) <= 6000 else t[:6000] + "\n…(6000자까지만 보여 준다)"

    def edit_file(path, old, new):
        try:
            p = sb.resolve(path)
        except ValueError as e:
            return f"오류: {e}"
        if not p.is_file():
            return f"오류: 파일이 없다 : {path}"
        old, new = str(old), str(new)
        if old == "":
            return "오류: old 가 비어 있다"
        text = p.read_text(encoding="utf-8")
        k = text.count(old)
        if k == 0:
            return "오류: old 와 같은 글자가 파일에 없다(들여쓰기 · 줄바꿈까지 같아야 한다)"
        if k > 1:
            return f"오류: old 가 파일에 {k}번 나온다. 한 번만 나오도록 앞뒤 줄을 더 넣는다"
        if len(text) - len(old) + len(new) > 6000:
            return "오류: 고친 뒤의 파일이 너무 크다"
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"수정함 : {path} (지운 줄 {len(old.splitlines())} · 쓴 줄 {len(new.splitlines())})"

    P = lambda **kw: {"type": "string", **kw}
    tools = [
        Tool("grep", "샌드박스 안의 .py 파일에서 글자 pattern(정규식이 아니라 그대로의 글자)이 들어 있는 줄을 찾아 `파일:줄번호: 줄 내용` 으로 돌려준다. 최대 30줄.",
             {"properties": {"pattern": P(description="찾을 글자 (예 : get_user()")}, "required": ["pattern"]}, grep),
        Tool("read_file", "파일 하나의 전체 내용을 돌려준다. 없는 파일이나 샌드박스 밖의 경로는 오류를 돌려준다.",
             {"properties": {"path": P(description="샌드박스 기준 상대 경로 (예 : payment/users.py)")}, "required": ["path"]}, read_file),
        Tool("edit_file", "파일 안의 글자 old 를 new 로 바꾼다. old 는 파일 안에 정확히 한 번만 나와야 하고(들여쓰기 · 줄바꿈까지 같아야 함) 아니면 오류를 돌려주고 아무것도 바꾸지 않는다. 샌드박스 밖의 경로는 거부한다.",
             {"properties": {"path": P(description="상대 경로 (예 : payment/checkout.py)"), "old": P(description="바꿀 기존 글자"), "new": P(description="새 글자")}, "required": ["path", "old", "new"]}, edit_file),
    ]
    if with_tests:
        tools.insert(0, Tool("run_tests", "프로젝트의 시험 전체를 python -m unittest 로 실행한다. 모두 통과하면 `PASS (통과/전체)`, 하나라도 깨지면 `FAIL` 과 깨진 시험의 이름 · 오류 기록을 돌려준다. 인자는 없다.",
                             {"properties": {}}, run_tests))
    return tools


# 손으로 불러 본다
with Sandbox(project_files("게스트")) as sb:
    T = {t.name: t for t in make_tools(sb)}
    print("모델에게 보이는 도구 이름 :", list(T))
    print("\n① run_tests() →\n" + T["run_tests"].fn())
    print("\n② grep('get_user(') →\n" + T["grep"].fn("get_user("))
    print("\n③ read_file('payment/checkout.py') →\n" + T["read_file"].fn("payment/checkout.py"))
    print("④ 거부되는 경우들")
    print("  read_file('../../etc/passwd') →", T["read_file"].fn("../../etc/passwd"))
    print("  read_file('/etc/passwd')      →", T["read_file"].fn("/etc/passwd"))
    print("  edit_file(old 가 여러 번)     →", T["edit_file"].fn("payment/checkout.py", "user", "u"))
    print("  edit_file(old 가 없음)        →", T["edit_file"].fn("payment/checkout.py", "없는 글자", "x"))
    B = BUGS["게스트"]
    print("\n⑤ edit_file(정답 패치) →", T["edit_file"].fn(B["file"], B["bad"], B["good"]))
    print("⑥ run_tests() →", T["run_tests"].fn())

# 시간 제한 : 끝나지 않는 시험
with Sandbox({"tests/test_loop.py": "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_forever(self):\n        while True:\n            pass\n"}) as sl:
    t0 = time.time()
    code, out, to = sl.run_unittest(timeout=2)
print(f"\n⑦ 끝나지 않는 시험에 시간 제한 2초 → 시간 초과 {to} · 걸린 시간 {time.time() - t0:.1f}초 · 도구가 돌려주는 글 : {tests_text(code, out, to)[:60]}")
''')

# ───────────────────────── 셀 4 ─────────────────────────
b.md(r'''
### 셀 4 · 채점기 점검 — 숨은 확인이 맞게 판정하는지 먼저 시험한다
- 모델을 돌리기 전에 채점기를 시험한다. 채점기가 틀리면 아래의 모든 숫자가 틀린다
- **숨은 확인** : 고친 `payment/` 를 새 폴더로 옮기고, 거기에 **원래 시험 파일 12개 + 숨은 시험 2개**를 놓고 돌린다. 판정은 세 가지다
  - **통과** : 14개 모두 통과
  - **부분** : 원래 깨지던 시험은 모두 통과했지만 다른 시험(전에 통과하던 것 또는 숨은 시험)이 깨진다 — 버그는 고쳤지만 다른 것을 망가뜨렸거나 규칙을 어긴 경우
  - **실패** : 원래 깨지던 시험이 아직 깨진다(또는 시험이 돌지 않는다 · 시간 초과)
- 아래 표는 버그마다 같은 프로젝트에 ① 버그 그대로 ② 정답 패치 ③ 엉뚱한 곳 수정 ④ 증상만 가린 수정 ⑤(게스트) 시험 파일을 고쳐 통과시키기 ⑥(게스트) 문법 오류가 남은 수정 을 **직접** 넣어 보고, 기대한 판정과 맞는지 본다. 하나라도 어긋나면 이 셀이 멈춘다
''')
b.code(r'''
# 셀 4 · 채점기 점검
def hidden_check(sb, bug):
    """고친 payment/ 를 새 폴더로 옮겨 원래 시험 + 숨은 시험으로 판정한다. 시험 파일은 모델이 고친 것이 아니라 원본을 쓴다"""
    with Sandbox({**TESTS, "tests/test_hidden.py": BUGS[bug]["hidden"]}) as ck:
        shutil.copytree(sb.root / "payment", ck.root / "payment", ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        code, out, to = ck.run_unittest()
    res, ran = parse(out)
    passed = {k for k, v in res.items() if v == "ok"}
    if not to and code == 0 and passed >= ALL_IDS[bug]:
        verdict = "통과"
    elif not to and TARGETS[bug] <= passed:
        verdict = "부분"
    else:
        verdict = "실패"
    return {"판정": verdict, "통과 시험": len(passed & ALL_IDS[bug]), "전체 시험": len(ALL_IDS[bug]), "깨진 시험": sorted(k.split(".", 1)[1] for k in ALL_IDS[bug] - passed)}


def good_edit(bug):
    B = BUGS[bug]
    return [(B["file"], B["bad"], B["good"])]


TAMPER = [("tests/test_checkout.py", '        r = checkout({}, [("키보드", 59000, 1)])\n        self.assertEqual(r["total"], 59000)\n', "        pass\n"),
          ("tests/test_checkout.py", '        r = checkout({}, [("마우스", 10000, 12)])\n        self.assertEqual(r["total"], 116400)\n', "        pass\n")]
CASES = {
    "게스트": [("버그 그대로", [], "실패"), ("정답 패치", good_edit("게스트"), "통과"),
             ("엉뚱한 곳 수정(실버 할인율)", [("payment/discount.py", '"silver": 5', '"silver": 6')], "실패"),
             ("증상만 가림(게스트 user_id 를 0 으로)", [("payment/checkout.py", BUGS["게스트"]["bad"], "    uid = user.id if user else 0\n    grade = user.grade if user else \"guest\"\n")], "부분"),
             ("시험 파일을 고쳐 통과시킴", TAMPER, "실패"),
             ("문법 오류가 남은 수정", [("payment/checkout.py", BUGS["게스트"]["bad"], BUGS["게스트"]["good"] + "    (\n")], "실패")],
    "경계값": [("버그 그대로", [], "실패"), ("정답 패치", good_edit("경계값"), "통과"),
             ("엉뚱한 곳 수정(골드 할인율)", [("payment/discount.py", '"gold": 10', '"gold": 12')], "실패"),
             ("경계를 한 칸 넓게 잡음(9개부터 할인)", [("payment/discount.py", "qty > BULK_MIN", "qty >= BULK_MIN - 1")], "부분")],
    "반올림": [("버그 그대로", [], "실패"), ("정답 패치", good_edit("반올림"), "통과"),
             ("엉뚱한 곳 수정(실버 할인율)", [("payment/discount.py", '"silver": 5', '"silver": 6')], "실패"),
             ("올림으로 바꿈", [("payment/discount.py", BUGS["반올림"]["bad"], "    return -((-amount * (100 - rate)) // 100)\n")], "부분"),
             ("파이썬 round 사용(0.5 는 짝수로)", [("payment/discount.py", BUGS["반올림"]["bad"], "    return round(amount * (100 - rate) / 100)\n")], "실패")],
}
rows = []
for bug, cases in CASES.items():
    for name, edits, want in cases:
        with Sandbox(project_files(bug)) as sb:
            ed = {t.name: t for t in make_tools(sb)}["edit_file"].fn
            for path, old, new in edits:
                r = ed(path, old, new)
                assert r.startswith("수정함"), (bug, name, r)
            chk = hidden_check(sb, bug)
        rows.append({"버그": bug, "넣은 수정": name, "기대": want, "판정": chk["판정"], "통과 시험": f"{chk['통과 시험']}/{chk['전체 시험']}", "일치": chk["판정"] == want,
                     "깨진 시험": ", ".join(chk["깨진 시험"])[:70]})
GR = pd.DataFrame(rows)
print(GR.to_string(index=False))
assert GR["일치"].all(), "채점기가 기대와 다르게 판정했다 — 위 표에서 일치가 False 인 줄을 본다"
print(f"\n채점기 점검 : {len(GR)}건 모두 기대한 판정과 일치 (통과 {int((GR['판정'] == '통과').sum())} · 부분 {int((GR['판정'] == '부분').sum())} · 실패 {int((GR['판정'] == '실패').sum())})")
''')

# ───────────────────────── 셀 5 ─────────────────────────
b.md(r'''
### 셀 5 · 한 궤적 따라가기
- `run_agent` 로 게스트 버그를 고치게 한다(온도 0). 끝난 뒤 궤적을 강의 장표와 같은 모양으로 출력한다. **Thought** = 그 스텝에서 모델이 쓴 글(`assistant.content`), **Action** = 도구 호출, **Observation** = 도구 결과(시험 로그가 여기에 들어온다)
- 모델이 글 없이 바로 호출만 하면 본문 Thought 는 비어 있다. 생각 토큰(`reasoning_content`)으로 추론하는 모델이면 그 앞부분을 「(생각)」으로 대신 보여 주고, 둘 다 없으면 비어 있다고 적는다
- 그림은 `draw_trajectory` 다. 붉은 칸은 `FAIL` 이거나 오류인 관찰이다
- 끝난 뒤 **숨은 확인**으로 코드가 성공을 채점한다. 모델이 「고쳤다」고 한 것과 채점 결과가 같은지 본다
- 이 셀에서 한 번의 실행을 감싸는 함수 `run_one` 도 만든다(샌드박스 만들기 → 도구 → 루프 → 숨은 확인 → 지표). 아래 셀들이 이 함수를 쓴다
''')
b.code(r'''
# 셀 5 · 한 궤적 따라가기
SYSTEM_06 = ("너는 파이썬 프로젝트의 버그를 고치는 코딩 도우미다. 프로젝트는 샌드박스 안에 있고(payment/ 와 tests/), 도구로만 파일을 읽고 고칠 수 있다. "
             "원인이 있는 곳만 최소한으로 고치고 시험 파일은 고치지 않는다. 확인되지 않은 것은 지어내지 않는다. 끝낼 때는 무엇을 고쳤는지 한두 문장으로 답한다.")


def system_for(tests):
    return SYSTEM_06 + (" 수정한 뒤에는 run_tests 로 확인하고, 시험이 모두 통과하면 끝낸다." if tests else " 이 환경에서는 시험을 실행할 수 없다. 코드를 읽고 판단해 고친다.")


def run_gated(client, system, user, tools, max_steps=8, temperature=0.0, seed=0, max_tokens=512, sample=0, max_obs_chars=1500):
    """run_agent 와 같은 루프. 다른 점은 하나 : run_tests 가 PASS 를 돌려주면 모델이 답하기 전에 코드가 끝낸다(종료 조건 = 시험 통과)"""
    trace = Trace()
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
        passed = False
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
                except Exception as e:
                    obs, err = f"오류: {type(e).__name__}: {e}", "도구 오류"
            obs = obs[:max_obs_chars]
            trace.add(f"도구 {c['name']}", "tool", time.perf_counter() - t0, stage=trace.new_stage())
            st["calls"].append({"name": c["name"], "args": c["arguments"], "obs": obs, "error": err})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
            passed = passed or (c["name"] == "run_tests" and obs.startswith("PASS"))
        if passed:
            final, stopped = "(시험 통과 — 코드가 루프를 끝냈다)", "final"
            break
    traj = Trajectory(steps, final, stopped, trace, msgs)
    traj.ended_by = "코드" if passed else ("모델" if stopped == "final" else "스텝 소진")
    return traj


def analyze(traj, sb, bug, base, tests, gate, i):
    flat = [(st["i"], c) for st in traj.steps for c in st["calls"]]
    ok = lambda c: not c["obs"].startswith("오류")
    t_idx = [k for k, (_, c) in enumerate(flat) if c["name"] == "run_tests"]
    after = flat[t_idx[0] + 1:] if t_idx else flat           # 첫 run_tests 이후(시험이 없는 세트는 처음부터)
    norm = lambda c: os.path.normpath(str(c["args"].get("path", "")))
    seen = {norm(c) for _, c in after if c["name"] == "read_file" and ok(c)}
    seen_all = {norm(c) for _, c in flat if c["name"] == "read_file" and ok(c)}
    log_files = set(re.findall(r"payment/\w+\.py", flat[t_idx[0]][1]["obs"])) if t_idx else set()     # 첫 시험 로그에 이름이 나온 소스 파일
    changed, touched = 0, False
    for rel, text in base.items():
        cur = (sb.root / rel).read_text(encoding="utf-8")
        if cur != text:
            if rel.startswith("tests/"):
                touched = True
            else:
                changed += sum(1 for ln in difflib.ndiff(text.splitlines(), cur.splitlines()) if ln[:2] in ("+ ", "- "))
    pass_step = next((s for s, c in flat if c["name"] == "run_tests" and c["obs"].startswith("PASS")), None)
    chk = hidden_check(sb, bug)
    ended = getattr(traj, "ended_by", None) or ("모델" if traj.stopped == "final" else "스텝 소진")
    return {"버그": bug, "시험 도구": tests, "종료 루프": "코드" if gate else "모델", "i": i, "판정": chk["판정"], "통과 시험": chk["통과 시험"], "전체 시험": chk["전체 시험"],
            "스텝": traj.n_steps, "토큰": traj.trace.tokens, "지연(초)": round(traj.trace.latency, 1), "끝낸 쪽": ended,
            "시험 실행": len(t_idx), "본 파일(첫 시험 뒤)": len(seen), "읽은 파일(전체)": len(seen_all),
            "로그의 파일을 읽음": bool(log_files & seen_all) if log_files else None,
            "수정 횟수": sum(1 for _, c in flat if c["name"] == "edit_file" and c["obs"].startswith("수정함")), "수정 줄": changed,
            "최소 수정": bool(0 < changed <= CFG["min_factor"] * REF_LINES[bug]), "시험 파일 수정": touched,
            "PASS 본 적": pass_step is not None, "통과 뒤 LLM 호출": (traj.n_steps - pass_step) if pass_step else 0,
            "거짓 완료": bool(ended in ("모델", "코드") and chk["판정"] != "통과"), "깨진 시험": ", ".join(chk["깨진 시험"])}


def run_one(bug, tests=True, gate=False, i=0, temperature=None):
    """한 번의 실행 : 새 샌드박스 → 도구 → 루프 → 숨은 확인 → 지표. 샌드박스는 끝나면 지운다"""
    temperature = CFG["temperature"] if temperature is None else temperature
    base = project_files(bug)
    with Sandbox(base) as sb:
        runner = run_gated if gate else run_agent
        traj = runner(llm, system_for(tests), BUGS[bug]["task"], make_tools(sb, tests), max_steps=CFG["max_steps"], temperature=temperature, seed=CFG["seed"] + i, max_tokens=CFG["max_tokens"], sample=i)
        row = analyze(traj, sb, bug, base, tests, gate, i)
    row["traj"] = traj
    return row


def show_tao(traj, goal, obs_chars=300):
    """궤적을 강의 장표와 같은 모양으로 출력한다"""
    print("[목표]", goal)
    for st in traj.steps:
        th = re.sub(r"\s+", " ", st["thought"]).strip()
        if not th and st.get("reasoning"):                      # 모델이 글 대신 생각 토큰으로 추론했으면 그 앞부분을 Thought 로 보여 준다
            th = "(생각) " + re.sub(r"\s+", " ", st["reasoning"]).strip()[:160]
        th = th or "(글 없이 바로 호출)"
        if not st["calls"]:
            print(f"Thought: {th}\n  → 도구 호출 없음 = 최종 답으로 끝")
            continue
        for j, c in enumerate(st["calls"]):
            a = ", ".join(f"{k}={re.sub(chr(10), '⏎', str(v))[:48]!r}" for k, v in c["args"].items())
            print(f"Thought: {th if j == 0 else '(같은 스텝의 호출)'}   Action: {c['name']}({a})")
            o = c["obs"][:obs_chars] + ("…" if len(c["obs"]) > obs_chars else "")
            print("Observation:", o.replace("\n", "\n             "))


def tidy(traj):
    """그림용으로 줄인 사본 — 한 줄에 들어가도록 글을 줄이고, 관찰은 도구 칸의 「결과」로 합친다. FAIL · 오류인 칸은 붉게"""
    short = lambda v, n: (lambda s: s if len(s) <= n else s[:n - 1] + "…")(re.sub(r"\s+", " ", str(v)).strip())
    steps = []
    for st in traj.steps:
        calls = []
        for c in st["calls"]:
            a = c["args"]
            args = {"path": Path(str(a.get("path", ""))).name} if c["name"] in ("read_file", "edit_file") else ({"pattern": short(a.get("pattern", ""), 12)} if c["name"] == "grep" else {})
            args["결과"] = short(c["obs"].split("\n")[0], 16)
            bad = c["obs"].startswith(("오류", "FAIL"))
            calls.append({**c, "args": args, "obs": "", "error": "실패" if bad else ""})
        steps.append({**st, "thought": short(st["thought"], 40), "calls": calls})
    return dataclasses.replace(traj, steps=steps, final=short(traj.final, 40))


ONE, ONE_ROW = None, None
try:
    ONE_ROW = run_one("게스트", True, i=CFG["trace_sample"], temperature=CFG["trace_temperature"])
    ONE = ONE_ROW["traj"]
    show_tao(ONE, BUGS["게스트"]["task"])
    print(f"\n끝난 이유 : {ONE.stopped} (끝낸 쪽 : {ONE_ROW['끝낸 쪽']}) · 호출한 도구 : {ONE.calls}")
    print(f"통계 : {ONE.trace.stats()} · 온도 {CFG['trace_temperature']} · 시드 {CFG['seed']}")
    print(f"\n숨은 확인 : {ONE_ROW['판정']} ({ONE_ROW['통과 시험']}/{ONE_ROW['전체 시험']})" + (f" · 깨진 시험 : {ONE_ROW['깨진 시험']}" if ONE_ROW["깨진 시험"] else ""))
    print(f"바꾼 줄 {ONE_ROW['수정 줄']} (정답 패치 {REF_LINES['게스트']}줄) · 최소 수정 {ONE_ROW['최소 수정']} · 시험 파일 수정 {ONE_ROW['시험 파일 수정']}")
    rows_n = 1 + sum(1 + len(st["calls"]) for st in ONE.steps)
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * rows_n + 0.9))
    draw_trajectory(tidy(ONE), title=f"게스트 버그 · 궤적 하나 (숨은 확인 : {ONE_ROW['판정']})", ax=ax, show_obs=False)
    plt.tight_layout(); plt.show()
except (LLMUnavailable, CacheMiss) as e:
    print("이 셀은 LM Studio 서버(또는 강사 캐시)가 있어야 돈다.\n", e, "\n→ 셀 6 ~ 9 는 안내만 내고 넘어간다. 셀 2 ~ 4 의 결과는 그대로 읽을 수 있다.")
''')

# ───────────────────────── 셀 6 ─────────────────────────
b.md(r'''
### 셀 6 · 실패 로그가 수정 범위를 좁히는가(반성)
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
- 같은 게스트 과제를 `n_runs` 번 돌린다(표본 번호 0, 1, … · 온도 `temperature`). 한 번마다 새 샌드박스다
- 측정하는 것
  - **본 파일 수(첫 시험 뒤)** : 첫 `run_tests` 이후 `read_file` 로 연 서로 다른 파일 수. 프로젝트의 소스는 4개다. `grep` 결과에 이름만 나온 파일은 세지 않는다
  - **로그의 파일을 읽음** : 첫 시험 로그(Traceback)에 이름이 나온 소스 파일을 실제로 열었는가
  - **수정 횟수** : 성공한 `edit_file` 호출 수. **수정 줄** : 버그가 있던 원래 소스와 비교해 지워진 줄 + 새로 쓴 줄(정답 패치는 줄 수가 `REF_LINES` 로 정해져 있다)
  - **최소 수정** : 바꾼 줄이 1 이상이고 정답 패치 줄 수의 `min_factor` 배 이하
- 한계 : 이 지표들은 수정이 얼마나 컸는지를 말해 줄 뿐 옳은지는 말하지 않는다(옳은지는 숨은 확인의 판정). 그리고 시험 로그 때문에 범위가 좁아졌는지 그냥 그랬는지는 이 셀만으로는 가를 수 없다. 로그가 없는 조건과의 비교는 셀 7 에서 한다
- 여러 번의 궤적을 `draw_trajectory_flow` 로 겹쳐 그린다. 같은 입력이어도 갈래가 갈라지는지 본다
''')
b.code(r'''
# 셀 6 · 실패 로그가 수정 범위를 좁히는가
def bars_e(ax, labels, means, stds, colors, title, fmt="{:.1f}"):
    """막대 + 표준편차 막대. 값 글자는 오차 막대 위에 둔다(막대 선이 글자를 가로지르지 않게)"""
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, ec="#444", lw=0.8)
    top = max(m + s for m, s in zip(means, stds)) or 1
    for xi, m, sd in zip(x, means, stds):
        ax.text(xi, m + sd + 0.02 * top, fmt.format(m), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, top * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")


def vc(xs):
    return {k: int(v) for k, v in pd.Series(list(xs)).value_counts().items()}


def table(rows, cols):
    return pd.DataFrame([{k: r[k] for k in cols} for r in rows])


def summ(rows):
    """실행 묶음의 요약 — 비율은 float, 평균 · 표준편차는 따로"""
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "traj"} for r in rows])
    f = lambda col: (float(d[col].mean()), float(d[col].std(ddof=0)))
    return {"실행 수": len(d), "통과율": float((d["판정"] == "통과").mean()), "부분율": float((d["판정"] == "부분").mean()), "실패율": float((d["판정"] == "실패").mean()),
            "스텝": f("스텝"), "토큰": f("토큰"), "거짓 완료율": float(d["거짓 완료"].mean()), "최소 수정률": float(d["최소 수정"].mean()),
            "본 파일(첫 시험 뒤)": f("본 파일(첫 시험 뒤)"), "수정 횟수": f("수정 횟수"), "수정 줄": f("수정 줄"), "시험 실행": f("시험 실행"),
            "시험 파일 수정률": float(d["시험 파일 수정"].mean()), "PASS 본 적 비율": float(d["PASS 본 적"].mean()), "통과 뒤 LLM 호출": f("통과 뒤 LLM 호출")}


COLS6 = ["i", "판정", "스텝", "토큰", "시험 실행", "본 파일(첫 시험 뒤)", "로그의 파일을 읽음", "수정 횟수", "수정 줄", "최소 수정", "끝낸 쪽", "거짓 완료"]
RA = None
if ONE is None:
    print("셀 5 가 끝나지 않아 이 셀은 넘어간다.")
else:
    try:
        t0 = time.time()
        RA = pmap(lambda i: run_one("게스트", True, i=i), range(CFG["n_runs"]))        # 표본마다 새 샌드박스 · 서로 독립 → 겹쳐 돈다(결과는 표본 번호 순서)
        print(f"게스트 버그 · run_tests 있음 · {len(RA)}번 (온도 {CFG['temperature']} · 시드 {CFG['seed']} · 표본 0..{len(RA) - 1}) · {time.time() - t0:.0f}초 · 서버 호출 {llm.n_live}번 · 캐시 {llm.n_cache}번")
        print(table(RA, COLS6).to_string(index=False))
        SA = summ(RA)
        print(f"\n정답 패치 {REF_LINES['게스트']}줄, 최소 수정의 상한 {CFG['min_factor'] * REF_LINES['게스트']}줄")
        print(f"본 파일(첫 시험 뒤) {ms([r['본 파일(첫 시험 뒤)'] for r in RA])} / 소스 4개 · 수정 횟수 {ms([r['수정 횟수'] for r in RA])} · 수정 줄 {ms([r['수정 줄'] for r in RA])} · 시험 실행 {ms([r['시험 실행'] for r in RA])}")
        print(f"최소 수정 비율 {SA['최소 수정률']:.2f} · 로그의 파일을 읽은 비율 {np.mean([bool(r['로그의 파일을 읽음']) for r in RA]):.2f} · 숨은 확인 통과 {SA['통과율']:.2f} · 부분 {SA['부분율']:.2f} · 실패 {SA['실패율']:.2f}")
        fig = plt.figure(figsize=(13, 4.2))
        a1 = fig.add_axes([0.04, 0.12, 0.40, 0.75])
        draw_trajectory_flow([r["traj"] for r in RA], ax=a1, title="게스트 버그 · 겹친 궤적")
        xs = np.arange(len(RA))
        ax = [fig.add_axes([0.52, 0.12, 0.13, 0.7]), fig.add_axes([0.70, 0.12, 0.13, 0.7]), fig.add_axes([0.88, 0.12, 0.10, 0.7])]
        ax[0].bar(xs, [r["본 파일(첫 시험 뒤)"] for r in RA], color=COLOR["code"], ec="#444", lw=0.8); ax[0].axhline(4, color="#bbb", ls=":"); ax[0].set_title("본 파일 수(점선 = 소스 4개)", fontsize=9, loc="left")
        ax[1].bar(xs, [r["수정 줄"] for r in RA], color=[COLOR["tool"] if r["최소 수정"] else COLOR["bad"] for r in RA], ec="#444", lw=0.8)
        ax[1].axhline(REF_LINES["게스트"], color="#3E7A3A", ls="--", lw=1); ax[1].axhline(CFG["min_factor"] * REF_LINES["게스트"], color="#B91C1C", ls=":", lw=1)
        ax[1].set_title("수정 줄(초록선 정답 패치 · 빨강선 상한)", fontsize=9, loc="left")
        ax[2].bar(xs, [r["수정 횟수"] for r in RA], color=COLOR["llm"], ec="#444", lw=0.8); ax[2].set_title("수정 횟수", fontsize=9, loc="left")
        for a in ax:
            a.set_xticks(xs); a.set_xticklabels([f"{i}" for i in xs], fontsize=8); a.set_xlabel("표본 번호", fontsize=8)
        plt.show()
    except (LLMUnavailable, CacheMiss) as e:
        RA = None
        print("이 셀은 LM Studio 서버(또는 강사 캐시)가 있어야 돈다.\n", e)
''')

# ───────────────────────── 셀 7 ─────────────────────────
b.md(r'''
### 셀 7 · 검증 신호가 없으면
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
- 같은 게스트 과제 · 같은 `n_runs` · 같은 온도로 도구 세트에서 `run_tests` 만 뺀다(`grep` · `read_file` · `edit_file`). 시스템 지시도 「시험을 실행할 수 없다」로 바뀐다. 모델은 코드를 읽고 판단해 고치고 스스로 끝낸다
- 비교하는 것 : 숨은 확인 통과율 · 스텝 수 · **거짓 완료**(모델이 스스로 끝냈는데 숨은 확인이 통과가 아닌 비율)
- 「검증 신호가 있으면 루프가 스스로 수렴한다」가 숫자로 나오는지 본다. 표본이 작으므로 차이가 우연의 폭 안인지도 함께 본다. 시험이 있는 쪽에도 거짓 완료가 남을 수 있다 — 보이는 시험이 숨은 시험보다 약하기 때문이다. 있으면 있는 그대로 남긴다
''')
b.code(r'''
# 셀 7 · 검증 신호가 없으면
RB = None
if RA is None:
    print("셀 6 이 끝나지 않아 이 셀은 넘어간다.")
else:
    try:
        t0 = time.time()
        RB = pmap(lambda i: run_one("게스트", False, i=i), range(CFG["n_runs"]))       # 표본마다 새 샌드박스 · 서로 독립 → 겹쳐 돈다(결과는 표본 번호 순서)
        print(f"게스트 버그 · run_tests 없음 · {len(RB)}번 (온도 {CFG['temperature']} · 시드 {CFG['seed']}) · {time.time() - t0:.0f}초 · 서버 호출 {llm.n_live}번 · 캐시 {llm.n_cache}번")
        print(table(RB, ["i", "판정", "스텝", "토큰", "본 파일(첫 시험 뒤)", "수정 횟수", "수정 줄", "최소 수정", "끝낸 쪽", "거짓 완료"]).to_string(index=False))
        S = {"run_tests 있음": summ(RA), "run_tests 없음": summ(RB)}
        cmp = pd.DataFrame({k: {"숨은 확인 통과율": v["통과율"], "부분": v["부분율"], "실패": v["실패율"], "거짓 완료율": v["거짓 완료율"],
                                "스텝(평균)": v["스텝"][0], "스텝(표준편차)": v["스텝"][1], "토큰(평균)": v["토큰"][0], "수정 줄(평균)": v["수정 줄"][0]} for k, v in S.items()})
        print("\n" + cmp.to_string(float_format=lambda v: f"{v:.2f}"))
        n_ = CFG["n_runs"]
        print(f"\n표본이 {n_}개씩이라 비율 하나가 1/{n_} = {1 / n_:.2f} 씩 움직인다. 통과율 차이 {S['run_tests 있음']['통과율'] - S['run_tests 없음']['통과율']:+.2f}, 거짓 완료율 차이 {S['run_tests 있음']['거짓 완료율'] - S['run_tests 없음']['거짓 완료율']:+.2f}")
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        lab = ["run_tests\n있음", "run_tests\n없음"]
        colr = [COLOR["tool"], COLOR["bad"]]
        bars(ax[0], lab, [S[k]["통과율"] for k in S], colors=colr, title="숨은 확인 통과율", fmt="{:.2f}"); ax[0].set_ylim(0, 1.15)
        bars_e(ax[1], lab, [S[k]["스텝"][0] for k in S], [S[k]["스텝"][1] for k in S], colr, "스텝 수(평균 ± 표준편차)")
        bars(ax[2], lab, [S[k]["거짓 완료율"] for k in S], colors=colr, title="거짓 완료 비율(끝냈는데 숨은 확인 불통과)", fmt="{:.2f}"); ax[2].set_ylim(0, 1.15)
        plt.tight_layout(); plt.show()
    except (LLMUnavailable, CacheMiss) as e:
        RB = None
        print("이 셀은 LM Studio 서버(또는 강사 캐시)가 있어야 돈다.\n", e)
''')

# ───────────────────────── 셀 8 ─────────────────────────
b.md(r'''
### 셀 8 · 버그 세 종류
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
- 게스트 `None`(강의의 것) · 할인율 경계값(off-by-one) · 반올림 오류. 버그마다 시험 · 정답 패치 · 숨은 시험은 셀 2 에 정의되어 있다. 게스트는 셀 6 의 실행을 그대로 쓰고, 나머지 두 버그를 `n_runs` 번씩 돌린다(모두 `run_tests` 있음 · 같은 온도)
- 표 : 숨은 확인의 통과 · 부분 · 실패 비율, 스텝 수, 토큰, 최소 수정 비율, 거짓 완료 비율. 어느 버그가 더 어려운지 · 스텝이 더 드는지는 숫자를 보고 판단한다
''')
b.code(r'''
# 셀 8 · 버그 세 종류
RALL = None
if RA is None:
    print("셀 6 이 끝나지 않아 이 셀은 넘어간다.")
else:
    try:
        t0 = time.time()
        RALL = {"게스트": RA}
        jobs8 = [(bug, i) for bug in BUG_NAMES[1:] for i in range(CFG["n_runs"])]            # 버그 × 표본 번호 : 모두 서로 독립
        res8 = pmap(lambda j: run_one(j[0], True, i=j[1]), jobs8)                            # 겹쳐 돌되 결과는 입력 순서 그대로
        for bug in BUG_NAMES[1:]:
            RALL[bug] = [r for (b_, _), r in zip(jobs8, res8) if b_ == bug]
        print(f"버그 {len(RALL)}종 × {CFG['n_runs']}번 (온도 {CFG['temperature']} · 시드 {CFG['seed']}) · {time.time() - t0:.0f}초 · 서버 호출 {llm.n_live}번 · 캐시 {llm.n_cache}번")
        SB = {b_: summ(v) for b_, v in RALL.items()}
        T8 = pd.DataFrame({b_: {"정답 패치 줄": REF_LINES[b_], "깨지는 시험(원래)": len(TARGETS[b_]), "통과율": s["통과율"], "부분율": s["부분율"], "실패율": s["실패율"],
                                "스텝(평균)": s["스텝"][0], "스텝(표준편차)": s["스텝"][1], "토큰(평균)": s["토큰"][0], "수정 줄(평균)": s["수정 줄"][0], "최소 수정률": s["최소 수정률"], "거짓 완료율": s["거짓 완료율"]}
                           for b_, s in SB.items()}).T
        print(T8.to_string(float_format=lambda v: f"{v:.2f}"))
        for b_, v in RALL.items():
            print(f"[{b_}] 스텝 {ms([r['스텝'] for r in v])} · 토큰 {ms([r['토큰'] for r in v], 0)} · 판정 {vc([r['판정'] for r in v])}")
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        cc = [COLOR["agent"], COLOR["llm"], COLOR["tool"]]
        bars(ax[0], BUG_NAMES, [SB[b_]["통과율"] for b_ in BUG_NAMES], colors=cc, title="숨은 확인 통과율", fmt="{:.2f}"); ax[0].set_ylim(0, 1.15)
        bars_e(ax[1], BUG_NAMES, [SB[b_]["스텝"][0] for b_ in BUG_NAMES], [SB[b_]["스텝"][1] for b_ in BUG_NAMES], cc, "스텝 수(평균 ± 표준편차)")
        bars_e(ax[2], BUG_NAMES, [SB[b_]["토큰"][0] for b_ in BUG_NAMES], [SB[b_]["토큰"][1] for b_ in BUG_NAMES], cc, "토큰(평균 ± 표준편차)", "{:.0f}")
        plt.tight_layout(); plt.show()
    except (LLMUnavailable, CacheMiss) as e:
        RALL = None
        print("이 셀은 LM Studio 서버(또는 강사 캐시)가 있어야 돈다.\n", e)
''')

# ───────────────────────── 셀 9 ─────────────────────────
b.md(r'''
### 셀 9 · 종료 조건 — 코드가 끝내는 루프 vs 모델이 끝내는 루프
- 서로 독립인 실행은 `pmap` 으로 겹쳐 돈다(동시에 몇 개인지는 서버 설정에 달림). 지연 값은 겹쳐 돌 때 단독 실행보다 길게 나올 수 있다
- **모델 종료 루프** : `run_agent`. 모델이 도구 호출 없이 글만 내면 끝난다. 시스템 지시에 「시험이 모두 통과하면 끝낸다」고 적어 두었을 뿐, 끝낼지는 모델이 정한다
- **코드 종료 루프** : `run_gated`(셀 5). `run_tests` 가 `PASS` 를 돌려주면 모델이 답하기 전에 코드가 끊는다. 종료 조건이 「시험 통과」 그 자체다
- 두 루프는 같은 질문 · 같은 표본 번호를 쓰므로 통과를 보는 시점까지는 같은 응답(같은 궤적)을 쓴다. 그래서 차이는 **통과 뒤**에서만 생긴다 : 통과 뒤 모델이 하는 호출(최종 답 · 추가 행동)의 몫이다. 통과 전에 모델이 스스로 끝낸 경우는 두 루프가 같다
- 비교하는 것 : 스텝 수 · 토큰 · 통과 뒤 LLM 호출 수 · 숨은 확인 통과율 · 거짓 완료(끝냈는데 숨은 확인 불통과). 코드가 끝냈어도 보이는 시험이 약하면 거짓 완료가 남을 수 있다. 세 버그 × `n_runs` 번을 모아 본다
''')
b.code(r'''
# 셀 9 · 종료 조건
GATE = None
if RALL is None:
    print("셀 8 이 끝나지 않아 이 셀은 넘어간다.")
else:
    try:
        t0 = time.time()
        ME = [r for b_ in BUG_NAMES for r in RALL[b_]]
        # 모델 종료 루프(ME)는 셀 8 에서 이미 끝났다. 코드 종료 루프는 그 캐시를 다시 쓰므로 그 뒤에 돌린다(두 루프 사이는 직렬, 각 루프 안은 병렬)
        jobs9 = [(b_, i) for b_ in BUG_NAMES for i in range(CFG["n_runs"])]
        CE = pmap(lambda j: run_one(j[0], True, gate=True, i=j[1]), jobs9)
        GATE = {"모델 종료": ME, "코드 종료": CE}
        print(f"세 버그 × {CFG['n_runs']}번씩 두 루프 (온도 {CFG['temperature']} · 시드 {CFG['seed']}) · {time.time() - t0:.0f}초 · 서버 호출 {llm.n_live}번 · 캐시 {llm.n_cache}번")
        SG = {k: summ(v) for k, v in GATE.items()}
        T9 = pd.DataFrame({k: {"통과율": s["통과율"], "거짓 완료율": s["거짓 완료율"], "스텝(평균)": s["스텝"][0], "스텝(표준편차)": s["스텝"][1], "토큰(평균)": s["토큰"][0], "토큰(표준편차)": s["토큰"][1],
                               "통과 뒤 LLM 호출(평균)": s["통과 뒤 LLM 호출"][0], "PASS 본 적 비율": s["PASS 본 적 비율"]} for k, s in SG.items()})
        print(T9.to_string(float_format=lambda v: f"{v:.2f}"))
        d_step = np.array([c["스텝"] - m["스텝"] for m, c in zip(ME, CE)])
        d_tok = np.array([c["토큰"] - m["토큰"] for m, c in zip(ME, CE)])
        same = sum(1 for m, c in zip(ME, CE) if (m["버그"], m["i"]) == (c["버그"], c["i"]) and m["판정"] == c["판정"])
        print(f"\n같은 (버그, 표본)끼리 짝지은 차이(코드 종료 − 모델 종료) : 스텝 {ms(d_step)} · 토큰 {ms(d_tok, 0)} · 판정이 같은 짝 {same}/{len(ME)}")
        print("끝낸 쪽 :", {k: vc([r['끝낸 쪽'] for r in v]) for k, v in GATE.items()})
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
        lab = ["모델\n종료", "코드\n종료"]
        cc = [COLOR["llm"], COLOR["tool"]]
        bars(ax[0], lab, [SG[k]["통과율"] for k in GATE], colors=cc, title="숨은 확인 통과율", fmt="{:.2f}"); ax[0].set_ylim(0, 1.15)
        bars_e(ax[1], lab, [SG[k]["스텝"][0] for k in GATE], [SG[k]["스텝"][1] for k in GATE], cc, "스텝 수(평균 ± 표준편차)")
        bars_e(ax[2], lab, [SG[k]["토큰"][0] for k in GATE], [SG[k]["토큰"][1] for k in GATE], cc, "토큰(평균 ± 표준편차)", "{:.0f}")
        bars(ax[3], lab, [SG[k]["거짓 완료율"] for k in GATE], colors=cc, title="거짓 완료 비율", fmt="{:.2f}"); ax[3].set_ylim(0, 1.15)
        plt.tight_layout(); plt.show()
    except (LLMUnavailable, CacheMiss) as e:
        GATE = None
        print("이 셀은 LM Studio 서버(또는 강사 캐시)가 있어야 돈다.\n", e)
''')

# ───────────────────────── 셀 10 ─────────────────────────
b.md(r'''
### 셀 10 · 정리 · 결과 저장
- 측정한 숫자를 `results/measured_06.json` 에 저장한다. 서버가 없어 돌지 못한 셀은 「측정하지 않음」으로 적는다
- 마지막으로 임시 폴더를 모두 지우고, 남은 것이 없는지 확인한다
''')
b.code(r'''
# 셀 10 · 정리 · 결과 저장
def clean(d):
    return {k: v for k, v in d.items() if k != "traj"}


NA = "측정하지 않음(서버 없음)"
RES = {"설정": {"모델": llm.model, **CFG, "버그": BUG_NAMES, "정답 패치 줄 수": REF_LINES, "보이는 시험 수": N_VISIBLE, "숨은 확인의 전체 시험 수": {k: len(v) for k, v in ALL_IDS.items()}},
       "채점기 점검": GR.to_dict(orient="records"),
       "한 궤적(온도 0)": ({**clean(ONE_ROW), "호출 순서": ONE.calls} if ONE_ROW else NA),
       "반성(게스트 · run_tests 있음)": ({"요약": summ(RA), "실행": [clean(r) for r in RA]} if RA else NA),
       "검증 신호 없음(게스트)": ({"요약": summ(RB), "실행": [clean(r) for r in RB]} if RB else NA),
       "버그 세 종류": ({b_: summ(v) for b_, v in RALL.items()} if RALL else NA),
       "종료 조건": ({k: summ(v) for k, v in GATE.items()} if GATE else NA)}
save_measured("06", RES)
print(f"LLM 호출 : 서버 {llm.n_live}번 · 캐시 {llm.n_cache}번")

mine = lambda: [d for d in Path(tempfile.gettempdir()).glob(SBX + "*") if (d / ".sbx06").exists()]      # 표식 파일이 있는 우리 폴더만
left_before = len(mine())
for p in list(LIVE):
    shutil.rmtree(p, ignore_errors=True)
LIVE.clear()
removed = sweep_stale()
left = mine()
print(f"임시 폴더 : 정리 전 남은 것 {left_before}개 → 지움 {removed}개 → 지금 남은 것 {len(left)}개")
assert not left, left
print("루프가 스스로 수렴하게 만든 것은 모델이 아니라 시험이 돌려주는 PASS/FAIL 이었는지, 그 신호가 얼마나 믿을 만했는지(보이는 시험 vs 숨은 확인)를 위 숫자에서 확인한다. 해석은 아래 확인 질문에서 숫자를 보고 한다.")
''')

b.md(r'''
### 확인 질문
1. 셀 5 의 궤적에서 Thought · Action · Observation 은 각각 코드의 어느 값인가(`assistant.content` · `tool_calls` · `role: "tool"` 메시지)? 시험 로그가 Observation 으로 들어온 뒤 다음 Action 이 어떻게 달라졌는가? Thought 가 비어 있는 스텝이 있었다면, 그래도 루프가 돌아간 이유는 무엇인가?
2. 셀 6 의 본 파일 수 · 수정 줄을 정답 패치 줄 수와 비교해 보자. 첫 시험 로그에 이름이 나온 파일을 에이전트가 열었는가? 셀 7 에서 `run_tests` 를 뺐을 때 통과율 · 스텝 수 · 거짓 완료가 어떻게 달라졌는가 — 표본이 작다는 점을 감안해 차이를 어디까지 믿을 수 있는지 말해 보자. 시험이 있는데도 거짓 완료가 남았다면 그 이유를 보이는 시험과 숨은 확인의 차이로 설명해 보자.
3. 셀 8 에서 버그 종류에 따라 통과율 · 스텝 수가 달랐는가? 셀 9 에서 코드가 끝내는 루프와 모델이 끝내는 루프의 차이는 어느 구간에서만 생기는가(같은 표본이라 통과 시점까지는 같은 궤적이다)? 종료 조건을 「시험 통과」로 코드에 옮겨도 거짓 완료가 남는다면, 종료 조건의 품질은 무엇이 정하는가?
''')

b.save()
