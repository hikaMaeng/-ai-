# 8-1 · 에이전트의 구조와 워크플로우 — 실습

강의 장표(Google Slides)의 2~7번(워크플로우 패턴 · 에이전트 정의)과 20번(코딩 에이전트의 한 궤적)을 손으로 돌려 보는 노트북 6개다.
**같은 문제를 워크플로우로도, 에이전트로도 풀어** 두 방식의 차이를 눈으로 본다. 외부 도구(n8n 등) 없이 노트북 안에서 그림으로 보여 준다.

> 이 폴더의 계획 · 덱 수정안은 [개편안.md](개편안.md).

## 노트북

| 번호 | 노트북 | 보여 주는 것 |
|---|---|---|
| 01 | `01_에이전트_루프_직접_만들기` | 도구 호출 루프(`run_agent`)를 직접 짜고, 한 번 돌 때의 궤적 · 지연 · 컨텍스트 증가를 그림으로 본다 |
| 02 | `02_워크플로우_세_패턴과_에이전트` | 같은 과제를 체이닝 · 라우팅 · 병렬화 워크플로우와 에이전트로 풀어 정확도 · 호출 수 · 지연을 비교한다 |
| 03 | `03_행동_분포_들여다보기` | 같은 입력에서 에이전트의 행동이 매번 어떻게 갈리는지(확률 정책) 표본 빈도로 본다 |
| 04 | `04_복리_오류와_모델_크기` | 단계마다 오류가 곱해지는 p^n 을 가짜 정책으로 실험하고, 모델 크기별 차이를 본다 |
| 05 | `05_자율성의_세_축` | 자율성을 이루는 세 축을 바꿔 가며 결과 · 안전을 비교한다 |
| 06 | `06_코딩_에이전트의_한_궤적` | 작은 프로젝트(테스트 12개)의 버그를 에이전트가 `run_tests` · `grep` · `read_file` · `edit_file` 로 고치는 궤적 |

공통 코드는 `notebooks/agent_lab.py`(LM Studio 호출 · 응답 캐시 · 도구 루프 · 추적 · 그림), 과제 데이터와 채점기는 `notebooks/agent_tasks.py`다.
합성 과제 A~D(공지 요약 체인 · 문의 라우팅 · 회의록 병렬 · 환불 문의 응답)는 시드를 고정하고 규칙으로 채점한다.

## 실행 환경

루트 [README](../README.md)의 통합 환경을 쓴다(강의별 컨테이너를 따로 만들지 않는다).

```bash
cp env.example .env        # 처음 한 번. LAB_GPU 와 아래 LM Studio 값을 고친다
docker compose up -d       # http://localhost:8888 → 8-1/notebooks 를 연다. 커널은 「Python 3 (기본)」
```

### LM Studio

이 실습의 모델은 **호스트 PC 의 LM Studio 서버**(OpenAI 호환 API)가 맡는다. 컨테이너는 GPU 를 쓰지 않는다.

1. LM Studio 에서 도구 호출을 지원하는 모델을 로드하고 Developer 탭에서 서버를 켠다. 모델은 **VRAM 8GB 이내**를 권장한다.
2. 루트 `.env` 에 주소와 모델을 적고 `docker compose up -d` 로 컨테이너를 다시 만든다.

| 변수 | 뜻 | 기본 |
|---|---|---|
| `LMSTUDIO_URL` | 서버 주소. 컨테이너에서는 `host.docker.internal` 로 호스트에 닿는다 (강사 PC 는 포트 12345) | `http://host.docker.internal:1234/v1` |
| `AGENT_MODEL` | LM Studio 의 모델 식별자 | 비우면 노트북 기본값 |
| `AGENT_MODE` | `auto`(캐시에 있으면 재생 · 없으면 호출) · `live`(항상 호출) · `replay`(캐시만, 서버 없이) | `auto` |
| `AGENT_MIN_TOKENS` | 생각을 먼저 쓰는 모델이 답이 빈 채로 잘릴 때 `max_tokens` 하한(예: 4096). 느려진다 | `0` |

서버 점검: `python 8-1/tools/check_lmstudio.py --url http://localhost:12345/v1` — 모델 · 대화 · 도구 호출 · seed 재현 · logprobs 를 확인한다(표준 라이브러리만 쓴다).

응답은 `notebooks/cache/<모델이름>/` 에 저장된다. 같은 모델 이름의 캐시를 받으면 서버 없이 `AGENT_MODE=replay` 로 재생할 수 있다.

### 알려진 제약

- **생각 모델**: 생각 토큰이 `max_tokens` 를 먼저 쓰면 답이 빈 문자열로 온다. 노트북이 잘림 경고를 출력한다. 모델 쪽 생각 끄기 옵션(`reasoning_effort` 등)은 시험한 서버에서 효과가 없었다 → `AGENT_MIN_TOKENS` 로 대응.
- **logprobs 미지원**: 시험한 서버는 응답에 logprobs 가 없다. 03 은 표본 빈도로 분포를 추정한다.
- **06**: `edit_file` 은 문자열 치환이라 들여쓰기 오류가 날 수 있다. 연습용 샌드박스는 악성 코드를 막지 못한다 — 신뢰하는 코드에만 쓴다.
- **측정값 없음**: 실제 모델의 측정 결과는 아직 이 문서에 없다. 아래 「강사용」의 dry 모드 숫자는 의미가 없다.

## 강사용

노트북은 **생성기**(`tools/build_notebook_NN.py`)로 만든다. 셀을 고칠 때는 생성기를 고치고 다시 만든다.
JupyterLab 자동 저장이 원본 `.ipynb` 에 실행 결과를 써 넣으므로, 실행한 뒤에는 생성기로 원본을 복원한다.

```bash
python 8-1/tools/build_notebook_01.py            # 01 다시 만들기 (02~06 도 같은 방식)
bash 8-1/tools/run_notebooks.sh 01 02            # 컨테이너 안에서 nbconvert 실행, 결과는 notebooks/_out/ (gitignore)
AGENT_DRY=1 bash 8-1/tools/run_notebooks.sh 06   # 서버 없이 코드 경로만 시험(가짜 LLM, tools/dry/dry_NN.py)
```

dry 모드: 환경변수 `AGENT_DRY=NN`, 또는 JupyterLab 화면에서 시험할 때 `notebooks/.agent_dry` 파일에 번호(예: `06`)를 적는다. 시험이 끝나면 그 파일을 지운다. 숫자는 의미가 없고 코드 경로와 그림만 확인한다.

## 검증 기록

| 대상 | 방법 | 결과 |
|---|---|---|
| 01~05 | dry nbconvert + 실제 Chrome JupyterLab 전체 실행(통합 환경 이전의 8894 컨테이너) | 오류 0, 그림 렌더 확인(01=3 · 02=7 · 03=5 · 04=5 · 05=8) |
| 06 | dry nbconvert | 오류 0, 채점기 자체 점검 15/15. Chrome 실행은 [이슈 #3](https://github.com/hikaMaeng/-ai-/issues/3) 슬롯에서 별도 기록 |
| 실제 LLM | — | 미측정 |
