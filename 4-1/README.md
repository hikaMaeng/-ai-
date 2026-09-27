# 4강 실습 — pgvector · 한국어 전문검색 · BM25 직접 구현

> 현대 AI의 원리와 구조 #4-1 벡터디비 실습 · 뉴런데브클래스
> 준비물: **Docker Desktop**(실행 중) · **LM Studio**(Qwen3-Embedding-8B 로드, 서버 켬). Python·PostgreSQL 설치는 필요 없습니다.
> 실습 화면은 **Jupyter 노트북**(http://localhost:8888) 하나. 데이터 적재 · 임베딩 서버 호출 · 검색 실험을 전부 노트북에서 한다.
> 4-2 실습도 이 폴더의 docker-compose 로 함께 뜬다 → [4-2](../4-2/)
>
> 진행 중: ④~⑥ 의 SQL 실습(sql/10~25)을 노트북으로 옮기는 중이다. 옮기기 전까지는 pgAdmin 에서 SQL 파일로 실행한다.

```
4-1/
├─ docker-compose.yml        db · jupyter · pgadmin 3개 서비스
├─ jupyter/                  4-1·4-2 공통 JupyterLab 이미지 (http://localhost:8888)
├─ notebooks/
│  └─ 01_데이터적재_임베딩.ipynb  KLUE-MRC 적재 → LM Studio 배치 임베딩 → emb128 HNSW
├─ db/
│  ├─ Dockerfile             pgvector(pg17) + mecab-ko + mecab-ko-dic + textsearch_ko
│  └─ init/                  최초 기동 시 1회: 익스텐션 · 스키마(docs, questions)
├─ pgadmin/                  서버 자동 등록(servers.json) · 비밀번호 파일(pgpass)
├─ loader/load.py            (서비스 아님) 4-2 의 prepare_klue.py 가 모듈로 불러 쓰는 예전 적재 코드
├─ sql/                      실습 SQL (번호 순서대로)
│  ├─ 10_tokenize.sql        형태소 분석 · tsvector · setweight
│  ├─ 11_ts_rank.sql         가중치 · 포화 · AND/OR 질의
│  ├─ 12_ts_rank_cd.sql      커버 밀집도
│  ├─ 13_normalization.sql   길이 정규화 옵션
│  ├─ 14_eval_ts_rank.sql    Hit@10 · MRR 측정
│  ├─ 20_bm25_tables.sql     역색인 테이블 설계
│  ├─ 21_bm25_build.sql      TF · DF · IDF 일괄 구축
│  ├─ 22_bm25_search.sql     점수 분해 · 검색 함수
│  ├─ 23_bm25_triggers.sql   문장 단위 트리거로 동적 갱신
│  ├─ 24_bm25_trigger_test.sql  INSERT/UPDATE/DELETE 검증
│  └─ 25_eval_all.sql        ts_rank vs BM25 vs 벡터 vs 하이브리드
├─ slides/                   실습 장표(pptx · pdf)
└─ 강의슬라이드_수정사항.md    원본 강의 장표의 수정할 문장 (슬라이드 번호별)
```

## 타임테이블 (60분)

| 시간 | 단계 | 하는 일 | 대기 시간 |
|---|---|---|---|
| 00–10 | ① 환경 기동 | `docker compose up -d --build` → 빌드 동안 Dockerfile·compose 설명 | 빌드 **2~8분** |
| 10–13 | ② 화면 열기 | Jupyter http://localhost:8888 · (보조) pgAdmin http://localhost:5050 | – |
| 13–16 | ③ 데이터 적재 | Jupyter → `4-1/notebooks/01_데이터적재_임베딩.ipynb` → Run → Run All Cells → **"텍스트 적재 완료"** 가 뜨면 ④로 | 텍스트 **약 10초**, 임베딩은 그 노트북에서 뒤로 계속 |
| 16–33 | ④ ts_rank | `10` → `14` | 14번 평가 **약 2분** |
| 33–52 | ⑤ BM25 구현 | `20` → `24` | 21번 구축 **약 30초** |
| 52–60 | ⑥ 종합 비교 | `25` | **약 30초** |

> 실측(2026-09-27, Windows · Docker 29.7.2, 캐시 없는 최초 빌드) `compose up --build` 7분 34초 → 텍스트 적재 9.9초.
> 임베딩(Qwen3-Embedding-8B Q4_K_M, LM Studio, 배치 16): 지문 약 0.6건/s(5,309개 약 2시간 30분) · 질문 약 4.9건/s(약 20분).
> 임베딩 속도는 LM Studio 를 돌리는 PC 성능에 따라 크게 달라진다. 수업 시간 안에 끝나지 않으면 `MAX_EMBED` 로 나눠 채운다.

## ① 환경 기동

```bash
git clone https://github.com/hikaMaeng/-ai-.git
cd ./-ai-/4-1
docker compose up -d --build
```

- 첫 실행은 이미지를 받고 mecab-ko 를 소스에서 컴파일합니다(12코어 1분 30초, 4코어 약 5분). 두 번째부터는 몇 초.
- Jupyter 이미지도 함께 빌드됩니다(캐시 없이 약 1분 30초). 4-1 · 4-2 모두 Jupyter 에서 실습합니다.
- Intel/AMD, Apple Silicon, Windows ARM 모두 동작합니다(`db/Dockerfile` 주석 참고).
- 포트(5432 · 5050 · 8888)는 **내 PC(127.0.0.1)에서만** 열립니다. pgAdmin·Jupyter 는 로그인이 없으므로 같은 Wi-Fi 의 다른 기기가 접속하지 못하게 막아 둔 것입니다.
- 확인:
  ```bash
  docker compose exec db psql -U lab -d lab -c "SELECT to_tsvector('korean', '무궁화꽃이 피었습니다')"
  ```
  `'꽃':2 '무궁화':1 '피':3` 이 나오면 형태소 분석기 정상.

## ② pgAdmin

1. 브라우저로 **http://localhost:5050** (로그인 없음)
2. 왼쪽 트리 **4-1 → lab (pgvector-ko)** 클릭 (비밀번호 자동)
3. `lab` DB 선택 → 상단 **Query Tool**
4. 실습 SQL 열기: Query Tool 의 **폴더 아이콘(Open File)** → `sql` 폴더 → 파일 선택
5. 블록을 드래그해서 **F5**(선택 영역 실행). 주석의 `관찰:` 과 결과를 비교하며 진행

psql 을 선호하면: `docker compose exec db psql -U lab -d lab -f /sql/11_ts_rank.sql`

## ③ 데이터 적재 — KLUE-MRC (Jupyter 노트북)

준비: LM Studio 에서 **Qwen3-Embedding-8B** 를 로드하고 서버를 켠다. 포트가 1234 가 아니면 compose 기동 때 넘긴다.

```bash
LMSTUDIO_URL=http://host.docker.internal:12345/v1 docker compose up -d      # 포트가 12345 인 예 (macOS/Linux)
$env:LMSTUDIO_URL="http://host.docker.internal:12345/v1"; docker compose up -d   # PowerShell
```

http://localhost:8888 → `4-1/notebooks/01_데이터적재_임베딩.ipynb` → **Run → Run All Cells**

| 단계 | 하는 일 |
|---|---|
| 1 | 데이터셋 parquet 가 볼륨(`/cache/hf`)에 없을 때만 허깅페이스에서 받는다 |
| 2 | 스키마가 `embedding vector(4096)` + `emb128 vector(128)` 인지 확인, 예전 볼륨이면 바꾼다 |
| 3 | `TRUNCATE … RESTART IDENTITY CASCADE` 후 지문·질문 텍스트를 COPY. BM25 역색인이 있으면 `bm25_rebuild()` |
| 4 | LM Studio 에서 Qwen3 임베딩 모델을 찾아 4096차원인지 확인 |
| 5 | `embedding IS NULL` 인 행만 `BATCH_SIZE` 개씩 임베딩 → UPDATE → COMMIT. 질문에는 instruction 을 붙인다 |
| 6 | 임베딩이 다 차면 `emb128` 에 HNSW(코사인) 인덱스 |

- 허깅페이스 [klue/klue](https://huggingface.co/datasets/klue/klue) 의 `mrc` validation (CC BY-SA 4.0)
  - 뉴스(한국경제·아크로팬)와 위키백과 지문 **5,309개**, 질문·정답 **5,841개**
  - 질문마다 정답 지문이 있어서 "검색이 정답을 상위 10위 안에 찾았나"로 품질을 잴 수 있다
- 3단계 출력 `✔ 텍스트 적재 완료` 가 뜨면 키워드 검색 실습(④⑤)을 시작해도 된다. 임베딩은 뒤에서 계속 돈다
- 부하 조절: 첫 셀의 `BATCH_SIZE`(한 요청의 문장 수) · `PAUSE_SEC`(배치 사이 쉬는 시간) · `MAX_EMBED`(이번 실행의 최대 건수)
- 중간에 멈춰도 배치마다 커밋되어 있으므로 「4. 임베딩 서버 확인」 제목 셀을 클릭하고 **Run → Run Selected Cell and All Below** 하면 남은 행부터 이어서 채운다 (Run All Cells 는 3번에서 테이블을 다시 비운다)
- 실측(Qwen3-Embedding-8B Q4_K_M, LM Studio 원격 기기, 배치 16): 지문 약 0.6건/s(전체 약 2시간 30분), 질문 약 4.9건/s(약 20분)

## ④ ts_rank 실습 — `sql/10` ~ `sql/14`

| 파일 | 핵심 관찰 |
|---|---|
| 10 | `simple` 설정은 '고양이가'와 '고양이'를 다른 토큰으로 본다. `korean` 은 조사·어미를 떼어낸다 |
| 11 | ts_rank 는 반복 횟수에 **포화**한다(0.061→0.094). 문장형 질문의 AND 질의는 0건 → OR 로 |
| 12 | ts_rank_cd 는 AND 질의에서 거리에 반비례. OR 질의에서는 밀집도 효과가 사라진다 |
| 13 | 정규화 1(로그)과 2(길이)의 차이. PG 는 평균 문서 길이(avgdl)를 모른다 |
| 14 | 질문 500개 Hit@10 : AND 0.05 / OR ts_rank 0.48 / ts_rank_cd 0.52 (결과는 `eval_result` 에 저장) |

## ⑤ BM25 직접 구현 — `sql/20` ~ `sql/24`

```
BM25(D,Q) = Σ IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))
IDF(t)    = ln(1 + (N − df + 0.5)/(df + 0.5))
```

| 파일 | 만드는 것 |
|---|---|
| 20 | `bm25_tf`(역색인) · `bm25_doclen` · `bm25_df` · `bm25_idf` · `bm25_stats` |
| 21 | tsvector 를 `unnest` 해서 전부 채우고 `bm25_refresh_idf()` |
| 22 | 한 문서의 점수를 항별로 분해 → `bm25_search(q, k, k1, b)` 함수 |
| 23 | **문장 단위 트리거 + 전이 테이블**로 INSERT/UPDATE/DELETE 반영, `bm25_rebuild()` |
| 24 | 문서 추가·수정·삭제 후 "트리거 유지값 = 재계산값" 대조 |

## ⑥ 종합 비교 — `sql/25` (임베딩 완료 후)

| 방법 | Hit@10 | MRR |
|---|---|---|
| ts_rank (OR) | 0.484 | 0.354 |
| ts_rank_cd (OR) | 0.524 | 0.329 |
| **BM25 (직접 구현)** | **0.938** | **0.838** |
| 벡터 (MiniLM, 지문 통째) | 0.376 | 0.204 |
| 하이브리드 RRF | 0.896 | 0.523 |

> 동점 문서의 정렬 순서에 따라 ±0.01 정도 달라질 수 있습니다.

## 문제 해결

| 증상 | 해결 |
|---|---|
| `port is already allocated` (5432/5050/8888) | macOS/Linux: `DB_PORT=15432 PGADMIN_PORT=15050 JUPYTER_PORT=18888 docker compose up -d`<br>PowerShell: `$env:DB_PORT=15432; $env:PGADMIN_PORT=15050; $env:JUPYTER_PORT=18888; docker compose up -d` |
| pgAdmin 이 안 열림 | 첫 기동은 20~30초 걸린다(첫 요청도 몇 초). `docker compose logs pgadmin` 에 `Listening at` 이 보이면 새로고침 |
| `error while creating mount source path … mkdir /run/desktop/mnt/host/<드라이브>: file exists` | 외장 드라이브 등 해당 드라이브를 Docker Desktop 이 못 읽는 상태. Docker Desktop 을 재시작하거나, 저장소를 C: 드라이브로 옮겨 실행 |
| 노트북에서 DB `Connection refused` | `docker compose ps` 로 db 가 healthy 인지 확인 |
| 「4. 임베딩 서버 확인」 셀에서 LM Studio 연결 실패 | LM Studio 서버가 켜졌는지, 포트가 1234 가 아니면 `LMSTUDIO_URL` 을 넘겨 다시 `docker compose up -d` |
| 25번에서 벡터 결과가 0 | 임베딩이 아직 진행 중. 적재 노트북의 진행 출력 확인, 멈췄다면 「4. 임베딩 서버 확인」 셀에서 Run Selected Cell and All Below |
| 전부 처음부터 | `docker compose down -v` 후 ①부터 |

> 주의: 실습 도중 `docker compose up -d --build` 를 다시 하면 컨테이너가 재생성되어 실행 중인 노트북·쿼리가 끊긴다.
