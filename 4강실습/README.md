# 4강 실습 — pgvector · 한국어 전문검색 · BM25 직접 구현

> 현대 AI의 원리와 구조 #4-1 벡터디비 실습 · 뉴런데브클래스
> 준비물: **Docker Desktop**(실행 중) 하나. Python·PostgreSQL 설치는 필요 없습니다.
> 소요 시간: **약 50분** (최초 이미지 빌드·데이터셋 다운로드 포함)

```
4강실습/
├─ docker-compose.yml        db · pgadmin · loader 3개 서비스
├─ db/
│  ├─ Dockerfile             pgvector(pg17) + mecab-ko + mecab-ko-dic + textsearch_ko
│  └─ init/                  최초 기동 시 1회: 익스텐션 · 스키마(docs, questions)
├─ pgadmin/                  서버 자동 등록(servers.json) · 비밀번호 파일(pgpass)
├─ loader/load.py            허깅페이스 KLUE-MRC → 형태소 분석 → COPY → 임베딩 → HNSW
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
└─ slides/                   실습 장표(pptx)
```

## 타임테이블 (60분)

| 시간 | 단계 | 하는 일 | 대기 시간 |
|---|---|---|---|
| 00–10 | ① 환경 기동 | `docker compose up -d --build` → 빌드 동안 Dockerfile·compose 설명 | 빌드 **1.5~5분** |
| 10–13 | ② pgAdmin | http://localhost:5050 접속, 쿼리 도구 열기 | – |
| 13–16 | ③ 데이터 적재 | `docker compose run --rm loader` → **"텍스트 적재 완료"** 가 뜨면 바로 ④로 | 로더 빌드+텍스트 **약 1~2분**, 임베딩은 뒤에서 계속(**3~8분**) |
| 16–33 | ④ ts_rank | `10` → `14` | 14번 평가 **약 2분** |
| 33–52 | ⑤ BM25 구현 | `20` → `24` | 21번 구축 **약 30초** |
| 52–60 | ⑥ 종합 비교 | `25` | **약 30초** |

> 실측(캐시 전혀 없는 최초 실행, 12코어 · Docker 메모리 8GB · 약 17MB/s 회선)
> `compose up` 1분 31초 → 로더 이미지 빌드 41초 → 텍스트 적재 완료 **2분 33초** → 임베딩 완료 5분 23초.
> 받는 용량은 약 0.8GB(이미지·파이썬 패키지·데이터셋 15MB·임베딩 모델 220MB).
> 4코어 노트북이나 느린 회선이면 2~3배 걸려도 임베딩은 ④⑤ 를 하는 동안 끝나므로 60분 안에 들어옵니다.

## ① 환경 기동

```bash
git clone https://github.com/hikaMaeng/-ai-.git
cd -ai-/4강실습
docker compose up -d --build
```

- 첫 실행은 이미지를 받고 mecab-ko 를 소스에서 컴파일합니다(12코어 1분 30초, 4코어 약 5분). 두 번째부터는 몇 초.
- Intel/AMD, Apple Silicon, Windows ARM 모두 동작합니다(`db/Dockerfile` 주석 참고).
- 확인:
  ```bash
  docker compose exec db psql -U lab -d lab -c "SELECT to_tsvector('korean', '무궁화꽃이 피었습니다')"
  ```
  `'꽃':2 '무궁화':1 '피':3` 이 나오면 형태소 분석기 정상.

## ② pgAdmin

1. 브라우저로 **http://localhost:5050** (로그인 없음)
2. 왼쪽 트리 **4강실습 → lab (pgvector-ko)** 클릭 (비밀번호 자동)
3. `lab` DB 선택 → 상단 **Query Tool**
4. 실습 SQL 열기: Query Tool 의 **폴더 아이콘(Open File)** → `sql` 폴더 → 파일 선택
5. 블록을 드래그해서 **F5**(선택 영역 실행). 주석의 `관찰:` 과 결과를 비교하며 진행

psql 을 선호하면: `docker compose exec db psql -U lab -d lab -f /sql/11_ts_rank.sql`

## ③ 데이터 적재 — KLUE-MRC

```bash
docker compose run --rm loader
```

- 허깅페이스 [klue/klue](https://huggingface.co/datasets/klue/klue) 의 `mrc` validation (CC BY-SA 4.0)
  - 뉴스(한국경제·아크로팬)와 위키백과 지문 **5,309개**, 질문·정답 **5,841개**
  - 질문마다 정답 지문이 있어서 "검색이 정답을 상위 10위 안에 찾았나"로 품질을 잴 수 있다
- 출력 `✔ 텍스트 적재 완료` 가 뜨면 **이 창은 그대로 두고** ④로 넘어간다. 임베딩은 25번에서만 쓴다.
- 옵션: `--no-embed`(텍스트만) · `--embed-only`(임베딩만) · `--splits train validation`(전체)

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
| `port is already allocated` (5432/5050) | macOS/Linux: `DB_PORT=15432 PGADMIN_PORT=15050 docker compose up -d`<br>PowerShell: `$env:DB_PORT=15432; $env:PGADMIN_PORT=15050; docker compose up -d` |
| pgAdmin 이 안 열림 | 첫 기동은 20~30초 걸린다. `docker compose logs pgadmin` |
| loader 가 `Connection refused` | `docker compose ps` 로 db 가 healthy 인지 확인 |
| 25번에서 벡터 결과가 0 | 임베딩이 아직 진행 중. loader 창의 `완료` 확인 또는 `--embed-only` 재실행 |
| 전부 처음부터 | `docker compose down -v` 후 ①부터 |

> 주의: 실습 도중 `docker compose run --build ...` 처럼 `--build` 를 붙이면 db 컨테이너가 재생성되어
> 실행 중인 쿼리가 끊긴다. 로더는 `docker compose run --rm loader` 로만 실행한다.
