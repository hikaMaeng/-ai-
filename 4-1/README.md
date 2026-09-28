# 4강 실습 — pgvector · 한국어 전문검색 · BM25 직접 구현

> 현대 AI의 원리와 구조 #4-1 벡터디비 실습 · 뉴런데브클래스
> 준비물: **Docker Desktop**(실행 중) · **LM Studio**(Qwen3-Embedding-8B 로드, 서버 켬). Python·PostgreSQL 설치는 필요 없습니다.
> 실습 화면은 **Jupyter 노트북**(http://localhost:8888) 하나. 데이터 적재 · 임베딩 서버 호출 · 검색 실험을 전부 노트북에서 한다.
> 4-2 실습도 이 폴더의 docker-compose 로 함께 뜬다 → [4-2](../4-2/)

```
4-1/
├─ docker-compose.yml        db · jupyter · pgadmin 3개 서비스
├─ jupyter/                  4-1·4-2 공통 JupyterLab 이미지 (http://localhost:8888)
├─ notebooks/                실습 노트북 (번호 순서대로). 코드 셀 첫 줄 = "# 셀 N · 제목"
│  ├─ 01_데이터적재_임베딩.ipynb  실습1 KLUE-MRC 적재 → LM Studio 배치 임베딩 → emb128 HNSW
│  ├─ 01b_진행확인.ipynb      실습1 임베딩 진행 건수 확인 (적재 중 다른 탭에서)
│  ├─ 02_코퍼스_형태소.ipynb    실습2 코퍼스 구성 · mecab-ko · tsvector · tsquery
│  ├─ 03_ts_rank.ipynb       실습3 ts_rank · ts_rank_cd · 길이 정규화 · 평가
│  ├─ 04_BM25.ipynb          실습4 BM25 항 분해 · k1 · b · 트리거 · 토큰 수와 IDF
│  ├─ 05_벡터검색.ipynb        실습5 연산자 · HNSW · MRL 차원 · 후보 재정렬 · 종합 평가
│  └─ 06_필터.ipynb          실습6 PostFilter · PreFilter · iterative scan · 부분 인덱스
├─ db/
│  ├─ Dockerfile             pgvector(pg17) + mecab-ko + mecab-ko-dic + textsearch_ko
│  └─ init/                  최초 기동 시 1회: 01 익스텐션 · 02 스키마(docs, questions) · 03 BM25 역색인(bm25_tf · bm25_df 테이블 2 · 트리거 4)
├─ pgadmin/                  서버 자동 등록(servers.json) · 비밀번호 파일(pgpass)
└─ 강의슬라이드_수정사항.md    강의 장표(원본은 Google Slides)의 수정할 문장 (슬라이드 번호별)
```

## 타임테이블

| 단계 | 노트북 | 하는 일 | 계산 시간 (강사 PC 실측) |
|---|---|---|---|
| ① 환경 기동 | – | `docker compose up -d --build` | 빌드 **2~8분** |
| ② 화면 열기 | – | Jupyter http://localhost:8888 · (보조) pgAdmin http://localhost:5050 | – |
| ③ 실습1 | 01 | 데이터 적재 → 임베딩 | 텍스트 약 40초 · 임베딩 약 3시간 |
| ④ 실습2 | 02 | 코퍼스와 형태소 분석 | 수 초 |
| ⑤ 실습3 | 03 | ts_rank 계열 평가 | 약 5분 (셀 9) |
| ⑥ 실습4 | 04 | BM25 | 약 1분 |
| ⑦ 실습5 | 05 | 벡터 검색 (임베딩 완료 후) | 약 1분 30초 |
| ⑧ 실습6 | 06 | 필터 (임베딩 완료 후) | 약 30초 |

> 실측(2026-09-27, Windows · Docker 29.7.2, 캐시 없는 최초 빌드) `compose up --build` 7분 34초.
> 임베딩(Qwen3-Embedding-8B Q4_K_M, LM Studio, 배치 16): 지문 약 0.5~0.6건/s(5,309개 약 2시간 30분~3시간) · 질문 약 5건/s(약 20분).
> 임베딩 속도는 LM Studio 를 돌리는 PC 성능에 따라 크게 달라진다. 실습2~4 는 임베딩 없이 텍스트 적재만으로 할 수 있다.

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

## ② 화면

- **Jupyter** http://localhost:8888 → `4-1` → `notebooks`. 모든 실습은 여기서 한다
  - 노트북 전체 실행 : 메뉴 **Run → Run All Cells** · 셀 하나 : **Shift + Enter**
  - 노트북마다 셀 1 이 설정. 고치는 값은 셀 1 에만 있다
- **pgAdmin** http://localhost:5050 (보조, 로그인 없음) → 왼쪽 트리 **4-1 → lab (pgvector-ko)** → 테이블 내용을 눈으로 확인

## ③ 실습1 · 데이터 적재 — `01_데이터적재_임베딩.ipynb`

준비: LM Studio 에서 **Qwen3-Embedding-8B** 를 로드하고 서버를 켠다.

테이블(docs · questions)은 DB 최초 기동 때 `db/init` 이 만들어 둔다. 노트북은 비우고 채우기만 한다.

| 셀 | 하는 일 |
|---|---|
| 1 | **설정** — `DB_URL` · `LMSTUDIO_URL` · `EMBED_MODEL` · `BATCH_SIZE` · `PAUSE_SEC` · `MAX_EMBED` · `QUERY_TASK`. 고치는 값은 여기에만 |
| 2 | 라이브러리와 DB 연결 |
| 3 | 데이터셋 parquet 가 볼륨(`/cache/hf/parquet`)에 없을 때만 허깅페이스에서 받는다 |
| 4 | 지문 내용의 md5 로 중복을 지워 지문 5,309 · 질문 5,841 표로 정리 |
| 5 | `TRUNCATE … RESTART IDENTITY CASCADE` 후 `INSERT … SELECT FROM unnest(열별 배열)` 두 문장으로 지문·질문 적재. 지문은 CTE 에서 `doc_tsv()` 한 번으로 `tsv`·`doclen` 을 채우고, 질문의 정답 지문은 `doc_key` 조인으로 |
| 6 | LM Studio 모델 목록 출력, `EMBED_MODEL` 로 시험 임베딩(4096차원 확인) |
| 7 | 배치 임베딩 함수: 빈 행 `BATCH_SIZE` 개 → 임베딩 → UPDATE → COMMIT 반복 |
| 8 · 9 | 지문 임베딩(지시문 없음) · 질문 임베딩(`Instruct: … Query:` 지시문) |
| 10 · 11 | `emb128` HNSW 인덱스와 확인 표 · 맛보기 검색(원본 4096 vs emb128) |

- 셀 1 의 `LMSTUDIO_URL` : 컨테이너 안의 localhost 는 컨테이너 자신이므로 `host.docker.internal`. LM Studio 포트가 1234 가 아니면 여기서 바꾼다
- 셀 1 의 `EMBED_MODEL` : LM Studio 설치마다 이름이 다를 수 있다. 셀 6 이 로드된 모델 목록을 보여 주고, 없는 이름이면 멈춘다
- 실행 : 메뉴 **Run → Run All Cells**. 셀 5 의 `✔ 텍스트 적재 완료` 가 뜨면 실습2~4 를 시작해도 된다
- 멈춤 : **Kernel → Interrupt Kernel**. 배치마다 커밋하므로 채운 행은 남는다
- 이어서 : **셀 6** 클릭 → **Run → Run Selected Cell and All Below** (Run All Cells 는 셀 5 에서 테이블을 다시 비운다)
- 진행 확인 : `01b_진행확인.ipynb` (커널이 따로라 임베딩 중에도 실행된다). 탭을 새로 고치면 적재 노트북의 진행 출력은 멈춘 것처럼 보여도 계산은 계속된다
- 허깅페이스 [klue/klue](https://huggingface.co/datasets/klue/klue) 의 `mrc` validation (CC BY-SA 4.0). 뉴스(한국경제·아크로팬)와 위키백과 지문 **5,309개**, 질문·정답 **5,841개**

## 검색 품질 평가 (실습3 · 4 · 5 공통)

질문 5,841개 중 11개마다 하나씩 고른 **500개**로 잰다. 정답 지문이 상위 10위 안이면 Hit, MRR = 정답 순위 역수의 평균.
결과는 `notebooks/results/검색평가.csv` 에 쌓여 뒤 노트북의 비교표에 쓰인다.

| 방법 | Hit@10 | MRR | 노트북 |
|---|---|---|---|
| **BM25** (k1 1.2 · b 0.75) | **0.938** | **0.837** | 04 셀 8 |
| 벡터 4096 전수 비교 | 0.884 | 0.766 | 05 셀 7 |
| emb128 후보 40 → 4096 재정렬 | 0.850 | 0.746 | 05 셀 6 |
| 벡터 emb128 HNSW | 0.804 | 0.660 | 05 셀 7 |
| OR ts_rank_cd | 0.530 | 0.328 | 03 셀 9 |
| OR ts_rank | 0.484 | 0.354 | 03 셀 9 |
| OR 정규화2(길이) | 0.376 | 0.233 | 03 셀 9 |
| AND ts_rank | 0.048 | 0.044 | 03 셀 9 |

> 실측 2026-09-28, Qwen3-Embedding-8B Q4_K_M. 동점 문서는 id 순으로 정렬해 재실행해도 같다.

## ④ 실습2 · 코퍼스와 형태소 — `02_코퍼스_형태소.ipynb`

| 셀 | 핵심 관찰 |
|---|---|
| 3 · 4 | 한국경제 2,982 · 위키백과 1,623 · 아크로팬 704. 지문 184~2,044자(중앙 920자) · 형태소 35~675개(평균 281) |
| 5 | mecab-ko 는 품사 태그(NNG · JKS · VV …)를 붙인다 |
| 6 | `simple` 은 '고양이가' · '고양이' 를 다른 토큰으로 → '고양이'로 첫 문장을 못 찾음. `korean` 은 찾음 |
| 7 | 영문 english_stem · 숫자 simple · 한글 korean_stem. '17에서' 의 '에서' 는 조사인데 형태소로 남는다 |
| 8 · 9 | plainto(AND) · websearch · 구문(<->) 매칭 수. `tsv @@` 는 GIN 인덱스(Bitmap Index Scan) |
| 10 | 문장형 질문의 AND 는 0건, OR 는 1,884건 → 순위 함수가 필요 |

## ⑤ 실습3 · ts_rank — `03_ts_rank.ipynb`

| 셀 | 핵심 관찰 |
|---|---|
| 3 | 가중치 배열 {D, C, B, A}. 제목(A) 가중치를 끄면 상위 5개가 바뀐다 |
| 4 | 같은 단어 n번 : ts_rank 0.061 → 0.094 로 **포화**, ts_rank_cd 는 0.1 × n 으로 선형 |
| 5 | ts_rank_cd 는 AND 질의에서 거리가 멀수록 낮다(0.1 · 0.011 · 0.003). OR 질의에서는 거리 효과가 사라진다 |
| 6 · 7 | 정규화 1(로그) · 2(길이)는 짧은 지문을 끌어올린다. PG 는 코퍼스 평균 길이(avgdl)를 모른다 |
| 8 | OR ts_rank 에서 정답 지문은 2위. 1위는 '에디션' 이 제목에 든 다른 기사 |
| 9 | 위 평가표의 ts_rank 계열 |

## ⑥ 실습4 · BM25 — `04_BM25.ipynb`

BM25 역색인 테이블 · 함수 · 트리거는 `db/init/03_bm25.sql` 이 DB 최초 기동 때 설치한다. 노트북은 쓰고 재기만 한다.

```
BM25(D,Q) = Σ IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))
IDF(t)    = ln(1 + (N − df + 0.5)/(df + 0.5))
```

| 셀 | 핵심 관찰 |
|---|---|
| 3 | N 5,309 · avgdl 281.3. 가장 흔한 '것' IDF 0.236, 한 지문에만 있는 '말라카이트' 8.172 |
| 4 | 정답 지문 점수 24.345 = 말라카이트 12.963 + 에디션 11.382 (bm25_search 와 일치) |
| 6 | ts_rank 2위였던 정답이 BM25 1위 |
| 7 | INSERT · UPDATE(본문) · UPDATE(source) · DELETE 에 df 가 따라온다. 트리거 유지값 = 재계산값 (tf · df · doclen 불일치 0) |
| 8 | k1 · b 를 바꿔도 Hit@10 0.930~0.946 · MRR 0.818~0.838. 지문 길이가 고른 코퍼스라 b 의 영향이 작다 |
| 9 | IDF 를 끄면 MRR 감소 : 질의 형태소 4개 이하 −0.288 · 5~6개 −0.172 · 7~8개 −0.063 · 9개 이상 −0.061 |

## ⑦ 실습5 · 벡터 검색 — `05_벡터검색.ipynb` (임베딩 완료 후)

| 셀 | 핵심 관찰 |
|---|---|
| 3 | 앞 128차원을 자르기만 하면 길이 약 0.2. 코사인 거리는 같고(0.294) 내적 · L2 는 달라진다 → emb128 은 다시 길이 1로 |
| 4 | 원본 4096 은 Seq Scan(질의당 약 60ms), emb128 은 HNSW Index Scan(약 0.7ms) |
| 5 | MRL : 앞 d 차원 전수 비교 Hit@10 — 32 0.642 · 128 0.810 · 512 0.878 · 4096 0.884 |
| 6 | emb128 후보 N → 4096 재정렬. 후보 40 0.850 · 160 0.882. `hnsw.ef_search`(기본 40)보다 많이 달라면 40행만 온다 |
| 8 | BM25 만 맞힌 질문 42 · 벡터만 맞힌 질문 15 · 둘 다 놓침 16 → 합칠 이유 (4-2 퓨전) |
| 9 | LM Studio 로 내 질문을 임베딩해 검색 (셀 1 의 `LMSTUDIO_URL` · `EMBED_MODEL` · `MY_QUERY`) |

## ⑧ 실습6 · 필터 — `06_필터.ipynb` (임베딩 완료 후)

| 셀 | 핵심 관찰 |
|---|---|
| 3 | 출처별로 따로 검색해 합친 상위 10 = 한 테이블 검색 상위 10 (유사도는 절대값) |
| 5 | `WHERE category = … ORDER BY <=> LIMIT 10` 을 HNSW 로 하면 ef_search 40개를 뽑고 거른다 → 분야 비율 13% 에서 3.5행, 1.5% 에서 0.4행. 분야가 아주 작으면(46개 이하) 플래너가 인덱스를 버리고 전수 비교 |
| 6 | PreFilter(`MATERIALIZED` CTE) : 항상 10행 · 정확도 1 |
| 7 | `SET hnsw.iterative_scan = relaxed_order` : 모자라면 더 탐색해 10행 · 정확도 0.98~1.0 |
| 8 | 부분 인덱스(`… WHERE category = '자동차'`) : 조건별 그래프, 10행 · 정확도 1.0 |
| 9 | 세 방식의 행 수 · 정확도 · 질의당 ms (지문 5,309개라 시간 차이는 작다) |

## 문제 해결

| 증상 | 해결 |
|---|---|
| `port is already allocated` (5432/5050/8888) | macOS/Linux: `DB_PORT=15432 PGADMIN_PORT=15050 JUPYTER_PORT=18888 docker compose up -d`<br>PowerShell: `$env:DB_PORT=15432; $env:PGADMIN_PORT=15050; $env:JUPYTER_PORT=18888; docker compose up -d` |
| pgAdmin 이 안 열림 | 첫 기동은 20~30초 걸린다(첫 요청도 몇 초). `docker compose logs pgadmin` 에 `Listening at` 이 보이면 새로고침 |
| `error while creating mount source path … mkdir /run/desktop/mnt/host/<드라이브>: file exists` | 외장 드라이브 등 해당 드라이브를 Docker Desktop 이 못 읽는 상태. Docker Desktop 을 재시작하거나, 저장소를 C: 드라이브로 옮겨 실행 |
| 노트북에서 DB `Connection refused` | `docker compose ps` 로 db 가 healthy 인지 확인 |
| 실습1 셀 6 · 실습5 셀 9 에서 LM Studio 연결 실패 | LM Studio 서버가 켜졌는지 확인. 포트가 1234 가 아니면 셀 1 의 `LMSTUDIO_URL` 을 고친다 |
| 실습5 · 6 셀 2 에서 `실습1 의 지문 임베딩이 다 차야 한다` | 임베딩이 아직 진행 중. `01b_진행확인.ipynb` 로 확인, 멈췄다면 실습1 셀 6 에서 Run Selected Cell and All Below |
| 전부 처음부터 | `docker compose down -v` 후 ①부터 |

> 주의: 실습 도중 `docker compose up -d --build` 를 다시 하면 컨테이너가 재생성되어 실행 중인 노트북이 끊긴다.
