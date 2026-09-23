# pgvector(PostgreSQL 17) + 한국어 형태소 분석기(mecab-ko + textsearch_ko)
# ------------------------------------------------------------------
# 1단계(builder): mecab-ko, mecab-ko-dic, textsearch_ko 를 소스에서 빌드
# 2단계(최종):    빌드 산출물만 복사해 가벼운 이미지를 만든다
# ------------------------------------------------------------------
ARG PG_MAJOR=17

FROM pgvector/pgvector:pg${PG_MAJOR} AS builder
ARG PG_MAJOR
ARG MECAB_URL=https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
ARG MECAB_DIC_URL=https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
ARG TEXTSEARCH_KO_URL=https://github.com/i0seph/textsearch_ko/archive/refs/heads/master.tar.gz

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool autotools-dev curl ca-certificates \
      postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 1) mecab-ko 엔진
#    소스에 동봉된 config.guess/config.sub 가 2014년판이라 arm64(Apple Silicon, Windows ARM)를 모른다.
#    autotools-dev 의 최신 파일로 덮어써서 amd64/arm64 어디서든 빌드되게 한다.
RUN curl -fsSL "$MECAB_URL" | tar xz \
 && cd mecab-*-ko-* \
 && cp /usr/share/misc/config.guess /usr/share/misc/config.sub . \
 && ./configure --prefix=/usr/local \
 && make -j"$(nproc)" && make install && ldconfig

# 2) mecab-ko-dic 사전 (세종 말뭉치 기반 한국어 사전)
RUN curl -fsSL "$MECAB_DIC_URL" | tar xz \
 && cd mecab-ko-dic-* \
 && ./autogen.sh \
 && cp /usr/share/misc/config.guess /usr/share/misc/config.sub . \
 && ./configure --with-dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic \
 && make && make install

# 3) textsearch_ko: mecab-ko 를 PostgreSQL 전문검색 파서/사전으로 연결하는 익스텐션
RUN curl -fsSL "$TEXTSEARCH_KO_URL" | tar xz \
 && cd textsearch_ko-* \
 && make USE_PGXS=1 && make USE_PGXS=1 install

# ------------------------------------------------------------------
FROM pgvector/pgvector:pg${PG_MAJOR}
ARG PG_MAJOR

# mecab 라이브러리·사전·설정 파일
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/etc/mecabrc /usr/local/etc/mecabrc
COPY --from=builder /usr/local/bin/mecab /usr/local/bin/mecab
# textsearch_ko 익스텐션 파일
COPY --from=builder /usr/lib/postgresql/${PG_MAJOR}/lib/ts_mecab_ko.so /usr/lib/postgresql/${PG_MAJOR}/lib/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/textsearch_ko* /usr/share/postgresql/${PG_MAJOR}/extension/

RUN ldconfig \
 && echo '형태소 분석기 동작 확인' | mecab | head -3
