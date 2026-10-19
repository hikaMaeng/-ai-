# 이 저장소의 작업 규칙

> **[최우선] 모든 응답 · 중간 진행 보고 · 계획 · 요약 · 질문은 한국어로 쓴다.**
> 영어 설명 문장을 쓰지 않는다. 영어는 코드 · 명령어 · 파일 경로 · 고유명사 · 원문 인용에만 남긴다.
> 긴 작업 도중이나 컨텍스트 요약 뒤에도 같다.

- 강의 장표 원본은 Google Slides. 저장소에는 실습(도커 · 노트북 · 문서)만 둔다
- 실습 환경 : 저장소 루트 `docker-compose.yml` 하나(jupyter · db + 선택 pgadmin · comfy). 이미지는 `docker/`. 자세한 절차는 루트 `README.md`
- Jupyter 컨테이너는 루트 compose 의 `ai4-jupyter` 하나만 쓴다. 강의 폴더에 compose · Dockerfile 을 새로 만들지 않고, `docker run` 으로 Jupyter 컨테이너를 따로 띄우지 않는다(일괄 실행은 `docker compose exec`)
