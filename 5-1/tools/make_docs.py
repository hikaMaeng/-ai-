"""5-1 실습 02 용 문서 스크린샷 6장 생성기 (강사용).

    python tools/make_docs.py [--font <한글 TTF 경로>]   → ../docs/doc_*.png + docs.csv

공정 · 안전 현장의 서식(시험 기록표 · 허가서 · 점검표 · MSDS 요약 · 교육 일정 · 온도 기록)을
가상의 값으로 그린다. 실제 설비 · 회사와 무관하다. 글꼴은 나눔고딕(SIL OFL)을 기본으로 찾는다.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / 'docs'
W, H = 900, 1100
FONT_CANDIDATES = ['C:/Windows/Fonts/NanumGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                   '/Library/Fonts/NanumGothic.ttf', 'C:/Windows/Fonts/malgun.ttf']

DOCS = [
    ('doc_1_pressure_test.png', '배관 압력 시험 기록표', 'table', {
        'info': [('라인 번호', 'P-1203 (냉각수 공급)'), ('설계 압력', '10 barg'), ('시험 압력', '15 barg (설계의 1.5배)'),
                 ('시험 매체', '물'), ('시험일', '2026-09-14')],
        'head': ['시각', '압력(barg)', '온도(°C)', '비고'],
        'rows': [['09:00', '15.0', '22', '가압 완료'], ['09:10', '15.0', '22', '-'], ['09:20', '14.9', '22', '-'],
                 ['09:30', '15.0', '23', '유지 종료']],
        'foot': ['판정 : 합격 — 30분 유지 중 압력 강하 없음, 누설 없음', '시험자 : 김OO    입회 : 박OO']}),
    ('doc_2_hot_work_permit.png', '화기 작업 허가서', 'form', {
        'info': [('작업 장소', '저장탱크 T-201 상부 배관'), ('작업 내용', '배관 지지대 용접 · 그라인딩'),
                 ('작업 시간', '2026-09-15 13:00 ~ 17:00'), ('작업 인원', '용접공 2명, 화재 감시자 1명')],
        'head': ['확인 항목', '결과', '확인'],
        'rows': [['가연성 가스 농도(LEL)', '0 %', 'O'], ['산소 농도', '20.9 %', 'O'], ['소화기 2대 비치', '비치 완료', 'O'],
                 ['불티 비산 방지포', '설치 완료', 'O'], ['화재 감시자 지정', '이OO', 'O']],
        'foot': ['허가자 : 생산팀장    작업 책임자 : 정비팀 최OO', '작업 종료 후 30분간 화재 감시 유지']}),
    ('doc_3_pump_inspection.png', '펌프 월간 점검표', 'table', {
        'info': [('설비 번호', 'P-101A (원료 이송 펌프)'), ('점검월', '2026년 9월'), ('점검자', '설비팀 한OO')],
        'head': ['점검 항목', '측정값', '기준', '판정'],
        'rows': [['진동 속도', '2.1 mm/s', '4.5 이하', '양호'], ['베어링 온도', '58 °C', '80 이하', '양호'],
                 ['메커니컬 실 누설', '없음', '없음', '양호'], ['윤활유 레벨', '정상', '정상', '양호'],
                 ['토출 압력', '6.2 barg', '5.5 ~ 7.0', '양호']],
        'foot': ['특이 사항 : 없음. 다음 점검 2026년 10월']}),
    ('doc_4_msds_toluene.png', '물질안전보건자료 요약 — 톨루엔', 'form', {
        'info': [('물질명', '톨루엔 (Toluene, C7H8)'), ('인화점', '4 °C — 인화성 액체 구분 2'),
                 ('노출 기준', 'TWA 50 ppm'), ('주요 유해성', '흡입 시 중추신경 억제, 피부 자극')],
        'head': ['구분', '내용'],
        'rows': [['호흡 보호구', '유기용제용 방독 마스크'], ['손 보호구', '니트릴 또는 바이톤 장갑'],
                 ['눈 보호구', '보안경 (비산 시 고글)'], ['보관', '서늘하고 환기 잘 되는 곳, 화기 금지'],
                 ['누출 시', '점화원 제거 후 흡착재로 회수']],
        'foot': ['자세한 내용은 공급사 MSDS 원본 참조']}),
    ('doc_5_training_schedule.png', '10월 안전 교육 일정표', 'table', {
        'info': [('대상', '생산 · 정비 · 협력업체 전원'), ('장소', '본관 2층 교육장')],
        'head': ['날짜', '과정', '대상', '시간'],
        'rows': [['10/02', '밀폐 공간 작업', '정비팀', '2h'], ['10/08', '화학물질 취급 (MSDS)', '생산팀', '2h'],
                 ['10/15', '추락 방지 · 안전대', '협력업체', '3h'], ['10/22', '소방 훈련', '전원', '1h'],
                 ['10/29', '공정 안전 관리(PSM)', '관리자', '4h']],
        'foot': ['미이수자는 11월 보충 교육 필수']}),
    ('doc_6_reactor_log.png', '반응기 R-301 온도 기록', 'chart', {
        'info': [('설비', '반응기 R-301'), ('운전 기준', '170 ~ 180 °C'), ('경보 설정', '185 °C')],
        'series': [172, 174, 175, 176, 178, 181, 184, 187, 189, 186, 182, 178],
        'foot': ['14:30 경보 발생 — 냉각수 밸브 수동 개방 후 16:00 정상 복귀', '원인 조사 : 냉각수 유량 저하 (필터 막힘)']}),
]


def font(path, size):
    return ImageFont.truetype(path, size)


def draw_doc(fpath, title, kind, d, out):
    im = Image.new('RGB', (W, H), 'white')
    g = ImageDraw.Draw(im)
    F = lambda s: font(fpath, s)
    g.rectangle([30, 30, W - 30, H - 30], outline='#333333', width=2)
    g.text((60, 60), title, fill='#111111', font=F(40))
    g.line([60, 120, W - 60, 120], fill='#333333', width=2)
    y = 145
    for k, v in d['info']:
        g.text((60, y), k, fill='#555555', font=F(22))
        g.text((260, y), v, fill='#111111', font=F(22))
        y += 42
    y += 20
    if kind == 'chart':
        xs0, ys0, cw, chh = 110, y + 20, W - 190, 420
        lo, hi = 165, 195
        g.rectangle([xs0, ys0, xs0 + cw, ys0 + chh], outline='#888888', width=1)
        for t in range(lo, hi + 1, 5):
            yy = ys0 + chh - (t - lo) / (hi - lo) * chh
            g.line([xs0, yy, xs0 + cw, yy], fill='#E5E7EB', width=1)
            g.text((xs0 - 50, yy - 11), str(t), fill='#555555', font=F(18))
        ya = ys0 + chh - (185 - lo) / (hi - lo) * chh
        g.line([xs0, ya, xs0 + cw, ya], fill='#D32F2F', width=2)
        g.text((xs0 + cw - 110, ya - 30), '경보 185 °C', fill='#D32F2F', font=F(18))
        s = d['series']
        pts = [(xs0 + i / (len(s) - 1) * cw, ys0 + chh - (v - lo) / (hi - lo) * chh) for i, v in enumerate(s)]
        g.line(pts, fill='#1F4E99', width=3)
        for i, (px, py) in enumerate(pts):
            g.ellipse([px - 4, py - 4, px + 4, py + 4], fill='#1F4E99')
            if i % 2 == 0:
                g.text((px - 22, ys0 + chh + 8), f'{11 + i // 2}:{"00" if i % 2 == 0 else "30"}', fill='#555555', font=F(16))
        g.text((xs0, ys0 - 30), '온도 (°C) — 30분 간격', fill='#333333', font=F(18))
        y = ys0 + chh + 50
    else:
        cols = d['head']
        widths = [int((W - 120) / len(cols))] * len(cols)
        if len(cols) == 2:
            widths = [240, W - 120 - 240]
        x0 = 60
        rh = 52
        rows = [cols] + d['rows']
        for r, row in enumerate(rows):
            xx = x0
            for c, cell in enumerate(row):
                g.rectangle([xx, y, xx + widths[c], y + rh], outline='#666666', width=1,
                            fill='#E8EDF3' if r == 0 else 'white')
                g.text((xx + 12, y + 13), cell, fill='#111111', font=F(22))
                xx += widths[c]
            y += rh
        y += 30
    for line in d['foot']:
        g.text((60, y), line, fill='#111111', font=F(22))
        y += 40
    im.save(out, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--font', default=next((p for p in FONT_CANDIDATES if Path(p).exists()), None))
    a = ap.parse_args()
    if not a.font:
        raise SystemExit('한글 TTF 글꼴을 찾지 못했다 → --font 로 지정')
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'docs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'title', 'kind'])
        for name, title, kind, d in DOCS:
            draw_doc(a.font, title, kind, d, OUT / name)
            w.writerow([name, title, kind])
    print('written', len(DOCS), 'docs to', OUT)


if __name__ == '__main__':
    main()
