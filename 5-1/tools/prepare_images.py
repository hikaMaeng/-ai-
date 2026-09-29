"""5-1 실습 이미지 준비 스크립트 (강사용 — 수강생은 실행하지 않는다)

저장소의 5-1/images (사진 150장 + images.csv) 를 만든다. 결과물은 저장소에 커밋돼 있으므로
수업 중에는 이 스크립트를 돌릴 필요가 없다. 원본을 바꾸거나 다시 고를 때만 쓴다.

    python prepare_images.py --hardhat-zip <test.zip 경로> --hardhat-zip <valid.zip 경로>

만드는 것
  pet_*.jpg    Oxford-IIIT Pet(test) 8품종 × 5장        CC BY-SA 4.0
  hh_off_*.jpg 현장 사진, 전원 안전모 미착용              CC BY 4.0 (Roboflow Hard Hats)
  hh_on_*.jpg  현장 사진, 전원 안전모 착용                 CC BY 4.0
  shape_*.png  직접 그린 도형(관계 · 결합 · 개수)         이 저장소에서 생성
"""
import argparse, csv, io, json, math, random, re, zipfile
from pathlib import Path

import requests
from PIL import Image, ImageDraw

IMG = Path(__file__).resolve().parents[1] / 'images'   # 5-1/images
MAX_SIDE = 320

PET_BREEDS = ['shiba_inu', 'samoyed', 'pug', 'beagle', 'chihuahua', 'siamese', 'persian', 'bengal']
PET_PER = 5
HH_PER = 20
SEED = 5


def save_jpg(im, name):
    im = im.convert('RGB')
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    im.save(IMG / name, quality=85, optimize=True)


def pets(rows):
    # datasets-server 의 /rows 로 test 분할을 100행씩 훑어 품종마다 앞의 PET_PER 장을 모은다
    base = 'https://datasets-server.huggingface.co/rows?dataset=timm/oxford-iiit-pet&config=default&split=test'
    picked = {b: [] for b in PET_BREEDS}
    names = None
    for off in range(0, 3669, 100):
        j = requests.get(f'{base}&offset={off}&length=100', timeout=120).json()
        names = names or next(f['type']['names'] for f in j['features'] if f['name'] == 'label')
        for r in j['rows']:
            b = names[r['row']['label']]
            if b in picked and len(picked[b]) < PET_PER:
                picked[b].append(r['row'])
        if all(len(v) == PET_PER for v in picked.values()):
            break
    for breed in PET_BREEDS:
        got = picked[breed]
        assert len(got) == PET_PER, (breed, len(got))
        for i, row in enumerate(got, 1):
            im = Image.open(io.BytesIO(requests.get(row['image']['src'], timeout=60).content))
            name = f'pet_{breed}_{i}.jpg'
            save_jpg(im, name)
            rows.append(dict(file=name, group='pet', label=breed, detail=row['label_cat_dog'] and 'dog' or 'cat',
                             source='timm/oxford-iiit-pet test', source_id=row['image_id'], license='CC BY-SA 4.0'))
        print('pet', breed, 'ok')


# Roboflow Hard Hats 는 여러 출처를 합친 데이터다. 미착용(no-hardhat) 사진 대부분은 실내 웹캠 셀카라
# 착용 사진(공사 현장)과 배경부터 달라, CLIP 이 안전모가 아니라 배경으로 가를 수 있다.
# → 두 쪽 모두 현장 사진 출처(파일명이 6자리 숫자 · ppe_ · helmet 으로 시작)에서만 고른다.
SITE = re.compile(r'^(\d{6}_jpg|ppe_\d|helmet)')
# 널리 알려진 인물(정치인 등)이 식별되는 사진, 같은 장면의 중복은 교육 자료에서 뺀다
#   후보 인화지를 눈으로 확인해 원본 파일명(".rf." 앞부분)으로 기록
EXCLUDE = {'004689_jpg', '005516_jpg', '003656_jpg', '000926_jpg', '000813_jpg',   # 알려진 인물
           '000100_jpg',                                                           # 착용 쪽, 알려진 인물
           '005541_jpg'}                                                           # 같은 사진이 두 분할에 중복


def hardhats(rows, zpaths):
    off, on = [], []
    for zp in zpaths:
        z = zipfile.ZipFile(zp)
        j = json.loads(z.read('_annotations.coco.json'))
        by_img = {}
        for a in j['annotations']:
            by_img.setdefault(a['image_id'], []).append(a)
        imgs = {im['id']: im for im in j['images']}
        for iid, anns in sorted(by_img.items()):
            im = imgs[iid]
            if not SITE.match(im['file_name']) or len(anns) > 5 or im['file_name'].split('.rf.')[0] in EXCLUDE:
                continue
            area = im['width'] * im['height']
            cats = [a['category_id'] for a in anns]
            big = lambda c: max([a['bbox'][2] * a['bbox'][3] for a in anns if a['category_id'] == c] or [0]) / area
            if set(cats) == {1} and big(1) >= 0.02:             # 전원 미착용, 머리가 화면의 2% 이상
                off.append((zp, im['file_name'], cats.count(1), cats.count(0)))
            elif set(cats) == {0} and big(0) >= 0.02:           # 전원 착용
                on.append((zp, im['file_name'], 0, cats.count(0)))
    rnd = random.Random(SEED)
    rnd.shuffle(off); rnd.shuffle(on)
    for tag, items, label in (('off', off[:HH_PER], 'no-hardhat'), ('on', on[:HH_PER], 'hardhat')):
        for i, (zp, fn, n_off, n_on) in enumerate(items, 1):
            name = f'hh_{tag}_{i:02d}.jpg'
            save_jpg(Image.open(io.BytesIO(zipfile.ZipFile(zp).read(fn))), name)
            rows.append(dict(file=name, group='hardhat', label=label, detail=f'미착용 {n_off} · 착용 {n_on}',
                             source=f'keremberke/hard-hat-detection {Path(zp).stem} (Roboflow Hard Hats)', source_id=fn,
                             license='CC BY 4.0'))
    print('hardhat 후보 off', len(off), '/ on', len(on), '→ 각', HH_PER)


COLORS = {'red': (220, 40, 40), 'blue': (40, 80, 220)}


def shapes(rows):
    rnd = random.Random(SEED)
    W = 224
    # ① 관계: 빨강 네모가 파랑 네모 위 / 아래
    for top in ('red', 'blue'):
        bottom = 'blue' if top == 'red' else 'red'
        for i in range(1, 11):
            im = Image.new('RGB', (W, W), 'white'); d = ImageDraw.Draw(im)
            s = rnd.randint(46, 64); x = rnd.randint(40, W - 40 - s)
            y1 = rnd.randint(20, 40); y2 = y1 + s + rnd.randint(10, 30)
            d.rectangle([x, y1, x + s, y1 + s], fill=COLORS[top])
            x2 = min(max(x + rnd.randint(-12, 12), 10), W - 10 - s)
            d.rectangle([x2, y2, x2 + s, y2 + s], fill=COLORS[bottom])
            name = f'shape_above_{top}_{i:02d}.png'; im.save(IMG / name)
            rows.append(dict(file=name, group='shape_relation', label=f'{top} above {bottom}', detail='',
                             source='generated', source_id='', license='이 저장소'))
    # ② 결합: 빨강 원 + 파랑 네모 / 파랑 원 + 빨강 네모 (좌우는 무작위)
    for circ in ('red', 'blue'):
        sq = 'blue' if circ == 'red' else 'red'
        for i in range(1, 11):
            im = Image.new('RGB', (W, W), 'white'); d = ImageDraw.Draw(im)
            s = rnd.randint(50, 66); y = rnd.randint(60, W - 60 - s)
            xs = [rnd.randint(16, 40), rnd.randint(124, 150)]
            if rnd.random() < 0.5: xs.reverse()
            d.ellipse([xs[0], y, xs[0] + s, y + s], fill=COLORS[circ])
            y2 = min(max(y + rnd.randint(-15, 15), 10), W - 10 - s)
            d.rectangle([xs[1], y2, xs[1] + s, y2 + s], fill=COLORS[sq])
            name = f'shape_bind_{circ}circle_{i:02d}.png'; im.save(IMG / name)
            rows.append(dict(file=name, group='shape_binding', label=f'{circ} circle and {sq} square', detail='',
                             source='generated', source_id='', license='이 저장소'))
    # ③ 개수: 검은 원 2~6개 (서로 겹치지 않게)
    for n in range(2, 7):
        for i in range(1, 7):
            im = Image.new('RGB', (W, W), 'white'); d = ImageDraw.Draw(im)
            placed = []
            while len(placed) < n:
                r = rnd.randint(14, 22); x = rnd.randint(r + 6, W - r - 6); y = rnd.randint(r + 6, W - r - 6)
                if all(math.hypot(x - a, y - b) > r + c + 8 for a, b, c in placed):
                    placed.append((x, y, r))
            for x, y, r in placed:
                d.ellipse([x - r, y - r, x + r, y + r], fill=(30, 30, 30))
            name = f'shape_count_{n}_{i:02d}.png'; im.save(IMG / name)
            rows.append(dict(file=name, group='shape_count', label=str(n), detail='',
                             source='generated', source_id='', license='이 저장소'))
    print('shapes ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hardhat-zip', required=True, action='append',
                    help='keremberke/hard-hat-detection 의 data/test.zip · data/valid.zip (여러 번 줄 수 있음)')
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    for f in IMG.glob('*'):
        f.unlink()   # images.csv 포함, 다시 만든다
    rows = []
    pets(rows)
    hardhats(rows, a.hardhat_zip)
    shapes(rows)
    with open(IMG / 'images.csv', 'w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=['file', 'group', 'label', 'detail', 'source', 'source_id', 'license'])
        w.writeheader(); w.writerows(rows)
    print('총', len(rows), '장 →', IMG)


if __name__ == '__main__':
    main()
