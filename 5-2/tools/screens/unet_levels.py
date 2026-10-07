# U-Net 구조 도해용 그림 4장 — 실제 생성 그림(SD1.5, 시드 42) 위에 해상도별 칸 격자 + 3×3 합성곱 창이 덮는 범위
#   64×64 · 32×32 는 창이 작아 오른쪽 위에 확대창(창 주변 3배 영역)을 함께 그림
#   docker run --rm -v "<5-2>:/r" ai4-lab/jupyter-clip python /r/tools/screens/unet_levels.py
from PIL import Image, ImageDraw
SRC = '/r/outputs/comfy/5-2/05/sd15_00001_.png'      # 512×512, "a red car on the beach"
S = 660                                               # 출력 한 변(슬라이드에서 165px 로 줄여 씀)
ORG = (242, 107, 58)
base = Image.open(SRC).convert('RGB').resize((S, S), Image.LANCZOS)
CX, CY = 0.50, 0.62                                   # 창 가운데(자동차 위)

def draw_grid(img, n, c, ox=0, oy=0, x_range=None, y_range=None, lw=1, alpha=90):
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    w, h = img.size
    for i in range(n + 1):
        x = ox + i * c; y = oy + i * c
        if -1 <= x <= w: d.line([(x, 0), (x, h)], fill=(255, 255, 255, alpha), width=lw)
        if -1 <= y <= h: d.line([(0, y), (w, y)], fill=(255, 255, 255, alpha), width=lw)
    return Image.alpha_composite(img, ov)

for n in (64, 32, 16, 8):
    c = S / n
    im = draw_grid(base.convert('RGBA'), n, c, lw=1 if n >= 32 else 2, alpha=80 if n >= 32 else 130)
    gx, gy = int(CX * n), int(CY * n)
    x0, y0 = (gx - 1) * c, (gy - 1) * c
    ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    d.rectangle([x0, y0, x0 + 3 * c, y0 + 3 * c], fill=ORG + (90,), outline=ORG + (255,), width=6)
    d.rectangle([gx * c, gy * c, (gx + 1) * c, (gy + 1) * c], outline=(255, 255, 255, 255), width=3)
    im = Image.alpha_composite(im, ov)
    if n in (64, 32):                                  # 확대창 : 창 가운데 기준 3배 영역 → 오른쪽 위 300px
        side = 9 * c                                   # 3×3 창의 3배 = 9칸
        L, T = gx * c + c / 2 - side / 2, gy * c + c / 2 - side / 2
        Z = 300
        crop = base.crop((int(L), int(T), int(L + side), int(T + side))).resize((Z, Z), Image.NEAREST if n == 64 else Image.BICUBIC).convert('RGBA')
        zc = Z / 9
        crop = draw_grid(crop, 9, zc, lw=2, alpha=170)
        zd = ImageDraw.Draw(crop)
        zd.rectangle([3 * zc, 3 * zc, 6 * zc, 6 * zc], outline=ORG + (255,), width=6)
        zd.rectangle([4 * zc, 4 * zc, 5 * zc, 5 * zc], outline=(255, 255, 255, 255), width=4)
        PX, PY = S - Z - 10, 10
        frame = ImageDraw.Draw(im)
        frame.line([(x0 + 3 * c, y0), (PX, PY + Z)], fill=(255, 255, 255, 230), width=3)
        frame.rectangle([PX - 4, PY - 4, PX + Z + 4, PY + Z + 4], fill=(255, 255, 255, 255))
        im.paste(crop, (PX, PY))
    im.convert('RGB').save(f'/r/screens/slide/U_level_{n}.png', optimize=True)
    print(n, 'cell', 512 // n, 'px window', 3 * 512 // n, 'px')
