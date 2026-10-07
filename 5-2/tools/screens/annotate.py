# ComfyUI 캡처 → 잘라서 번호 표시(①②…)를 얹은 슬라이드용 그림
#   좌표는 캡처를 가로 2000 으로 줄여 본 화면 기준(실제 픽셀 = 좌표 × 원본폭/2000)
from PIL import Image, ImageDraw, ImageFont
S, O = '/s', '/s/slide'
import os; os.makedirs(O, exist_ok=True)
FB = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
ORG = (242, 107, 58)

def mark(src, crop, items, out, width=2000):
    im = Image.open(f'{S}/{src}.png').convert('RGB')
    f = im.width / 2000
    im = im.crop(tuple(int(v * f) for v in crop))
    k = width / im.width
    im = im.resize((width, int(im.height * k)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    s = f * k                                  # 화면 좌표 → 출력 좌표 배율
    r = int(22 * s); font = ImageFont.truetype(FB, int(28 * s))
    for n, box, at in items:
        if box:
            x0, y0, x1, y1 = [(v - crop[i % 2]) * s for i, v in enumerate(box)]
            d.rounded_rectangle((x0, y0, x1, y1), radius=int(10 * s), outline=ORG, width=max(3, int(5 * s)))
        cx, cy = [(v - crop[i]) * s for i, v in enumerate(at)]
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ORG, outline='white', width=max(2, int(3 * s)))
        d.text((cx, cy), str(n), fill='white', font=font, anchor='mm')
    im.save(f'{O}/{out}.png', optimize=True)
    print(out, im.size)

TR = lambda b, dx=-26, dy=26: (b[2] + dx, b[1] + dy)      # 상자 오른쪽 위 안쪽

# A · SD1.5 그래프 실행 결과
b = [(5, 245, 352, 468), (400, 245, 748, 645), (5, 483, 352, 706), (796, 245, 1143, 830), (1191, 245, 1539, 438), (1587, 245, 1935, 592)]
mark('c1_sd15_graph', (0, 225, 1950, 845), [(i + 1, x, TR(x)) for i, x in enumerate(b)], 'A_sd15')

# B · KSampler 확대
rows = [194, 244, 294, 329, 408, 448, 488, 527, 567]
items = [(i + 1, None, (1252, y)) for i, y in enumerate(rows)] + [(10, (800, 590, 1139, 930), (1252, 760))]
items[1] = (2, (735, 214, 900, 274), (1252, 244))
mark('c2_ksampler', (680, 115, 1290, 965), items, 'B_ksampler', width=1200)

# E · FLUX
b = [(5, 198, 293, 335), (5, 345, 293, 532), (661, 198, 950, 360), (1003, 379, 1265, 399), (5, 688, 293, 875), (5, 543, 293, 678)]
mark('c5_flux_graph', (0, 175, 1945, 885), [(i + 1, x, TR(x) if i != 3 else (x[2] + 30, (x[1] + x[3]) / 2)) for i, x in enumerate(b)], 'E_flux')

# F · Qwen-Image
b = [(5, 145, 352, 308), (5, 322, 352, 515), (400, 145, 748, 338), (400, 560, 748, 752), (5, 705, 352, 930), (812, 364, 1127, 384)]
mark('c6_qwen_graph', (0, 115, 1945, 940), [(i + 1, x, TR(x) if i != 5 else (x[2] + 30, (x[1] + x[3]) / 2)) for i, x in enumerate(b)], 'F_qwen')

# G · Z-Image (출력 노드까지)
b = [(103, 57, 415, 230), (103, 242, 415, 415), (458, 57, 770, 230), (813, 520, 1127, 835), (1524, 57, 1836, 368)]
mark('c7_zimage_done', (95, 45, 1845, 1025), [(i + 1, x, TR(x)) for i, x in enumerate(b)], 'G_zimage')

# H · 인코더 바꿔치기 오류
b = [(63, 372, 400, 557), (825, 160, 1175, 628), (1640, 500, 1975, 655)]
mark('c8b_encoder_swap_detail', (60, 150, 2000, 905), [(i + 1, x, TR(x)) for i, x in enumerate(b)], 'H_swap')

# C · 샘플러 · 스케줄러 목록 나란히
def crop(src, box):
    im = Image.open(f'{S}/{src}.png').convert('RGB'); f = im.width / 2000
    return im.crop(tuple(int(v * f) for v in box))
L = crop('c3_sampler_list', (955, 115, 1160, 1110)); R = crop('c4_scheduler_list', (955, 545, 1160, 752))
k = 1.0
H_ = L.height + 120
C = Image.new('RGB', (L.width * 2 + 120, H_), (245, 246, 248)); d = ImageDraw.Draw(C)
ft = ImageFont.truetype(FB, 46)
C.paste(L, (30, 100)); C.paste(R, (L.width + 90, 100))
d.text((30, 30), '샘플러 이름', fill=(31, 41, 55), font=ft); d.text((L.width + 90, 30), '스케줄러', fill=(31, 41, 55), font=ft)
f = Image.open(f'{S}/c3_sampler_list.png').width / 2000
def hl(x0, y, x1, n, side='L'):
    yy = (y - 115) * f + 100; xs = 30 if side == 'L' else L.width + 90
    off = 0 if side == 'L' else 0
    d.rounded_rectangle((xs, yy - 22, xs + (x1 - x0) * f, yy + 22), radius=8, outline=ORG, width=5)
for y in (130, 174, 513, 693, 1076): hl(958, y, 1158, 0)
for y in (557, 603, 648):
    yy = (y - 545) * f + 100; xs = L.width + 90
    d.rounded_rectangle((xs, yy - 22, xs + 195 * f, yy + 22), radius=8, outline=ORG, width=5)
C.save(f'{O}/C_lists.png', optimize=True); print('C_lists', C.size)

# D · 단계별 미리보기 띠(이미 만든 그림)
Image.open(f'{S}/c9_zimage_preview_strip.png').save(f'{O}/D_strip.png', optimize=True); print('D_strip')
