# CFG 장용 ComfyUI 캡처 두 장(SD1.5 · FLUX) — 실제 실행 화면에서 잘라 번호 표시
#   docker run --rm -v "<5-2/screens>:/s" -v "<5-2/tools/screens>:/w" ai4-lab/jupyter-clip python /w/cfg_view.py
from PIL import Image, ImageDraw, ImageFont
FB = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
ORG = (242, 107, 58)

def panel(src, crop, items, title, width=1000):
    im = Image.open(f'/s/{src}.png').convert('RGB'); f = im.width / 2000
    im = im.crop(tuple(int(v * f) for v in crop)); k = width / im.width
    im = im.resize((width, int(im.height * k)), Image.LANCZOS)
    HB = 56; out = Image.new('RGB', (width, im.height + HB), (34, 34, 38)); out.paste(im, (0, HB))
    d = ImageDraw.Draw(out); d.text((16, HB // 2), title, fill='white', font=ImageFont.truetype(FB, 30), anchor='lm')
    s = f * k; r = 20; fn = ImageFont.truetype(FB, 26)          # 화면 좌표(가로 2000 기준) → 출력 픽셀
    for n, box in items:
        x0, y0, x1, y1 = (box[0] - crop[0]) * s, (box[1] - crop[1]) * s + HB, (box[2] - crop[0]) * s, (box[3] - crop[1]) * s + HB
        d.rounded_rectangle((x0, y0, x1, y1), radius=10, outline=ORG, width=5)
        cx, cy = x1 - 4, y0 + 4
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ORG, outline='white', width=3); d.text((cx, cy), str(n), fill='white', font=fn, anchor='mm')
    return out

a = panel('c1_sd15_graph', (395, 240, 1150, 585),
          [(1, (402, 247, 748, 437)), (2, (402, 455, 748, 580)), (3, (800, 308, 1000, 352)), (4, (812, 462, 1128, 487))],
          'SD1.5 — CFG 켬 (cfg 7.5)')
b = panel('c5_flux_graph', (330, 190, 1285, 535),
          [(5, (664, 200, 948, 275)), (6, (1004, 378, 1264, 400)), (7, (334, 372, 620, 530))],
          'FLUX.1-dev — 가이던스 증류 (cfg 1)')
W = Image.new('RGB', (a.width + b.width + 24, max(a.height, b.height)), 'white')
W.paste(a, (0, 0)); W.paste(b, (a.width + 24, 0))
W.save('/s/slide/N_cfg_comfy.png', optimize=True); print(W.size)
