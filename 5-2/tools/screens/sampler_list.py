# 강의 9장 「sampler」 옆 그림 : KSampler 의 「샘플러 이름」을 눌러 펼친 목록, 본문에서 설명하는 샘플러를 표시
#   docker run --rm -v "<5-2/screens>:/s" -v "<5-2/tools/screens>:/w" ai4-lab/jupyter-clip python /w/sampler_list.py
from PIL import Image, ImageDraw, ImageFont
FB = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
ORG = (242, 107, 58)
im = Image.open('/s/c3_sampler_list.png').convert('RGB')
f = im.width / 2000                                   # 좌표 = 캡처를 가로 2000 으로 본 화면 기준
crop = (742, 95, 1165, 1112)
im = im.crop(tuple(int(v * f) for v in crop))
H = 2000; k = H / im.height
im = im.resize((int(im.width * k), H), Image.LANCZOS)
d = ImageDraw.Draw(im)
s = k * f                                             # 화면 좌표 → 출력 좌표
X = lambda x: (x - crop[0]) * s
Y = lambda y: (y - crop[1]) * s
# 눌러서 펼친 칸 : 「샘플러 이름」 위젯
d.rounded_rectangle((X(746), Y(470), X(958), Y(506)), radius=int(10 * s), outline=ORG, width=int(4 * s))
# 본문에서 설명하는 샘플러 (목록에서의 화면 y)
ROWS = {'euler': 130, 'euler_ancestral': 174, 'heun': 220, 'dpmpp_2m': 513, 'dpmpp_2m_sde': 558,
        'ddpm': 693, 'lcm': 715, 'res_multistep': 828, 'ddim': 1076, 'uni_pc': 1098}
for name, y in ROWS.items():
    d.rounded_rectangle((X(960), Y(y - 11), X(1158), Y(y + 11)), radius=int(6 * s), outline=ORG, width=int(3 * s))
out = Image.new('RGB', (im.width, im.height + 110), (34, 34, 38))
out.paste(im, (0, 110))
ImageDraw.Draw(out).text((24, 55), 'ComfyUI · 샘플러 이름 목록', fill='white', font=ImageFont.truetype(FB, 52), anchor='lm')
out.save('/s/slide/S9_sampler_list.png', optimize=True)
print(out.size)
