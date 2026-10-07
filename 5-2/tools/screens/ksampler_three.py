# 강의 7장 「seed · sampler · scheduler」 옆 그림 : KSampler 위젯 줄만 잘라 세 칸을 표시
#   docker run --rm -v "<5-2/screens>:/s" -v "<5-2/tools/screens>:/w" ai4-lab/jupyter-clip python /w/ksampler_three.py
from PIL import Image, ImageDraw, ImageFont
FB = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
ORG = (242, 107, 58)
im = Image.open('/s/c2_ksampler.png').convert('RGB')
f = im.width / 2000                                   # 좌표 = 가로 2000 으로 본 화면 기준
crop = (740, 305, 1200, 590)
im = im.crop(tuple(int(v * f) for v in crop))
W = 1230; k = W / im.width
im = im.resize((W, int(im.height * k)), Image.LANCZOS)
HB = 70                                               # 위 띠
out = Image.new('RGB', (W, im.height + HB), (34, 34, 38))
out.paste(im, (0, HB))
d = ImageDraw.Draw(out)
d.text((22, HB // 2), 'ComfyUI · KSampler 노드', fill='white', font=ImageFont.truetype(FB, 40), anchor='lm')
for yy in (138, 560, 665):                            # 출력 그림에서 시드 · 샘플러 이름 · 스케줄러 줄의 가운데(눈으로 잰 값)
    d.rounded_rectangle((8, yy - 44, W - 8, yy + 44), radius=40, outline=ORG, width=7)
out = out.crop((0, 0, W, 722))                        # 노이즈 제거량 줄은 뺌
out.save('/s/slide/S7_ksampler_three.png', optimize=True)
print(out.size)
