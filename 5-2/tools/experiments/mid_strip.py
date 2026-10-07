from PIL import Image, ImageDraw, ImageFont
src = Image.open('/work/5-2/outputs/exp/mid.png')
FB = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
ft = ImageFont.truetype(FB, 26)
tiles = [src.crop(((i % 2) * 256, (i // 2) * 256, (i % 2) * 256 + 256, (i // 2) * 256 + 256)) for i in range(6)]
names = ['a red car on the beach', 'a cat sitting on a sofa', 'a lighthouse on a cliff']
W = Image.new('RGB', (6 * 256 + 5 * 8 + 2 * 24, 256 + 80), 'white'); d = ImageDraw.Draw(W)
x = 0
for k in range(3):
    for j in range(2):
        W.paste(tiles[k * 2 + j], (x, 40))
        d.text((x + 128, 318), 'mid 어텐션 켬' if j == 0 else 'mid 어텐션 끔', fill=(31, 41, 55) if j == 0 else (106, 76, 156), font=ft, anchor='mm')
        x += 256 + (8 if j == 0 else 0)
    d.text((x - 260, 20), names[k], fill=(95, 107, 120), font=ImageFont.truetype(FB, 22), anchor='mm')
    x += 24 + 8
W.save('/work/5-2/screens/slide/M_mid_onoff.png'); print(W.size)
