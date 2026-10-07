import glob, numpy as np
from PIL import Image, ImageDraw, ImageFont
fs = sorted(glob.glob('/s/zseq_*.png'))
fr = []
for f in fs:
    a = np.asarray(Image.open(f).convert('RGB')).astype(int)
    g = (a[60, 15:-15, 1] > 100) & (a[60, 15:-15, 0] < 90) & (a[60, 15:-15, 2] < 90)
    fr.append(g.sum() / (a.shape[1] - 30))
steps = [round(x * 30) for x in fr]
print(list(zip(range(1, len(fs) + 1), steps)))
# 미리보기가 있는 프레임 중 단계가 바뀌는 것만, 원하는 단계 근처를 고름
cand = [(i, s) for i, s in enumerate(steps) if s > 0]
pick, seen = [], set()
for target in [1, 3, 5, 8, 12, 18, 30]:
    i, s = min(cand, key=lambda t: abs(t[1] - target))
    if i not in seen: seen.add(i); pick.append((i, s))
font = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf', 28) if __import__('os').path.exists('/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf') else ImageFont.load_default()
tiles = []
for i, s in pick:
    im = Image.open(fs[i]).crop((6, 703, 474, 1171)).resize((360, 360), Image.LANCZOS)
    tiles.append((s, im))
W = Image.new('RGB', (len(tiles) * 372 - 12, 410), 'white'); d = ImageDraw.Draw(W)
for j, (s, im) in enumerate(tiles):
    W.paste(im, (j * 372, 0)); d.text((j * 372 + 180, 385), f'{s} / 30 단계', fill='black', font=font, anchor='mm')
W.save('/s/c9_zimage_preview_strip.png'); print('pick', pick)
