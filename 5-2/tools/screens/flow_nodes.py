# 강의 「플로 매칭 — ComfyUI 화면」 장용 그림 : Qwen-Image 그래프 캡처에서 모델 로더 · 모델 샘플링(AuraFlow) · KSampler 부분만 잘라 번호 표시
#   좌표는 잘라 1500px 로 키운 그림 기준(눈으로 잰 값)
#   docker run --rm -v "<5-2/screens>:/s" -v "<5-2/tools/screens>:/w" ai4-lab/jupyter-clip python /w/flow_nodes.py
from PIL import Image, ImageDraw, ImageFont
FB = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
ORG = (242, 107, 58)
im = Image.open('/s/c6_qwen_graph.png').convert('RGB')
f = im.width / 2000                                   # 잘라내기 좌표 = 가로 2000 으로 본 화면 기준
crop = (0, 120, 1150, 625)
im = im.crop(tuple(int(v * f) for v in crop))
W = 1500; im = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)
d = ImageDraw.Draw(im); font = ImageFont.truetype(FB, 34)
def box(n, b, at):
    d.rounded_rectangle(b, radius=14, outline=ORG, width=6)
    d.ellipse((at[0] - 26, at[1] - 26, at[0] + 26, at[1] + 26), fill=ORG, outline='white', width=3)
    d.text(at, str(n), fill='white', font=font, anchor='mm')
box(1, (8, 34, 460, 248), (432, 60))                  # Unet Loader (GGUF) 노드
box(2, (545, 114, 957, 148), (530, 131))              # 모델 샘플링 : 시프트
box(3, (545, 152, 957, 186), (530, 169))              # 모델 샘플링 : 샘플링 flow
box(4, (1060, 278, 1472, 312), (1046, 295))           # KSampler : 스텝 수
box(5, (1060, 350, 1472, 386), (1046, 368))           # KSampler : 샘플러 이름 euler
box(6, (1060, 388, 1472, 422), (1046, 405))           # KSampler : 스케줄러 simple
im.save('/s/slide/FLOW_nodes.png', optimize=True); print(im.size)
