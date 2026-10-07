# CFG w 별 생성 그림 띠 — 실습 02 셀 9 와 같은 계산(같은 프롬프트 · 시드 42 · DDIM 20 · 빈 프롬프트 = ε_u)을 512px 로 다시 저장
#   docker exec ai4-gen python /work/5-2/tools/experiments/cfg_strip.py
#   결과 : 5-2/screens/slide/N_cfg_w.png
import os, glob, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor
torch.set_grad_enabled(False)
DEV, DT = 'cuda', torch.float16
pipe = StableDiffusionPipeline.from_single_file('/models/checkpoints/v1-5-pruned-emaonly.safetensors', torch_dtype=DT).to(DEV)
tok, te, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEV).eval(); cproc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
PROMPT = 'a red car on the beach'
enc = lambda t: te(tok([t], padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to(DEV)).last_hidden_state
c, u = enc(PROMPT), enc('')

def denoise(w):
    s = DDIMScheduler.from_config(pipe.scheduler.config); s.set_timesteps(20)
    z = torch.randn((1, 4, 64, 64), generator=torch.Generator(DEV).manual_seed(42), device=DEV, dtype=DT)
    for t in s.timesteps:
        if w == 1: e = unet(z, t, encoder_hidden_states=c).sample
        else:
            eu, ec = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([u, c])).sample.chunk(2); e = eu + w * (ec - eu)
        z = s.step(e, t, z).prev_sample
    x = vae.decode(z / vae.config.scaling_factor).sample
    return Image.fromarray(((x[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8'))

def score(im):
    o = clip(**cproc(text=[PROMPT], images=im, return_tensors='pt', padding=True).to(DEV))
    return F.cosine_similarity(o.image_embeds, o.text_embeds).item()

FB = (glob.glob('/usr/share/fonts/**/NanumGothicBold.ttf', recursive=True) + glob.glob('/usr/share/fonts/**/DejaVuSans-Bold.ttf', recursive=True))[0]
S, G, HB = 400, 24, 96
W = [1, 3, 7.5, 15]
out = Image.new('RGB', (len(W) * S + (len(W) - 1) * G, S + HB), 'white'); d = ImageDraw.Draw(out)
for k, w in enumerate(W):
    im = denoise(w); sc = score(im); x = k * (S + G)
    out.paste(im.resize((S, S), Image.LANCZOS), (x, HB))
    d.text((x + S // 2, 30), f'w = {w}', fill=(40, 40, 40), font=ImageFont.truetype(FB, 36), anchor='mm')
    d.text((x + S // 2, 74), f'CLIP 일치도 {sc:.2f}', fill=(63, 102, 148), font=ImageFont.truetype(FB, 30), anchor='mm')
    print(w, round(sc, 3))
os.makedirs('/work/5-2/screens/slide', exist_ok=True)
out.save('/work/5-2/screens/slide/N_cfg_w.png', optimize=True); print(out.size)
