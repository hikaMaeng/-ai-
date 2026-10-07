# sampler 장 1번 「DDPM : step 을 줄이면 품질 급락」 검증 — 같은 프롬프트 · 시드로 DDPM 20 · 50 · 1000 step 과 DDIM 20 step 비교
#   docker exec ai4-gen python /work/5-2/tools/experiments/ddpm_steps.py
#   결과 : 5-2/outputs/exp/ddpm_steps.png · 표(CLIP 일치도 · 남은 노이즈 양 · 시간)
import os, time, torch, numpy as np, torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor
torch.set_grad_enabled(False)
p = StableDiffusionPipeline.from_single_file('/models/checkpoints/v1-5-pruned-emaonly.safetensors', torch_dtype=torch.float16).to('cuda'); p.set_progress_bar_config(disable=True)
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').cuda().eval(); cp = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
P = ['a red car on the beach', 'a lighthouse on a cliff at sunset', 'a cat sitting on a sofa, full body']
def cs(im, t):
    o = clip(**cp(text=[t], images=im, return_tensors='pt', padding=True).to('cuda')); return F.cosine_similarity(o.image_embeds, o.text_embeds).item()
def leftover(im):   # 실습 02 셀 4 와 같은 계산 : 저주파(16×16)를 뺀 나머지의 표준편차 — 노이즈가 남으면 커짐
    g = np.asarray(im.convert('L'), dtype=np.float32)
    low = np.asarray(Image.fromarray(g).resize((16, 16), Image.BOX).resize((512, 512), Image.BICUBIC), dtype=np.float32)
    return float((g - low).std())
RUNS = [('DDPM', DDPMScheduler, 20), ('DDPM', DDPMScheduler, 50), ('DDPM', DDPMScheduler, 1000), ('DDIM', DDIMScheduler, 20)]
base = p.scheduler.config; ims = []
print('sampler step  CLIP일치도(3개 평균)  남은노이즈(3개 평균)  시간(1장)')
for name, S, n in RUNS:
    p.scheduler = S.from_config(base, steps_offset=0) if n == 1000 else S.from_config(base); c, l, ts = [], [], []   # 1000 step 은 offset 1 이면 t=1000 색인 오류
    for k, pr in enumerate(P):
        torch.cuda.synchronize(); t0 = time.time()
        im = p(pr, num_inference_steps=n, guidance_scale=7.5, generator=torch.Generator('cuda').manual_seed(42)).images[0]
        torch.cuda.synchronize(); ts.append(time.time() - t0)
        c.append(cs(im, pr)); l.append(leftover(im))
        if k == 0: ims.append(im.resize((256, 256)))
    print('%-5s %5d   %.3f              %.1f                %.1f초' % (name, n, np.mean(c), np.mean(l), np.mean(ts)))
os.makedirs('/work/5-2/outputs/exp', exist_ok=True)
W = Image.new('RGB', (256 * len(ims), 256), 'white')
for i, im in enumerate(ims): W.paste(im, (256 * i, 0))
W.save('/work/5-2/outputs/exp/ddpm_steps.png')
