# CFG w 별 CLIP 일치도 — 「0.20 · 0.34 가 무슨 뜻인가」의 잣대
#   그림 ↔ 자기 프롬프트 코사인과 그림 ↔ 무관한 프롬프트(나머지 5개) 코사인을 함께 잰다
#   docker exec ai4-gen python /work/5-2/tools/experiments/cfg_scale.py
import torch, numpy as np, torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor
torch.set_grad_enabled(False)
p = StableDiffusionPipeline.from_single_file('/models/checkpoints/v1-5-pruned-emaonly.safetensors', torch_dtype=torch.float16).to('cuda'); p.set_progress_bar_config(disable=True)
p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').cuda().eval(); cp = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
P = ['a red car on the beach', 'a cat sitting on a sofa, full body', 'a lighthouse on a cliff at sunset', 'a red cube on top of a blue sphere', 'two dogs and one cat on a sofa', 'a man riding a bicycle in a park']
W, SEEDS = [1, 3, 7.5, 15], [1, 2]
print('w     자기 프롬프트  무관한 프롬프트 5개 평균  (그림 %d장 평균)' % (len(P) * len(SEEDS)))
for w in W:
    own, other = [], []
    for i, pr in enumerate(P):
        for sd in SEEDS:   # guidance_scale 1 이면 diffusers 가 CFG 를 끔(ε_c 만, 예측 한 번)
            im = p(pr, num_inference_steps=20, guidance_scale=w, generator=torch.Generator('cuda').manual_seed(sd)).images[0]
            o = clip(**cp(text=P, images=im, return_tensors='pt', padding=True).to('cuda'))   # image_embeds · text_embeds 는 정규화된 벡터
            s = (o.image_embeds @ o.text_embeds.T)[0].float().cpu().numpy()
            own.append(s[i]); other.append(np.delete(s, i).mean())
    print('%-5s %.3f          %.3f' % (w, np.mean(own), np.mean(other)))
