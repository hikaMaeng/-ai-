import torch, numpy as np, torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor
torch.set_grad_enabled(False)
p = StableDiffusionPipeline.from_single_file('/models/checkpoints/v1-5-pruned-emaonly.safetensors', torch_dtype=torch.float16).to('cuda'); p.set_progress_bar_config(disable=True)
p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').cuda().eval(); cp = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
def cs(im, t):
    x = cp(text=[t], images=im, return_tensors='pt', padding=True).to('cuda'); o = clip(**x); return F.cosine_similarity(o.image_embeds, o.text_embeds).item()
def low(im):
    g = np.asarray(im.convert('L').resize((16, 16), Image.BOX), dtype=np.float32).ravel(); return g
att = p.unet.mid_block.attentions[0]; orig = att.forward
off = lambda hidden_states, *a, **k: (hidden_states,) if not k.get('return_dict', True) else type('o', (), {'sample': hidden_states})()
P = ['a red car on the beach', 'a cat sitting on a sofa, full body', 'a lighthouse on a cliff at sunset', 'a red cube on top of a blue sphere', 'two dogs and one cat on a sofa', 'a man riding a bicycle in a park']
R = {'on': [], 'off': []}; corr = []
for pr in P:
    for sd in [1, 2]:
        out = {}
        for mode in ['on', 'off']:
            att.forward = orig if mode == 'on' else off
            out[mode] = p(pr, num_inference_steps=20, guidance_scale=7.5, generator=torch.Generator('cuda').manual_seed(sd)).images[0]
            R[mode].append(cs(out[mode], pr))
        corr.append(np.corrcoef(low(out['on']), low(out['off']))[0, 1])
att.forward = orig
print('CLIP 일치도 평균 on %.3f off %.3f' % (np.mean(R['on']), np.mean(R['off'])))
print('구도(16x16) 상관 on↔off 평균 %.3f 최소 %.3f' % (np.mean(corr), np.min(corr)))
print('n', len(corr))
