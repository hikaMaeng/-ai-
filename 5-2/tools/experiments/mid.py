import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
torch.set_grad_enabled(False)
p = StableDiffusionPipeline.from_single_file('/models/checkpoints/v1-5-pruned-emaonly.safetensors', torch_dtype=torch.float16).to('cuda'); p.set_progress_bar_config(disable=True)
p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
att = p.unet.mid_block.attentions[0]
orig = att.forward
def off(hidden_states, *a, **k):
    return (hidden_states,) if not k.get('return_dict', True) else type('o', (), {'sample': hidden_states})()
ims = []
prompts = ['a red car on the beach', 'a cat sitting on a sofa, full body', 'a lighthouse on a cliff at sunset']
for pr in prompts:
    for mode in ['on', 'off']:
        att.forward = orig if mode == 'on' else off
        g = torch.Generator('cuda').manual_seed(42)
        ims.append(p(pr, num_inference_steps=20, guidance_scale=7.5, generator=g).images[0].resize((256, 256)))
att.forward = orig
W = Image.new('RGB', (512, 256 * len(prompts)), 'white')
for i, im in enumerate(ims): W.paste(im, ((i % 2) * 256, (i // 2) * 256))
W.save('/work/5-2/outputs/exp/mid.png'); print('ok')
