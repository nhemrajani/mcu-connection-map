"""One image, to prove the stack works end to end before building anything."""
import time, torch
from diffusers import AutoPipelineForText2Image
t0=time.time()
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("mps")
print(f"loaded in {time.time()-t0:.0f}s")
t0=time.time()
img = pipe(prompt="a silver age comic book panel, ben-day dots, heavy black ink, "
                  "flat four-colour printing, a glowing cube on a laboratory table",
           num_inference_steps=4, guidance_scale=0.0).images[0]
img.save("art/_smoke.png")
print(f"generated in {time.time()-t0:.1f}s -> art/_smoke.png  {img.size}")
