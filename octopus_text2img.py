
import os

dependencies = [
    "pip install -q Flask==3.1.0",
    "pip install -q torchsde",
    "pip install -q einops",
    "pip install -q diffusers",
    "pip install -q accelerate",
    "pip install -q xformers==0.0.28.post2"
]

for command in dependencies:
    os.system(command)

import random
import gc
import torch
import numpy as np
from PIL import Image
import io
from flask import Flask, request, jsonify

app = Flask(__name__)

def setup_environment():
    os.chdir("/content")
    os.system("git clone -b totoro3 https://github.com/camenduru/ComfyUI /content/TotoroUI")
    os.chdir("/content/TotoroUI")
    os.system("apt -y install -qq aria2")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors -d /content/TotoroUI/models/unet -o flux1-dev-fp8.safetensors")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d /content/TotoroUI/models/vae -o ae.sft")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -d /content/TotoroUI/models/clip -o clip_l.safetensors")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors -d /content/TotoroUI/models/clip -o t5xxl_fp8_e4m3fn.safetensors")
    print("Environment setup completed.")

setup_environment()

import nodes
from nodes import NODE_CLASS_MAPPINGS
from totoro_extras import nodes_custom_sampler
from totoro import model_management

with torch.inference_mode():
    DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
    UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
    RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
    BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
    KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
    BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
    SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
    VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
    VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

    clip = DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
    unet = UNETLoader.load_unet("flux1-dev-fp8.safetensors", "fp8_e4m3fn")[0]
    unet.model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    vae = VAELoader.load_vae("ae.sft")[0]

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    n2 = m * (q + 1) if (n * m) > 0 else m * (q - 1)
    return n1 if abs(n - n1) < abs(n - n2) else n2

@app.route('/v1/generate_image', methods=['POST'])
def generate_image():
    data = request.get_json()
    positive_prompt = data.get("positive_prompt", "cyberpunk octopus spelling out the words FLUX DEV, cinematic, dynamic shot")
    width = int(data.get("width", 512))
    height = int(data.get("height", 512))
    seed = int(data.get("seed", 0))
    steps = int(data.get("steps", 20))
    sampler_name = data.get("sampler_name", "euler")
    scheduler = data.get("scheduler", "simple")

    with torch.inference_mode():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if seed == 0:
            seed = random.randint(0, 18446744073709551615)

        cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
        cond = [[cond, {"pooled_output": pooled}]]
        noise = RandomNoise.get_noise(seed)[0]
        guider = BasicGuider.get_guider(unet, cond)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]
        latent_image = EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
        sample, _ = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        model_management.soft_empty_cache()
        decoded = VAEDecode.decode(vae, sample)[0].detach()

        pil_img = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        buffered.seek(0)

    return send_file(buffered, mimetype='image/png')

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {"setup": "Performed"}
    return jsonify(response), 201
