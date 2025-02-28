import os

dependencies = [
    'apt-get update --fix-missing && apt-get install -y --no-install-recommends aria2 cmake ghostscript git libegl-dev libffi-dev libfreetype6-dev libfribidi-dev libharfbuzz-dev libimagequant-dev libjpeg-turbo-progs libjpeg8-dev liblcms2-dev libopengl-dev libopenjp2-7-dev libssl-dev libtiff5-dev libwebp-dev libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxkbcommon-x11-0 meson netpbm python3-dev python3-numpy python3-setuptools python3-tk sudo tcl8.6-dev tk8.6-dev virtualenv wget xvfb zlib1g-dev # required by Pillow',
    "pip install -q accelerate==1.4.0",
    "pip install -q diffusers==0.32.2",
    "pip install -q einops==0.8.1",
    "pip install -q Flask==3.1.0",
    'pip install -q Pillow==11.1.0',
    "pip install -q torch==2.5.1",
    "pip install -q torchvision==0.20.1",
    "pip install -q torchsde==0.2.6",
    "pip install -q transformers==4.48.2",
    "pip install -q xformers==0.0.29"
]

for command in dependencies:
    os.system(command)

import base64
import gc
import io
import json
import numpy as np
import random
import shutil
import sys
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image

config_str = '''{
    "device_map": {
        "cuda:0": "10GiB",
        "cpu": "30GiB"
    },
    "required_python_version": "cp312",
    "functions": [
        {
            "name": "generate_image",
            "display_name": "Generate Image from Text",
            "description": "Generate an image based on the provided text description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "positive_prompt": { "type": "string", "description": "The description to generate the image." },
                    "width": { "type": "integer", "description": "Width of the generated image." },
                    "height": { "type": "integer", "description": "Height of the generated image." },
                    "seed": { "type": "integer", "description": "Random seed for generation." },
                    "steps": { "type": "integer", "description": "Number of steps in the generation process." },
                    "sampler_name": { "type": "string", "description": "The sampler to use for image generation." },
                    "scheduler": { "type": "string", "description": "The scheduler to use in the generation process." }
                },
                "required": ["positive_prompt", "width", "height"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        }
    ]
}'''

config = json.loads(config_str)
app = Flask(__name__)

BasicGuider=None
BasicScheduler=None
clip=None
EmptyLatentImage=None
KSamplerSelect=None
RandomNoise=None
SamplerCustomAdvanced=None
unet=None
vae=None
VAEDecode=None

def setup_model():
    global BasicGuider
    global BasicScheduler
    global clip
    global EmptyLatentImage
    global KSamplerSelect
    global RandomNoise
    global SamplerCustomAdvanced
    global unet
    global vae
    global VAEDecode

    if os.path.isdir("/tmp/content"):
        shutil.rmtree("/tmp/content")
        os.mkdir("/tmp/content")
    else:
        os.mkdir("/tmp/content")
    print("Changing directory to /tmp/content")
    os.chdir("/tmp/content")
    print("Cloning TotoroUI repository...")
    os.system("git clone -b totoro3 https://github.com/camenduru/ComfyUI /tmp/content/TotoroUI")
    os.chdir("/tmp/content/TotoroUI")
    print("Downloading model files...")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors -d /tmp/content/TotoroUI/models/unet -o flux1-dev-fp8.safetensors")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d /tmp/content/TotoroUI/models/vae -o ae.sft")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -d /tmp/content/TotoroUI/models/clip -o clip_l.safetensors")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors -d /tmp/content/TotoroUI/models/clip -o t5xxl_fp8_e4m3fn.safetensors")
    print("Environment setup completed.")

    sys.path.append('/tmp/content/TotoroUI')
    import nodes
    from nodes import NODE_CLASS_MAPPINGS
    from totoro_extras import nodes_custom_sampler
    from totoro import model_management

    with torch.inference_mode():
        print("Loading models...")
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
        print("Models loaded successfully.")

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@app.route('/v1/setup', methods=["POST"])
def setup():
    setup_model()
    response = {"setup": "Performed"}
    return jsonify(response), 201

@app.route('/v1/generate_image', methods=['POST'])
def generate_image():
    sys.path.append('/tmp/content/TotoroUI')
    from totoro import model_management

    data = request.get_json()
    positive_prompt = data.get("positive_prompt", "cyperpunk octopus spelling out the words FLUX DEV, cinematic, dynamic shot")
    width = int(data.get("width", 512))
    height = int(data.get("height", 512))
    seed = int(data.get("seed", 0))
    steps = int(data.get("steps", 20))
    sampler_name = data.get("sampler_name", "euler")
    scheduler = data.get("scheduler", "simple")

    print(f"Starting inference with: prompt='{positive_prompt}', width={width}, height={height}, seed={seed}, steps={steps}, sampler='{sampler_name}', scheduler='{scheduler}'")

    with torch.inference_mode():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            print("Moving UNET model to GPU...")
            unet.model.to('cuda:0')
        except Exception as e:
            print(f"Error moving UNET model to GPU: {e}")

        if seed == 0:
            seed = random.randint(0, 18446744073709551615)
        print(f"Using seed: {seed}")

        cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
        cond = [[cond, {"pooled_output": pooled}]]
        noise = RandomNoise.get_noise(seed)[0]
        guider = BasicGuider.get_guider(unet, cond)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]
        latent_image = EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
        sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        model_management.soft_empty_cache()
        decoded = VAEDecode.decode(vae, sample)[0].detach()

        if hasattr(unet, 'model') and isinstance(unet.model, torch.nn.Module):
            print("Moving UNET model back to CPU...")
            unet.model.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        else:
            print("The UNET model does not expose a direct .to() method.")

        # Create the PIL image from the decoded tensor
        pil_img = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        buffered.seek(0)

    encoded_content = base64.b64encode(buffered.read()).decode('utf-8')

    response = {
        "file_attachments": [
            {
                "content": encoded_content,
                "file_name": "image.png",
                "media_type": "image/png"
            }
        ]
    }
    return jsonify(response), 201

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=True)
