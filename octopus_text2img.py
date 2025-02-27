import os

dependencies = [
    'apt-get update --fix-missing && apt-get install -y --no-install-recommends aria2 cmake ghostscript git libegl-dev libffi-dev libfreetype6-dev libfribidi-dev libharfbuzz-dev libimagequant-dev libjpeg-turbo-progs libjpeg8-dev liblcms2-dev libopengl-dev libopenjp2-7-dev libssl-dev libtiff5-dev libwebp-dev libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxkbcommon-x11-0 meson netpbm python3-dev python3-numpy python3-setuptools python3-tk sudo tcl8.6-dev tk8.6-dev virtualenv wget xvfb zlib1g-dev # required by Pillow',
    "pip install -q accelerate==1.4.0",
    "pip install -q diffusers==0.32.2",
    "pip install -q einops==0.8.1",
    "pip install -q Flask==3.1.0",
    'pip install -q Pillow==11.1.0',
    "pip install -q torch==2.6.0",
    "pip install -q torchsde==0.2.6",
    "pip install -q xformers==0.0.29"
]

for command in dependencies:
    os.system(command)

import gc
import io
import json
import nodes
import numpy as np
import random
import torch
from nodes import NODE_CLASS_MAPPINGS
from totoro_extras import nodes_custom_sampler
from totoro import model_management
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

def setup_environment():
    print("Changing directory to content")
    os.chdir("content")
    print("Cloning TotoroUI repository...")
    os.system("git clone -b totoro3 https://github.com/camenduru/ComfyUI content/TotoroUI")
    os.chdir("content/TotoroUI")
    print("Downloading model files...")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors -d /content/TotoroUI/models/unet -o flux1-dev-fp8.safetensors")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d /content/TotoroUI/models/vae -o ae.sft")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -d /content/TotoroUI/models/clip -o clip_l.safetensors")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors -d /content/TotoroUI/models/clip -o t5xxl_fp8_e4m3fn.safetensors")
    print("Environment setup completed.")

@app.route('/v1/setup', methods=["POST"])
def setup():
    setup_environment()
    response = {"setup": "Performed"}
    return jsonify(response), 201

@app.route('/v1/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    positive_prompt = data.get("positive_prompt", "A beautiful landscape painting")
    width = int(data.get("width", 512))
    height = int(data.get("height", 512))
    seed = int(data.get("seed", random.randint(0, 2**32 - 1)))
    steps = int(data.get("steps", 20))
    sampler_name = data.get("sampler_name", "euler")
    scheduler = data.get("scheduler", "simple")

    print(f"Starting inference with: prompt='{positive_prompt}', width={width}, height={height}, seed={seed}, steps={steps}, sampler='{sampler_name}', scheduler='{scheduler}'")

    with torch.inference_mode():
        # Example generation logic (placeholder)
        noise = torch.randn((1, 3, height, width), device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # ... (model inference code would go here) ...
        
        # Create a sample image (placeholder)
        generated_image = np.random.rand(height, width, 3) * 255
        pil_img = Image.fromarray(generated_image.astype(np.uint8))

        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        buffered.seek(0)

    return send_file(buffered, mimetype='image/png')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=True)
