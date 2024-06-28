
import os

dependencies = [
    "pip install -q Flask==3.0.3",
    "pip install -q torch==2.3.1",
    "pip install -q opencv-python==4.10.0.84",
    "pip install -q torchvision==0.18.1",
]

for command in dependencies:
    os.system(command)

import json
import torch
import cv2
from flask import Flask, jsonify, request
from depth_anything_v2.dpt import DepthAnythingV2
import base64
import numpy as np
from io import BytesIO

config_str = '''{
    "device_map": {
        "cuda:0": "16GiB",
        "cuda:1": "16GiB",
        "cpu": "32GiB"
    },
    "required_python_version": "cp312",
    "models": {
        "model": "DepthAnythingV2"
    },
    "functions": [
        {
            "name": "depth_estimation",
            "display_name": "Depth Estimation with DepthAnything V2",
            "description": "Perform depth estimation on an image provided as URL with a prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": { "type": "string", "description": "URL of the image to process" },
                    "prompt": { "type": "string", "description": "Prompt to explain what is expected" }
                },
                "required": ["image_url", "prompt"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        }
    ]
}'''

config = json.loads(config_str)
app = Flask(__name__)

def command_result_as_int(command):
    return int(subprocess.check_output(command, shell=True).decode('utf-8').strip())

def select_device_with_larger_free_memory(available_devices):
    device = None
    memory = 0

    for available_device in available_devices:
        id = available_device.split(":")
        id = id[-1]
        free_memory = command_result_as_int(f"nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader --id={id}")
        if free_memory > memory:
            memory = free_memory
            device = available_device

    return device if device else "cpu"

def select_device():
    if not torch.cuda.is_available():
        return "cpu"

    device_map = config.get('device_map', {})
    available_devices = list(device_map.keys())
    return select_device_with_larger_free_memory(available_devices)

device = select_device()

model = None

@app.route("/v1/setup", methods=["POST"])
def setup():
    global model
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location=device))
    model.eval()
    model.to(device)
    response = {"setup": "Performed"}
    return jsonify(response), 201

@app.route('/v1/depth_estimation', methods=['POST'])
def depth_estimation():
    data = request.json
    image_url = data.get('image_url')
    prompt = data.get('prompt')

    if not image_url or not isinstance(image_url, str) or not image_url.strip():
        return jsonify({ "error": "Invalid image URL." }), 400
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return jsonify({ "error": "Invalid prompt." }), 400
    
    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({"error": "Unable to fetch the image from URL."}), 400

    image = np.asarray(bytearray(response.content), dtype="uint8")
    raw_img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    depth = model.infer_image(raw_img)
    depth = depth.squeeze().cpu().numpy()

    # Convert the depth map to a suitable format for Base64 encoding
    depth_min, depth_max = depth.min(), depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)
    depth_image = (depth * 255).astype(np.uint8)

    pil_image = Image.fromarray(depth_image)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    response = {
        "response": prompt,
        "file_attachments": [
            {
                "content": encoded_image,
                "file_name": "depth_estimation.png",
                "media_type": "image/png"
            }
        ]
    }
    return jsonify(response), 201

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
