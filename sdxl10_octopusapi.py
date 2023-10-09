import os
#os.environ["FLASK_ENV"] = "development"

### BEGIN USER EDITABLE SECTION ###

dependencies = [
    'pip install -q torch --index-url https://download.pytorch.org/whl/cu118',
    'pip install -q torchvision --index-url https://download.pytorch.org/whl/cu118',
    'pip install -q torchaudio --index-url https://download.pytorch.org/whl/cu118',
    'pip install -q python-dotenv',
    'pip install -q accelerate',
    'pip install -q transformers',
    'pip install -q invisible-watermark',
    'pip install -q numpy>=1.17',
    'pip install -q PyWavelets>=1.1.1',
    'pip install -q opencv-python>=4.1.0.25',
    'pip install -q safetensors',
    'pip install -q xformers==0.0.20',
    'pip install -q git+https://github.com/huggingface/diffusers.git',
    'pip install -q flask pyngrok'
]

for command in dependencies:
    os.system(command)


from diffusers import DiffusionPipeline
import torch

config_str = '''
{
   "device_map": {
    "cuda:0": "10GiB",
    "cuda:1": "10GiB",
    "cpu": "30GiB"
    },
    "required_python_version": "cp311",
    "models": [
        {
            "key": "stabilityai/stable-diffusion-xl-base-1.0",
            "name": "base_model",
            "access_token": "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"
        },
        {
            "key": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "name": "refiner_model",
            "access_token": "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"
        }
    ],
    "functions": [
        {
            "name": "text-to-image",
            "description": "Generates an image with SDXL based on a positive and negative prompt to describe what should be on the image and what not",
            "parameters": {
                "type": "object",
                "properties": {
                    "value1": {
                        "type": "string",
                        "description": "the positive prompt for image generation"
                    },
                    "value2": {
                        "type": "string",
                        "description": "the negative prompt for image generation"
                    }
                },
                "required": ["value1", "value2"]
            },
            "input_type": "json",
            "return_type": "image/jpeg"
        }
    ]
}
'''
### END USER EDITABLE SECTION ###

import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify


config = json.loads(config_str)
app = Flask(__name__)

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.device = self.select_device()

    def select_device(self):
        if not torch.cuda.is_available():
            return "cpu"

        device_map = self.config.get('device_map', {})
        available_devices = list(device_map.keys())
        return available_devices[0] if available_devices else "cpu"

    def setup(self):
        self.models.clear()

        # Loading models from configuration
        for model_info in self.config["models"]:
            model_key = model_info["key"]
            model_name = model_info["name"]
            model_access_token = model_info["access_token"]

            try:
                ### BEGIN USER EDITABLE SECTION ###
                model = DiffusionPipeline.from_pretrained(model_key,torch_dtype=torch.bfloat16, use_auth_token=model_access_token)
                model.to(self.device)
                model.enable_xformers_memory_efficient_attention()
                self.models[model_name] = model
                ### END USER EDITABLE SECTION ###
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

    def infer(self, parameters):
        try:
            ### BEGIN USER EDITABLE SECTION ###
            base_model = self.models["base_model"]
            refiner_model = self.models["refiner_model"]

            images = base_model(prompt=[parameters["value1"]], negative_prompt=[parameters["value2"]], num_inference_steps=50).images
            torch.cuda.empty_cache() if self.device != "cpu" else None

            images = refiner_model(prompt=[parameters["value1"]], negative_prompt=[parameters["value2"]], image=images, num_inference_steps=50).images
            torch.cuda.empty_cache() if self.device != "cpu" else None

            if config["functions"][0]["return_type"] == "image/jpeg":
                buffered = BytesIO()
                images[0].save(buffered, format="JPEG")
                return buffered.getvalue()
            ### END USER EDITABLE SECTION ###
        except Exception as e:
            print(f"Error during inference: {e}")
            return None


model_manager = ModelManager(config)

@app.route('/v1/setup', methods=['POST'])
def setup():
    model_manager.setup()
    return jsonify({"status": "models loaded successfully", "setup": "Performed"}), 201

@app.route('/v1/<function_name>', methods=['POST'])
def generic_route(function_name):
    function_config = next((f for f in config["functions"] if f["name"] == function_name), None)

    if not function_config:
        return jsonify({"error": "Invalid endpoint"}), 404

    if function_config["input_type"] != "json":
        return jsonify({"error": f"Unsupported input type {function_config['input_type']}"}), 400

    data = request.json
    parameters = {k: data[k] for k in function_config["parameters"]["properties"].keys() if k in data}

    result = model_manager.infer(parameters)
    if result:
        return app.response_class(result, content_type=function_config["return_type"]), 201
    else:
        return jsonify({"error": "Error during inference"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Generic exception handler
    return jsonify(error=str(e)), 500
'''
import threading
from pyngrok import ngrok
import time

# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

# Set up Ngrok to create a tunnel to the Flask server
public_url = ngrok.connect(5000).public_url

function_names = [func['name'] for func in config["functions"]]

print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{5000}/\"")

# Loop over function_names and print them
for function_name in function_names:
    time.sleep(5)
    print(f'Endpoint here: {public_url}/{function_name}')

import requests

BASE_URL = f"{public_url}"


### BEGIN USER EDITABLE SECTION ###
def setup_test():
    response = requests.get(f"{BASE_URL}/setup")
    
    # Check if the request was successful
    if response.status_code == 200:
        return (True, response.json())  # True indicates success
    else:
        return (False, response.json())  # False indicates an error

def infer_test(positive_prompt="a cyberpunk logo with a brain", negative_prompt="bad quality"):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "value1": positive_prompt,
        "value2": negative_prompt
    }
    response = requests.post(f"{BASE_URL}/text-to-image", headers=headers, json=data)
    
    if response.status_code == 200:
        # Save the image to a file
        with open("output_image.jpeg", "wb") as img_file:
            img_file.write(response.content)
        print("Image saved as output_image.jpeg!")
        return (True, "Image saved successfully!")  # True indicates success
    else:
        return (False, response.json())  # False indicates an error

### END USER EDITABLE SECTION ###

# Testing
result_setup = setup_test()
print(result_setup)

result_infer = infer_test()
print(result_infer)
'''
