import os

### BEGIN USER EDITABLE SECTION ###
dependencies = [
    "pip install -q nvidia-cublas-cu12==12.1.3.1",
    "pip install -q nvidia-cuda-cupti-cu12==12.1.105",
    "pip install -q nvidia-cuda-nvrtc-cu12==12.1.105",
    "pip install -q nvidia-cuda-runtime-cu12==12.1.105",
    "pip install -q nvidia-cudnn-cu12==8.9.2.26",
    "pip install -q nvidia-cufft-cu12==11.0.2.54",
    "pip install -q nvidia-curand-cu12==10.3.2.106",
    "pip install -q nvidia-cusolver-cu12==11.4.5.107",
    "pip install -q nvidia-cusparse-cu12==12.1.0.106",
    "pip install -q nvidia-nccl-cu12==2.19.3",
    "pip install -q nvidia-nvjitlink-cu12==12.4.127",
    "pip install -q nvidia-nvtx-cu12==12.1.105",
    "pip install -q numpy==1.26.4",
    "pip install -q torch==2.2.2",
    "pip install -q torchaudio==2.2.2",
    "pip install -q torchvision==0.17.2",
    'pip install -q diffusers==0.21.4',
    'pip install -q accelerate==0.24.0',
    'pip install -q transformers==4.34.1',
    'pip install -q flask==2.2.5',
    'pip install -q pyngrok==7.0.0'
]

for command in dependencies:
    os.system(command)

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
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
            "key": "madebyollin/sdxl-vae-fp16-fix",
            "name": "VAE",
            "access_token": "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"
        },
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
            "display_name": "Text to image",
            "description": "Generates an image based on a positive and negative prompt",
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
import subprocess

config = json.loads(config_str)
app = Flask(__name__)

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.device = self.select_device()

    def command_result_as_int(self, command):
        return int(subprocess.check_output(command, shell=True).decode('utf-8').strip())

    def select_device_with_larger_free_memory(self, available_devices):
        device = None
        memory = 0

        for available_device in available_devices:
            id = available_device.split(":")
            id = id[-1]
            free_memory = self.command_result_as_int(f"nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader --id={id}")
            if free_memory > memory:
                memory = free_memory
                device = available_device

        return device if device else "cpu"

    def select_device(self):
        if not torch.cuda.is_available():
            return "cpu"

        device_map = self.config.get('device_map', {})
        available_devices = list(device_map.keys())
        return self.select_device_with_larger_free_memory(available_devices)

    def setup(self):
        self.models.clear()

        # Loading models from configuration
        for model_info in self.config["models"]:
            model_key = model_info["key"]
            model_name = model_info["name"]
            model_access_token = model_info["access_token"]

            try:
                ### BEGIN USER EDITABLE SECTION ###

                if model_name =="VAE":
                  model = AutoencoderKL.from_pretrained(model_key, torch_dtype=torch.float16)

                elif model_name == "base_model":
                  model = StableDiffusionXLPipeline.from_pretrained(
                        model_key,
                        vae=self.models["VAE"],
                        torch_dtype=torch.float16, variant="fp16",
                        use_safetensors=True,
                        add_watermarker=False,
                        use_auth_token = model_access_token
                  )
                  model.enable_vae_slicing()
                  model.enable_vae_tiling()

                  # performant sampler so we can use just 20 iterations
                  model.scheduler = DPMSolverMultistepScheduler.from_config(
                      model.scheduler.config,
                      algorithm_type="sde-dpmsolver++",
                      use_karras_sigmas=True)
                  
                elif model_name == "refiner_model":
                    model = DiffusionPipeline.from_pretrained(
                        model_key,
                        text_encoder_2=self.models["base_model"].text_encoder_2,
                        vae=self.models["VAE"],
                        torch_dtype=torch.float16, variant="fp16",
                        use_safetensors=True,
                        add_watermarker=False,
                        use_auth_token = model_access_token
                    )

                    model.scheduler = self.models["base_model"].scheduler

                model.to(self.device)
                self.models[model_name] = model
                ### END USER EDITABLE SECTION ###
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

    def infer(self, parameters):
        try:
            ### BEGIN USER EDITABLE SECTION ###
            base_model = self.models["base_model"]
            refiner_model = self.models["refiner_model"]

            images = base_model(prompt=[parameters["value1"]], negative_prompt=[parameters["value2"]], num_inference_steps=20).images
            torch.cuda.empty_cache() if self.device != "cpu" else None

            images = refiner_model(prompt=[parameters["value1"]], negative_prompt=[parameters["value2"]], image=images, num_inference_steps=20,denoising_start=0.8).images
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

@app.route('/setup', methods=['GET'])
def setup():
    model_manager.setup()
    return jsonify({"status": "models loaded successfully"})

@app.route('/<function_name>', methods=['POST'])
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
        return app.response_class(result, content_type=function_config["return_type"])
    else:
        return jsonify({"error": "Error during inference"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Generic exception handler
    return jsonify(error=str(e)), 500

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

def infer_test(positive_prompt="a cyberpunk icon of a octopus", negative_prompt="bad quality"):
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



#plotting only for notebook
from IPython.display import Image, display

display(Image(filename='/content/output_image.jpeg'))
