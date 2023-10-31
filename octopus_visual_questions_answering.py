import os
### BEGIN USER EDITABLE SECTION ###
os.system('pip install -q tqdm==4.66.1')
from tqdm import tqdm
dependencies = [
    "pip install -q torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118",
    "pip install -q torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118",
    "pip install -q torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118",
    "pip install -q requests==2.31.0",
    "pip install -q python-dotenv==1.0.0",
    "pip install -q accelerate==0.23.0",
    "pip install -q transformers==4.34.0",
    "pip install -q invisible-watermark==0.2.0",
    "pip install -q numpy==1.26.0",
    "pip install -q PyWavelets==1.4.1",
    "pip install -q opencv-python==4.5.5.64",
    "pip install -q safetensors==0.4.0",
    "pip install -q xformers==0.0.22",
    "pip install -q diffusers==0.21.4",
    "pip install -q flask==3.0.0",
    "pip install -q pyngrok==7.0.0",
    "pip install -q salesforce-lavis==1.0.2",
    "pip install -q Pillow==10.1.0"
]

for command in tqdm(dependencies):
    os.system(command)

import requests
from diffusers import DiffusionPipeline
import torch
from lavis.models import load_model_and_preprocess
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
            "key": "pretrain_flant5xl",
            "name": "blip2_t5"
        }
    ],
    "functions": [
        {
            "name": "VQA",
            "description": "Visual question answering. Answers a question based on an image",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "str",
                        "description": "Image"
                    },
                    "prompt": {
                        "type": "str",
                        "description": "The question to the given image"
                    }
                },
                "required": ["image", "prompt"]
            },
            "input_type": "json",
            "return_type": "string"
        }
    ]
}
'''
### END USER EDITABLE SECTION ###

import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
import re

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

            try:
                ### BEGIN USER EDITABLE SECTION ###
                model, vis_processors, _ = load_model_and_preprocess(
                    name=model_name, model_type=model_key, is_eval=True, device=self.device)
                self.models[model_name] = [model, vis_processors]
                ### END USER EDITABLE SECTION ###
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

    def infer(self, parameters):
        try:
            ### BEGIN USER EDITABLE SECTION ###
            model, vis_processors = self.models["blip2_t5"]
            image_code = parameters["image"]


            # check if input image is base64 or url
            base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')

            if "http://" in image_code or "https://" in image_code:
                raw_image = Image.open(requests.get(image_code, stream=True).raw).convert('RGB')

            elif base64_pattern.match(image_code):
                decoded_image_data = base64.b64decode(image_code)
                # Create a Pillow Image object from the decoded binary data
                raw_image = Image.open(io.BytesIO(decoded_image_data))

            else:
                raise ValueError("Image not in base64 or URL format")

            prompt = parameters["prompt"]

            image = vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)



            output_text = model.generate({"image": image,
                            "prompt": prompt},
                            use_nucleus_sampling=False,
                            )

            return output_text[0]

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

#import threading
#from pyngrok import ngrok
#import time

# Start the Flask server in a new thread
#threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

# Set up Ngrok to create a tunnel to the Flask server
#public_url = ngrok.connect(5000).public_url

#function_names = [func['name'] for func in config["functions"]]

#print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{5000}/\"")

# Loop over function_names and print them
#for function_name in function_names:
#    time.sleep(5)
#    print(f'Endpoint here: {public_url}/{function_name}')

#import requests
#from PIL import Image
#import io


#BASE_URL = f"{public_url}"


### BEGIN USER EDITABLE SECTION ###
#def setup_test():

#    response = requests.get(f"{BASE_URL}/setup")

    # Check if the request was successful
#    if response.status_code == 200:

#        return (True, response.json())  # True indicates success
#    else:
#        return (False, response.json())  # False indicates an error

#def infer_test(img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png', prompt="which city is this?"):
    # convert image to base64
#    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
#    buffer = io.BytesIO()
#    raw_image.save(buffer, format="JPEG")
#    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # create promt
#    prompt = "Question: " + prompt + " Answer:"

#    headers = {
#        "Content-Type": "application/json"
#    }
#    data = {
#        "image": base64_image,
#        "prompt": prompt
#    }
#    response = requests.post(f"{BASE_URL}/VQA", headers=headers, json=data)

#    if response.status_code == 200:

        # Save the image to a file
#        with open("output_text.txt", "wb") as file:
#            file.write(response.content)
#        print("Answer saved as output_text.txt!")
#        return (True, response.content)  # True indicates success
#    else:
#        return (False, response.json())  # False indicates an error


#def infer_test_url(img_url='https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' , prompt="which city is this?"):
    # create promt
#    prompt = "Question: " + prompt + " Answer:"

#    headers = {
#        "Content-Type": "application/json"
#    }
#    data = {
#        "image": img_url,
#        "prompt": prompt
#    }
#    response = requests.post(f"{BASE_URL}/VQA", headers=headers, json=data)

#    if response.status_code == 200:

        # Save the image to a file
#        with open("output_text.txt", "wb") as file:
#            file.write(response.content)
#        print("Answer saved as output_text.txt!")
#        return (True, response.content)  # True indicates success
#    else:
#        return (False, response.json())  # False indicates an error
### END USER EDITABLE SECTION ###

# Testing
#result_setup = setup_test()

#result_infer = infer_test()
#print(result_infer)

#result_infer_url = infer_test_url("https://hips.hearstapps.com/hmg-prod/images/high-angle-view-of-tokyo-skyline-at-dusk-japan-royalty-free-image-1664309926.jpg?resize=2048:*", "What can i see in this image? Describe it in detail")
#print(result_infer_url)
