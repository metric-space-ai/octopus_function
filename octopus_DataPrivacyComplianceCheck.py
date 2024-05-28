import os
#os.environ["FLASK_ENV"] = "development"

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
    "pip install -q torch==2.2.2",
    "pip install -q torchaudio==2.2.2",
    "pip install -q torchvision==0.17.2",
    "pip install -q python-dotenv==1.0.0",
    "pip install -q accelerate==0.23.0",
    "pip install -q transformers==4.34.0",
    "pip install -q safetensors==0.4.0",
    "pip install -q flask==3.0.0",
    "pip install -q pyngrok==7.0.0",

]

for command in dependencies:
    os.system(command)


import torch


config_str = """
{
   "device_map": {
    "cuda:0": "15GiB",
    "cuda:1": "15GiB",
    "cuda:2": "15GiB",
    "cuda:3": "15GiB"
    },
    "required_python_version": "cp311",
    "models": {
        "key": "metricspace/DataPrivacyComplianceCheck-3B-V0.9",
        "name": "base_model",
        "access_token": "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"          
    },
    "tokenizer": {
        "key": "metricspace/DataPrivacyComplianceCheck-3B-V0.9",
        "name": "tokenizer"
    },
    "functions": [{
        "name": "sensitive-information",
        "display_name": "Sensitive information filter",
        "description": "Predicts if the user input is sensitive and shows sensitive info",
        "parameters": {
            "type": "object",
            "properties": {
                "value1": {
                    "type": "string",
                    "description": "prompt - check for sensitive information"
                }
            },
            "required": ["value1"]
        },
        "input_type": "json",
        "return_type": "application/json"
    }]
}
"""



### END USER EDITABLE SECTION ###
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import re
import subprocess

def transform_output(output):
    print(output)
    response = output["response"]
    sensitive_part = re.search(r"Sensitive Information:(.*?)</s>", response, re.IGNORECASE)
    
    if sensitive_part:
        return {
            "is_sensitive": True,
            "sensitive_part": sensitive_part.group(1).strip()
        }
    else:
        return {
            "is_sensitive": False,
            "sensitive_part": None
        }


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
        model_info = self.config["models"]
        model_key = model_info["key"]
        model_name = model_info["name"]
        model_access_token = model_info["access_token"]


        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_key, torch_dtype=torch.bfloat16, use_auth_token=model_access_token)
            tokenizer = AutoTokenizer.from_pretrained(model_key, use_auth_token=model_access_token)
            model.to(self.device)
            #model.enable_xformers_memory_efficient_attention()
            #self.tokenizer[tokenizer_name] = tokenizer
            self.models[model_name] = model
            self.models['tokenizer'] = tokenizer
            ### END USER EDITABLE SECTION ###

        except Exception as e:
        
            print(f"Error loading model {model_name}: {e}")

    def infer(self, parameters):
        import traceback  # Importing the traceback module
    
        try:
            ### BEGIN USER EDITABLE SECTION ###
            base_model = self.models['base_model']
            tokenizer = self.models['tokenizer']
            inputs = tokenizer(f'Check for sensitive information: {parameters["value1"]}', return_tensors='pt').to(self.device)
            max_length = 512
            inputs_length = inputs.input_ids.shape[1]
            max_new_tokens_value = max_length - inputs_length

            outputs = base_model.generate(inputs.input_ids, max_new_tokens=max_new_tokens_value, do_sample=False, top_k=50, top_p=0.98)
            torch.cuda.empty_cache() if self.device != "cpu" else None
            response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
            response = {"response": response}
            response_json = transform_output(response)
            return json.dumps(response_json)
                    
            ### END USER EDITABLE SECTION ###
        except Exception as e:
            error_msg = f"Exception type: {type(e).__name__}\nException message: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)  # Print the detailed error message
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
        #return app.response_class(result, content_type=function_config["return_type"])
        return result, 201
    else:
        return jsonify({"error": "Error during inference"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Generic exception handler
    return jsonify(error=str(e)), 500

#import threading
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
    response = requests.post(f"{BASE_URL}/setup")
    
    # Check if the request was successful
    if response.status_code == 200:
        return (True, response.json())  # True indicates success
    else:
        return (False, response.json())  # False indicates an error

def infer_test(prompt="John, our patient, felt a throbbing headache and dizziness for two weeks. He was immediately advised to get a MRI scan of his brain. He reached out to Meed Hospital and got the scan done. The detailed report of his brain scan indicates a mass in the frontal lobe. His brain scan report accompanied by the MRI images include the identification number 12345 and is linked to his personal information. The doctor attending to him, Dr. Sally, is yet to discuss this with him in detail about his medical condition. In the following days, the healthcare team will chalk out a personalized treatment plan depending on John's medical history, condition and age."):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "value1": prompt,
    }
    response = requests.post(f"{BASE_URL}/sensitive-information", headers=headers, json=data)
    
    if response.status_code == 200:
        return (True, response.json())  # True indicates success
    else:
        return (False, response.json())  # False indicates an error

### END USER EDITABLE SECTION ###
time.sleep(5)
# Testing
result_setup = setup_test()
print(result_setup)

result_infer = infer_test()
print(result_infer[1])
