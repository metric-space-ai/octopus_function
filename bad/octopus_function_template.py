import base64, json, os, uuid
from datetime import datetime, timedelta
from flask import Flask, jsonify, request

app = Flask(__name__)

### Configuration section
config = """{
    "device_map": {
        "cuda:0": "10GiB",
        "cuda:1": "10GiB",
        "cpu": "30GiB"
    },
    "model_setup": {
        "file": "model.dat",
        "model_call_name": "model3B",
        "model_real_name": "model/model-3B"
    },
    "functions": [
        {
            "name": "function-foo-sync",
            "description": "Synchronous communication test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "value1": { "type": "string", "description": "First value" },
                    "value2": { "type": "string", "enum": ["abc", "def"], "description": "Second value" }
                },
                "required": ["value1", "value2"]
            },
            "input_type": "json",
            "return_type": "application/json"
        }
    ]}"""
config = json.loads(config)

### AI function section
file = config["model_setup"]["file"]
model = None

@app.route("/function-foo-sync", methods=["POST"])
def function_foo_sync():
# Start editing here
    print(f"function-foo-sync called")
    data = request.json
    device_map = data.get("device_map", "")
    value1 = data.get("value1", "")
    value2 = data.get("value2", "")

    response_text = "Some sync response text " + value1 + " " + value2

    response = {
        "response": response_text,
    }

    return jsonify(response), 201
# Finish editing here

@app.route("/setup", methods=["POST"])
def setup():
# Start editing here
    print(f"setup called")
    global model
    data = request.json
    force_setup = data.get("force_setup", False)

    if not os.path.isfile(file) or force_setup:
        f = open(file, "a")
        f.write("Lets create some model file")
        f.close()

    if model == None:
        model = True
# Finish editing here
    response = {
        "setup": "Performed"
    }

    return jsonify(response), 201

# This should not be in the final template
