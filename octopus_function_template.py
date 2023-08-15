import base64, json, os, uuid
from flask import Flask, jsonify, request

import utils

app = Flask(__name__)

### Configuration section
config = '''{
    "device_map": {
       "max_memory": {"0": "10GiB", "1": "10GiB", "cpu": "30GiB"}
    },
    "model_setup": {
        "file": "model.dat"
    },
    "functions": [
        {
            "name": "function_bar_async",
            "url_part": "function-bar-async",
            "description": "Asynchronous communication test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "value1": {
                        "type": "string",
                        "description": "First value"
                    },
                    "value2": { "type": "string", "enum": ["abc", "def"], "description": "Second value" }
                },
                "required": ["value1", "value2"]
            }
        },
        {
            "name": "function_foo_sync",
            "url_part": "function-foo-sync",
            "description": "Synchronous communication test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "value1": {
                        "type": "string",
                        "description": "First value"
                    },
                    "value2": { "type": "string", "enum": ["abc", "def"], "description": "Second value" }
                },
                "required": ["value1", "value2"]
            }
        }
    ]}'''
config = json.loads(config)

### AI function section
file = config["model_setup"]["file"]
model = None
results = {}

def get_result(id):
    return results[id]

def set_result(id, response):
    results[id] = response

def setup():
# Start editing here
    f = open(file, "a")
    f.write("Lets create some model file")
    f.close()
# Finish editing here

def setup_condition() -> bool:
# Start editing here
    if not os.path.isfile(file):
        return True
# Finish editing here
    return False

# Please load all models to memory here
def warmup():
# Start editing here
    global model
    if model == None:
        model = True
# Finish editing here

def warmup_condition() -> bool:
# Start editing here
    if model == None:
        return True
# Finish editing here
    return False

### AI service section
@app.route("/v1/{url_part}/setup".format(url_part = config["functions"][0]["url_part"]), methods=["GET"])
def function_bar_async_setup_status():
    if setup_condition():
        return {
            "setup": "NotPerformed"
        }, 200

    return {
        "setup": "Performed"
    }, 200

@app.route("/v1/{url_part}/setup".format(url_part = config["functions"][0]["url_part"]), methods=["POST"])
def function_bar_async_setup():
    data = request.json
    force_setup = data.get("force_setup", False)

    if setup_condition() or force_setup:
        setup()

    return {
        "setup": "Performed"
    }, 201

@app.route("/v1/{url_part}/warmup".format(url_part = config["functions"][0]["url_part"]), methods=["GET"])
def function_bar_async_warmup_status():
    if warmup_condition():
        return {
            "warmup": "NotPerformed"
        }, 200

    return {
        "warmup": "Performed"
    }, 200

@app.route("/v1/{url_part}/warmup".format(url_part = config["functions"][0]["url_part"]), methods=["POST"])
def function_bar_async_warmup():
    warmup()

    return {
        "warmup": "Performed"
    }, 201

@app.route("/v1/{url_part}".format(url_part = config["functions"][0]["url_part"]), methods=["POST"])
def function_bar_async():
# Start editing here
    if setup_condition():
        setup()
    if warmup_condition():
        warmup()

    data = request.json
    device_map = data.get("device_map", "")
    value1 = data.get("value1", "")
    value2 = data.get("value2", "")

    id = str(uuid.uuid4())
    status = "Initial"
    response_text = "Some async response text " + value1 + " " + value2
    estimated_response_at = utils.get_estimated_response_at(5)

    response = {
        "id": id,
        "estimated_response_at": estimated_response_at,
        "progress": 0,
        "status": status,
        "response": response_text,
        "file_attachements": list()
    }

    set_result(id, response)

    return jsonify(response), 201
# Finish editing here

@app.route("/v1/{url_part}/<string:id>".format(url_part = config["functions"][0]["url_part"]), methods=["GET"])
def function_bar_async_status(id):
# Start editing here
    response = get_result(id)

    estimated_response_at = utils.get_estimated_response_at(5)

    if response["progress"] < 25:
        response["estimated_response_at"] = estimated_response_at
        response["progress"] = 25
        response["status"] = "Processing"
    elif response["progress"] < 50:
        response["estimated_response_at"] = estimated_response_at
        response["progress"] = 50
    elif response["progress"] < 75:
        response["estimated_response_at"] = estimated_response_at
        response["progress"] = 75
    elif response["progress"] < 100:
        response["estimated_response_at"] = estimated_response_at
        response["progress"] = 100
        response["status"] = "Processed"
        content = str(base64.b64encode(b"test of generated content"))
        file_attachement = {
            "content": content,
            "file_name": "test.txt",
            "media_type": "text/plain",
        }
        file_attachements = list()
        file_attachements.append(file_attachement)
        response["file_attachements"] = file_attachements

    set_result(id, response)

    return jsonify(response), 200
# Finish editing here

@app.route("/v1/{url_part}/setup".format(url_part = config["functions"][1]["url_part"]), methods=["GET"])
def function_foo_sync_setup_status():
    if setup_condition():
        return {
            "setup": "NotPerformed"
        }, 200

    return {
        "setup": "Performed"
    }, 200

@app.route("/v1/{url_part}/setup".format(url_part = config["functions"][1]["url_part"]), methods=["POST"])
def function_foo_sync_setup():
    data = request.json
    force_setup = data.get("force_setup", False)

    if setup_condition() or force_setup:
        setup()

    return {
        "setup": "Performed"
    }, 201

@app.route("/v1/{url_part}/warmup".format(url_part = config["functions"][1]["url_part"]), methods=["GET"])
def function_foo_sync_warmup_status():
    if warmup_condition():
        return {
            "warmup": "NotPerformed"
        }, 200

    return {
        "warmup": "Performed"
    }, 200

@app.route("/v1/{url_part}/warmup".format(url_part = config["functions"][1]["url_part"]), methods=["POST"])
def function_foo_sync_warmup():
    warmup()

    return {
        "warmup": "Performed"
    }, 201

@app.route("/v1/{url_part}".format(url_part = config["functions"][1]["url_part"]), methods=["POST"])
def function_foo_sync():
# Start editing here
    if setup_condition():
        setup()
    if warmup_condition():
        warmup()

    data = request.json
    device_map = data.get("device_map", "")
    value1 = data.get("value1", "")
    value2 = data.get("value2", "")

    id = str(uuid.uuid4())
    status = "Processed"
    response_text = "Some sync response text " + value1 + " " + value2
    estimated_response_at = utils.get_estimated_response_at(5)

    response = {
        "id": id,
        "estimated_response_at": estimated_response_at,
        "progress": 100,
        "status": status,
        "response": response_text,
        "file_attachements": list()
    }

    set_result(id, response)

    return jsonify(response), 201
# Finish editing here

@app.route("/v1/{url_part}/<string:id>".format(url_part = config["functions"][1]["url_part"]), methods=["GET"])
def function_foo_sync_status(id):
# Start editing here
    response = get_result(id)

    return jsonify(response), 200
# Finish editing here

@app.route("/v1/health-check", methods=["GET"])
def health_check():
    return {
        "status": "Ok"
    }, 200
