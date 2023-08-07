import base64, os, uuid
from datetime import datetime, timedelta
from flask import Flask, jsonify, request

app = Flask(__name__)

results = {}

def get_estimated_response_at(seconds: int) -> str:
    estimated_response_at = str(datetime.now() + timedelta(seconds))
    estimated_response_at = estimated_response_at + "Z"
    estimated_response_at = estimated_response_at.replace(" ", "T")

    return estimated_response_at

@app.route("/v1/function-bar-async/setup", methods=["GET"])
def function_bar_async_setup_status():
    file = "model.dat"

    if function_bar_async_setup_condition(file):
        return {
            "setup": "NotPerformed"
        }, 200

    return {
        "setup": "Performed"
    }, 200

@app.route("/v1/function-bar-async/setup", methods=["POST"])
def function_bar_async_setup():
    data = request.json
    force_setup = data.get("force_setup", False)
    file = "model.dat"

    if function_bar_async_setup_condition(file) or force_setup:
        f = open(file, "a")
        f.write("Lets create some model file")
        f.close()

    return {
        "setup": "Performed"
    }, 201

def function_bar_async_setup_condition(file: str):
    if not os.path.isfile(file):
        return True

    return False

@app.route("/v1/function-bar-async", methods=["POST"])
def function_bar_async():
    data = request.json
    value1 = data.get("value1", "")
    value2 = data.get("value2", "")

    id = str(uuid.uuid4())
    status = "Initial"
    response_text = "Some async response text " + value1 + " " + value2
    estimated_response_at = get_estimated_response_at(seconds=5)

    response = {
        "id": id,
        "estimated_response_at": estimated_response_at,
        "progress": 0,
        "status": status,
        "response": response_text,
        "file_attachements": list()
    }

    results[id] = response

    return jsonify(response), 201

@app.route("/v1/function-bar-async/<string:id>", methods=["GET"])
def function_bar_async_status(id):
    response = results[id]

    estimated_response_at = get_estimated_response_at(seconds=5)

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

    results[id] = response

    return jsonify(response), 200

@app.route("/v1/function-foo-sync/setup", methods=["GET"])
def function_foo_sync_setup_status():
    file = "model.dat"

    if function_foo_sync_setup_condition(file):
        return {
            "setup": "NotPerformed"
        }, 200

    return {
        "setup": "Performed"
    }, 200

@app.route("/v1/function-foo-sync/setup", methods=["POST"])
def function_foo_sync_setup():
    data = request.json
    force_setup = data.get("force_setup", False)
    file = "model.dat"

    if function_foo_sync_setup_condition(file) or force_setup:
        f = open(file, "a")
        f.write("Lets create some model file")
        f.close()

    return {
        "setup": "Performed"
    }, 201

def function_foo_sync_setup_condition(file: str):
    if not os.path.isfile(file):
        return True

    return False

@app.route("/v1/function-foo-sync", methods=["POST"])
def function_foo_sync():
    data = request.json
    value1 = data.get("value1", "")
    value2 = data.get("value2", "")

    id = str(uuid.uuid4())
    status = "Processed"
    response_text = "Some sync response text " + value1 + " " + value2
    estimated_response_at = get_estimated_response_at(seconds=5)

    response = {
        "id": id,
        "estimated_response_at": estimated_response_at,
        "progress": 100,
        "status": status,
        "response": response_text,
        "file_attachements": list()
    }

    results[id] = response

    return jsonify(response), 201

@app.route("/v1/function-foo-sync/<string:id>", methods=["GET"])
def function_foo_sync_status(id):
    response = results[id]

    return jsonify(response), 200

@app.route("/v1/health-check", methods=["GET"])
def health_check():
    return {
        "status": "Ok"
    }, 200
