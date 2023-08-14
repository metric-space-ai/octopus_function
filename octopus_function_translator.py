import uuid
from flask import Flask, jsonify, request

import utils
from translator import Translator

app = Flask(__name__)

translator = Translator()

@app.route("/v1/function-translator/setup", methods=["GET"])
def function_translator_setup_status():
    if translator.setup_condition():
        return {
            "setup": "NotPerformed"
        }, 200

    return {
        "setup": "Performed"
    }, 200

@app.route("/v1/function-translator/setup", methods=["POST"])
def function_translator_setup():
    data = request.json
    force_setup = data.get("force_setup", False)

    if translator.setup_condition() or force_setup:
        translator.setup()

    return {
        "setup": "Performed"
    }, 201

@app.route("/v1/function-translator/warmup", methods=["GET"])
def function_translator_warmup_status():
    if translator.warmup_condition():
        return {
            "warmup": "NotPerformed"
        }, 200

    return {
        "warmup": "Performed"
    }, 200

@app.route("/v1/function-translator/warmup", methods=["POST"])
def function_translator_warmup():
    translator.warmup()

    return {
        "warmup": "Performed"
    }, 201

@app.route("/v1/function-translator", methods=["POST"])
def function_translator():
    data = request.json
    model_name = "3.3B"
    selection_mode = "Manually select"
    sentence_mode = "Sentence-wise"
    device_map = data.get("device_map", "")
    source = data.get("source_language", "")
    target = data.get("target_language", "")
    text = data.get("text", "")

    id = str(uuid.uuid4())
    status = "Initial"
    response_text = None
    estimated_response_at = utils.get_estimated_response_at(25)

    response = {
        "id": id,
        "estimated_response_at": estimated_response_at,
        "progress": 0,
        "status": status,
        "response": response_text,
        "file_attachements": list()
    }

    translator.set_result(id, response)

    translator.function_translator_start_thread(model_name, sentence_mode, selection_mode, source, target, text, id)

    return jsonify(response), 201

@app.route("/v1/function-translator/<string:id>", methods=["GET"])
def function_translator_status(id):
    response = translator.get_result(id)

    return jsonify(response), 200

@app.route("/v1/health-check", methods=["GET"])
def health_check():
    return {
        "status": "Ok"
    }, 200
