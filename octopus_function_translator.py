import base64, os, uuid
from datetime import datetime, timedelta
from flask import Flask, jsonify, request

from translator import Translator

app = Flask(__name__)

translator = Translator()

def get_estimated_response_at(seconds: int) -> str:
    estimated_response_at = str(datetime.now() + timedelta(seconds))
    estimated_response_at = estimated_response_at + "Z"
    estimated_response_at = estimated_response_at.replace(" ", "T")

    return estimated_response_at

@app.route("/v1/function-translator/setup", methods=["GET"])
def function_translator_setup_status():
    if translator.function_translator_setup_condition():
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

    if translator.function_translator_setup_condition() or force_setup:
        translator.function_translator_setup_execute()

    translator.function_translator_prepare()

    return {
        "setup": "Performed"
    }, 201

@app.route("/v1/function-translator", methods=["POST"])
def function_translator():
    data = request.json
    model_name = "3.3B"
    selection_mode = "Manually select"
    sentence_mode = "Sentence-wise"
    source = data.get("source_language", "")
    target = data.get("target_language", "")
    text = data.get("text", "")

    id = str(uuid.uuid4())
    status = "Initial"
    response_text = ""
    estimated_response_at = get_estimated_response_at(seconds=25)

    response = {
        "id": id,
        "estimated_response_at": estimated_response_at,
        "progress": 0,
        "status": status,
        "response": response_text,
        "file_attachements": list()
    }

    translator.function_translator_store_response(id, response)

    translator.function_translator_start_thread(model_name, sentence_mode, selection_mode, source, target, text, id)

    return jsonify(response), 201

@app.route("/v1/function-translator/<string:id>", methods=["GET"])
def function_translator_status(id):
    response = translator.function_translator_get_response(id)

    return jsonify(response), 200

@app.route("/v1/health-check", methods=["GET"])
def health_check():
    return {
        "status": "Ok"
    }, 200
