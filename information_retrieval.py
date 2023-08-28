import os
os.environ["FLASK_ENV"] = "development"

dependencies = [
    'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
    'pip install transformers',
    'pip -q install langchain',
    'pip install -q flask',
    'pip install pypdf',
    'pip install cython',
    'pip install sentence_transformers',
    'pip install chromadb',
    'pip install -U accelerate',
    'pip install -U sentencepiece',
    'pip install flask pyngrok'
]

for command in dependencies:
    os.system(command)


# ---------------------------------------------------
# creating the configuration script
# ---------------------------------------------------
config_str = '''
{
   "device_map": {
    "cuda:0": "10GiB",
    "cuda:1": "10GiB",
    "cpu": "30GiB"
    },
    "models": [
        {
            "key": "BAAI/bge-small-en",
            "name": "embeddings_model",
            "access_token": "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"
        },
        {
            "key": "meta-llama/Llama-2-7b-chat-hf",
            "name": "llama_model",
            "access_token": "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"
        }
    ],
    "functions": [
        {
            "name": "process_llm_response",
            "description": "When user queries something, the AI first search into the chroma database to search for answers",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Answer to the query to search from the database"
                    },
                    "llm_response": {
                        "type": "json",
                        "description": "Answer to the query to search from the database"
                    },
                },
                "required": ["query"]
            },
            "input_type": "json",
            "return_type": "application/json"
        }
    ]
}
'''

# ---------------------------------------------------
# importing the reqiored libraries
# ---------------------------------------------------
import json
import time
import torch
import textwrap
import requests
import threading

from pyngrok import ngrok
from flask import Flask, request, jsonify

from transformers import pipeline
from configurations.prompt_config import PROMPT_CONFIG

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

constants = PROMPT_CONFIG()

# ---------------------------------------------------
# create the model manager class
# ---------------------------------------------------
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

        # form the model in sync for the query retrieval
        for model_info in self.config["models"]:
            
            # get the embeddings model to embed the pdfs in the folder
            if model_info["name"] == 'embeddings_model':
                loader      = DirectoryLoader('data/new_papers', glob="./*.pdf", loader_cls=PyPDFLoader)
                documents   = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                embed_model     = HuggingFaceBgeEmbeddings(model_name=model_info["key"], 
                                                           model_kwargs={'device': self.device}, 
                                                           encode_kwargs={'normalize_embeddings': True},
                                                           query_instruction="Generate a representation for this sentence for retrieving related articles: ")

                vectordb        = Chroma.from_documents(documents=texts, embedding=embed_model, persist_directory='db')
                retriever       = vectordb.as_retriever(search_kwargs={"k": 5})

                self.models[model_info["name"]] = retriever

            elif model_info['name'] == 'llama':
                pipe = pipeline("text-generation", model=model_info["key"], max_length=2048, temperature=0.75, top_p=0.95, repetition_penalty=1.2, device=-1, token='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')
                llm  = HuggingFacePipeline(pipeline=pipe)

                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.models['embeddings_model'], chain_type_kwargs=constants.chain_type_kwargs, return_source_documents=True)
                qa_chain.to(self.device)
                self.models[model_info["name"]] = qa_chain

        return True


    def infer(self, parameters):
        try:
            ### BEGIN USER EDITABLE SECTION ###
            query_model = self.models["llama"]
            
            llm_response = query_model(parameters['query'])
            torch.cuda.empty_cache() if self.device != "cpu" else None

            if 'llm_response' in parameters:
                return process_llm_response(llm_response)
            ### END USER EDITABLE SECTION ###
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

# ---------------------------------------------------
# load configurations for the program block
# ---------------------------------------------------
config          = json.loads(config_str)
model_manager   = ModelManager(config)

# ---------------------------------------------------
# make the ap and the corresponding function
# ---------------------------------------------------
app     = Flask(__name__)


# startup the application
# ---------------------------------------------------
@app.route('/setup', methods=['GET'])
def setup():
    model_manager.setup()
    return jsonify({"status": "models loaded successfully"})

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
        return app.response_class(result, content_type=function_config["return_type"])
    else:
        return jsonify({"error": "Error during inference"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Generic exception handler
    return jsonify(error=str(e)), 500

# ---------------------------------------------------
# chat completion functions
# ---------------------------------------------------
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    unique_sources = set()

    for source in llm_response["source_documents"]:
        unique_sources.add(source.metadata['source'])

    for source in unique_sources:
        print(source)