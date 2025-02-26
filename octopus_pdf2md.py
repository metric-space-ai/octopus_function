import os

# Install dependencies
dependencies = [
    "pip install -q regex",
    "pip install -q flask",
    "pip install -q transformers",
    "pip install -q torch",
    "pip install -q Pillow",
    "pip install -q marker-pdf",
    "pip install -q torchvision",
    "pip install -q langchain-community",
    "pip install -q werkzeug",
    ]

for command in dependencies:
    os.system(command)

import json
import re
import subprocess
import shutil
import torch
import tempfile

import threading
from langchain_community.llms import Ollama
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from flask import Flask, jsonify, request
import base64
import requests

config_str = '''{
  "device_map": {
    "cuda:0": "15GiB",
    "cuda:1": "15GiB",
    "cuda:2": "15GiB",
    "cuda:3": "15GiB"
  },
  "required_python_version": "cp311",
  "models": [
    {
      "name": "ollama:llama3.1:8b"
    }
  ],
  "functions": [
    {
      "name": "pdf2markdown",
      "description": "This function converts pdf files to the markdown",
      "parameters": {
        "type": "object",
        "properties": {
          "file": {
            "type": "string",
            "description": "PDF file for conversion"
          }
        }
      },
      "input_type": "application/pdf",
      "return_type": "application/json"
    }
  ]
}
'''

config = json.loads(config_str)

ollama_rag_name = "llama3.1:8b"
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:5050')

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

app = Flask(__name__)

def convertPDF2Markdown(pdfFile, output_dir):
    # Use output directory to store intermediate files
    pdf_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(pdfFile))[0])

    if not os.path.exists(pdf_output_path):
        os.makedirs(pdf_output_path)

    def image_to_caption(path):
        raw_image = Image.open(path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)

    def process_pdf(input_path, output_path, batch_multiplier=1, max_pages=800):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        command = [
            'marker_single',
            input_path,
            output_path,
            f'--batch_multiplier={batch_multiplier}',
            f'--max_pages={max_pages}'
        ]
        try:
            subprocess.run(command, check=True)
            print(f'Successfully processed {input_path}')
        except subprocess.CalledProcessError as e:
            print(f'Failed to process {input_path}: {e}')

    def describe_images_in_md(markdown_file_path):
        with open(markdown_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Regex to find image references
        image_ref_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

        def replace_image_ref(match):
            image_path = match.group(1)
            directory_path = os.path.dirname(markdown_file_path)
            image_path = image_path.lower()
            image_path = os.path.join(directory_path, image_path)
            if os.path.exists(image_path):
                caption = image_to_caption(image_path)
                return f"Attachment:[ A Photo of {caption} ]"
            else:
                print('Image does not exist')
                return match.group(0)

        new_content = re.sub(image_ref_pattern, replace_image_ref, content)
        return new_content

    # Process the PDF and generate the markdown with image descriptions
    process_pdf(pdfFile, pdf_output_path)

    markdown_file_path = None

    for root, dirs, files in os.walk(pdf_output_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            print(file_path)
            if file_name.endswith('.md'):
                markdown_file_path = file_path
                break

    if markdown_file_path:
        mdfile = describe_images_in_md(markdown_file_path)
    else:
        mdfile = ""

    # Remove intermediate files after processing
    shutil.rmtree(pdf_output_path)

    def generate_summary(chunk):
        prompt = f"Give me a summary from the document, do not explain yourself, just provide me with a fairly detailed summary of the text about what it is about, nothing else: \\n\\n {chunk}"
        cached_llm = Ollama(model=ollama_rag_name, base_url=ollama_host)
        summary = cached_llm.invoke(prompt)

        return summary

    summary = generate_summary(mdfile)
    json_output = {'markdown': mdfile, 'summary': summary}

    return json_output

@app.route('/pdf2markdown', methods=['POST'])
def upload_pdf():
    data = request.json  # URL
    base64_string = data['file']

    # Decode the base64 string to binary data
    pdf_data = base64.b64decode(base64_string)

    # Use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "decoded.pdf")
        with open(file_path, "wb") as file:
            file.write(pdf_data)

        print(f"File created at: {file_path}")
        try:
            result = convertPDF2Markdown(file_path, "/tmp")
            # No need to remove the file explicitly, it will be removed when the temporary directory is deleted
            # json_result = {'response': result}
            # return json_result, 200
            return jsonify({"response": json.dumps(result)}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def start_app():
    app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False

def run_flask_in_thread():
    flask_thread = threading.Thread(target=start_app)
    flask_thread.start()

def send_pdf_to_convert():
    # Function to convert PDF to base64
    def pdf_to_base64(pdf_file_path):
        with open(pdf_file_path, "rb") as pdf_file:
            base64_string = base64.b64encode(pdf_file.read()).decode('utf-8')
        return base64_string

    base64_string = pdf_to_base64("/home/metricspace/florence_doors/likora_all_650/likora_all/31010_0602_9000_Rotodecor_V2.pdf")
    endpoint_url = "http://192.168.1.13:5000/convert_pdf"

    #base64_string = "JVBERi0xLjQKJaqrrK0KNCAwIG9iaiA8PAovTGluZWFyaXplZCAxL0wgMTQxMjMvTyAxL0UgMTIwODMvTiA1L1QgMTM4NjEvSCBbIDk1MiAxMTFdCi9GIDExNjA4L0kgMTQxMTYKL0wgMTQxODMvTiAzL1QgMTM4NjEvSCBbIDk0MCA5NTJdCi9GIDMyOC9JIDE0MTI5Cj4+CmVuZG9iago="

    payload = {
        'file': base64_string
    }

    try:
        # Send POST request to the endpoint
        response = requests.post(endpoint_url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            print("PDF successfully converted.")
            return response.json()
        else:
            print(f"Error occurred: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        return None

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }
    return jsonify(response), 201

if __name__ == "__main__":
    # Run the Flask app in a separate thread
    run_flask_in_thread()

    # Wait a moment for the server to start
    import time
    time.sleep(2)

    # Now call the test function
    send_pdf_to_convert()
