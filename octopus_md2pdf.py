import os

dependencies = [
    'pip install -q beautifulsoup4==4.13.3',
    "pip install -q Flask==3.1.0",
    "pip install -q Markdown==3.7",
    "pip install -q reportlab==4.3.1",
    "pip install -q requests==2.32.3",
]

for command in dependencies:
    os.system(command)

import base64
import json
import requests
from flask import Flask, jsonify, request
from bs4 import BeautifulSoup
from markdown import markdown
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

config_str = '''{
    "device_map": {
        "cuda:0": "10GiB",
        "cpu": "30GiB"
    },
    "required_python_version": "cp312",
    "functions": [
        {
            "name": "md2pdf",
            "display_name": "Convert Markdown from JSON file to PDF",
            "description": "This function converts given Markdown file to PDF file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL of JSON file" }
                },
                "required": ["url"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        }
    ]
}'''

config = json.loads(config_str)
app = Flask(__name__)

def extract_markdown_from_json(json_file, markdown_key="book"):
    """
    Extracts Markdown text from a JSON file.

    :param json_file: Path to the JSON file.
    :param markdown_key: The key in JSON where the Markdown content is stored.
    :return: Markdown content as a string.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if markdown_key not in data:
        raise ValueError(f"Key '{markdown_key}' not found in JSON. Available keys: {list(data.keys())}")

    return data[markdown_key]

def convert_markdown_to_pdf(markdown_text, output_pdf):
    """
    Converts Markdown text to a formatted PDF with proper headers, spacing, and styles.

    :param markdown_text: The Markdown content as a string.
    :param output_pdf: Path to save the output PDF.
    """
    # Convert Markdown to HTML
    html_content = markdown(markdown_text)

    # Extract text while keeping formatting
    soup = BeautifulSoup(html_content, "html.parser")

    # Create PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    for tag in soup.contents:
        if tag.name == "h1":
            elements.append(Paragraph(f"<font size=18><b>{tag.get_text()}</b></font>", styles["Title"]))
        elif tag.name == "h2":
            elements.append(Paragraph(f"<font size=16><b>{tag.get_text()}</b></font>", styles["Heading2"]))
        elif tag.name == "h3":
            elements.append(Paragraph(f"<font size=14><b>{tag.get_text()}</b></font>", styles["Heading3"]))
        elif tag.name == "p":
            elements.append(Paragraph(tag.get_text(), styles["BodyText"]))
        elif tag.name == "ul":
            for li in tag.find_all("li"):
                elements.append(Paragraph(f"• {li.get_text()}", styles["BodyText"]))
        elif tag.name == "strong":
            elements.append(Paragraph(f"<b>{tag.get_text()}</b>", styles["BodyText"]))
        elements.append(Spacer(1, 12))  # Add spacing

    doc.build(elements)
    print(f"✅ PDF successfully created: {output_pdf}")

@app.route('/v1/md2pdf', methods=['POST'])
def md2pdf():
    data = request.json
    json_url = data.get("url")

    try:
        response = requests.get(json_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

    json_file_path = f"/tmp/temp_{os.urandom(8).hex()}.json"
    with open(json_file_path, 'wb') as json_file:
        json_file.write(response.content)

    output_pdf_path = f"/tmp/temp_{os.urandom(8).hex()}.pdf"

    try:
        markdown_text = extract_markdown_from_json(json_file_path)
        convert_markdown_to_pdf(markdown_text, output_pdf_path)
    except Exception as e:
        print(f"❌ Error: {e}")

    file = open(output_pdf_path, mode='rb')
    fcontent = file.read()
    file.close()

    encoded_content = base64.b64encode(fcontent).decode('utf-8')

    response = {
        "file_attachments": [
            {
                "content": encoded_content,
                "file_name": "generated.pdf",
                "media_type": "application/pdf"
            }
        ]
    }
    return jsonify(response), 201

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }
    return jsonify(response), 201

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
