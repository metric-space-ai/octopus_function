import os

dependencies = [
    'pip install -q beautifulsoup4==4.12.3',
    'pip install -q Flask==3.0.2',
    'pip install -q openai==1.14.0',
    'pip install -q requests==2.31.0',
]

for command in dependencies:
    os.system(command)

import json, re, requests, textwrap
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from openai import OpenAI

config_str = '''{
    "device_map": {
        "cpu": "15GiB"
    },
    "required_python_version": "cp311",
    "models": {
        "model": "gpt-4-0125-preview"
    },
    "functions": [
        {
            "name": "internet_research_agent",
            "description": "Internet research agent allows user perform research in the internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "Full prompt string" }
                },
                "required": ["full_prompt"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        }
    ]}'''
config = json.loads(config_str)

app = Flask(__name__)

client = OpenAI()

def step1(prompt: str) -> str:
    content = str("I want to check the internet for the following thing. What is the best google search input to get the best results? Give me the google search input, nothing else. Here is the thing: " + prompt)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model=config["models"]["model"],
    )

    return chat_completion.choices[0].message.content

def step2(prompt: str, scrape_result: str) -> str:
    content = str("I will give you summaries of homepages that eventually provide useful information for the subject of interest: \"" + prompt + "\" Make a plan how to come to a good answer to the subject of interest. " + scrape_result)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model=config["models"]["model"],
    )

    return chat_completion.choices[0].message.content

def step2scrape(query: str) -> str:
    query = query.replace(" ", "+")
    google_url = str("https://www.google.com/search?q=" + query)
    url = str("http://localhost:8080/api/v1/scraper?url=" + google_url)
    html_result = requests.get(url)
    soup = BeautifulSoup(html_result.text, "html.parser")

    links = []
    website_content = []
    for link in soup.find_all('a'):
        link = link.get('href')

        if link and re.search("^http", link) and not re.search("google", link):
            links.append(link)

    if len(links) > 15:
        links = links[-15:]

    for link in links:
        url = str("http://localhost:8080/api/v1/scraper?url=" + link)
        html_result = requests.get(link)
        soup = BeautifulSoup(html_result.text, "html.parser")
        website_content.append(soup.get_text())
    
    result = ' '.join(website_content)
    result = result.replace("\n", "")
    result = textwrap.shorten(result, width=127000)

    return result

@app.route('/v1/internet-research-agent', methods=['POST'])
def internet_research_agent():
    data = request.json
    prompt = data.get("prompt", "")
    step1result = step1(prompt)
    step2scrape_result = step2scrape(step1result)
    step2result = step2(prompt, step2scrape_result)

    response = {
        "step1result": step1result,
        "step2scrape_result": step2scrape_result,
        "response": step2result,
    }

    return jsonify(response), 201

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }

    return jsonify(response), 201

app.run(host = "0.0.0.0", threaded=True, )
