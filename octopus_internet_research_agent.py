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

def step2(prompt: str, website_infos: []) -> str:
    content = str("I will give you summaries of homepages that eventually provide useful information for the subject of interest: \"" + prompt + "\" Make a plan how to come to a good answer to the subject of interest. ")
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

def step2scrape(query: str) -> []:
    query = query.replace(" ", "+")
    google_url = str("https://www.google.com/search?q=" + query)
    url = str("http://localhost:8080/api/v1/scraper?url=" + google_url)
    html_result = requests.get(url)
    soup = BeautifulSoup(html_result.text, "html.parser")

    links = []
    website_infos = []
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
        text = soup.get_text().replace("\n", "")
        text = text.replace("\t", " ")
        website_info = {
            "text": text,
            "url": url,
        }
        website_infos.append(website_info)

    return website_infos

def step3(prompt: str, website_infos: []) -> []:
    result_website_infos = []
    for website_info in website_infos:
        content = str("I will give you a subject of interest and the text of a homepage and you filter out marketing claims and spam. Make me  detailed report of all quantitative or qualitative information that are useful for the subject of interest or to answer the question in the subject. don't get distracted from the subject. Only use the information from the provided homepage.  When the homepage is marketing or spam, mark it clearly in the report, then this information is not very helpful. Just give me the detailed report, nothing else. Subject of interest: " + prompt + " Homepage: Please note: This website includes an accessibility system. Press Control-F11 to adjust the website to the visually impaired who are using a screen reader; Press Control-F10 to open an accessibility menu Accessibility. " + website_info["text"])
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=config["models"]["model"],
        )

        summary = chat_completion.choices[0].message.content

        result_website_info = {
            "text": website_info["text"],
            "summary": summary,
            "url": website_info["url"],
        }
        result_website_infos.append(result_website_info)

    return result_website_infos

def step4(prompt: str, strategy: str, website_infos: []) -> str:
    content = str("I give you a subject of interest. Give me an optimal answer to the subject of interest without relativizing. Refer your source of information to make clear, where you take your information from. Additionally I give you a strategy how to give the best answer and summaries of several homepage from google search that eventually provide useful information, so you do not have to look up the information yourself. But act like you looked up google by yourself. Therefor give academic  foodnotes like [1]  in your answer to refer the source of information. Just give me the conclusion and make the answer very clear and simple without referring the strategic help I give you. Subject of interest: " + prompt + " Strategy: " + strategy)

    i = 1
    for website_info in website_infos:
        content = str(content + " Summary of Homepage [" + str(i) + "] " + website_info["url"] + " " + website_info["summary"])
        i += 1

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

@app.route('/v1/internet_research_agent', methods=['POST'])
def internet_research_agent():
    data = request.json
    prompt = data.get("prompt", "")
    step1result = step1(prompt)
    website_infos = step2scrape(step1result)
    step2result = step2(prompt, website_infos)
    website_infos = step3(prompt, website_infos)
    step4result = step4(prompt, step2result, website_infos)

    response = {
        "step1result": step1result,
        "step2result": step2result,
        "response": step4result,
        "website_infos": website_infos,
    }

    return jsonify(response), 201

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }

    return jsonify(response), 201

app.run(host = "0.0.0.0", threaded=True, )
