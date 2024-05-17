import os

dependencies = [
    'pip install -q beautifulsoup4==4.12.3',
    'pip install -q Flask==3.0.3',
    'pip install -q openai==1.30.1',
    'pip install -q requests==2.31.0',
]

for command in dependencies:
    os.system(command)

import json, re, requests, textwrap, time
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from openai import OpenAI

config_str = '''{
    "device_map": {
        "cpu": "15GiB"
    },
    "required_python_version": "cp311",
    "models": {
        "model": "gpt-4o-2024-05-13"
    },
    "functions": [
        {
            "name": "internet_research_agent",
            "description": "Internet research agent allows user perform research in the internet. Use this function only when user wants to check internet.",
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
    print("step1")
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
    print("step2")
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
    print("step2scrape")
    links = []
    website_infos = []

    query = query.replace(" ", "+")
    url = str("http://localhost:8080/api/v1/scraper-search-service?prompt=" + query)

    while True:
        try:
            print("request")
            print("url")
            print(url)
            html_result = requests.get(url)
            links_json = html_result.json()
            if len(links_json) > 0:
                links = links_json
                break
        except:
            print("sleeping")
            time.sleep(5)

    print("links")
    print(links)
    for link in links:
        url = str("http://localhost:8080/api/v1/scraper-service?url=" + link)
        print(url)
        html_result = requests.get(url)
        soup = BeautifulSoup(html_result.text, "html.parser")
        text = soup.get_text().replace("\n", "")
        text = text.replace("\t", " ")
        website_info = {
            "link": link,
            "text": text,
            "url": url,
        }
        website_infos.append(website_info)

    return website_infos

def step3(prompt: str, website_infos: []) -> []:
    print("step3")
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
            "link": website_info["link"],
            "text": website_info["text"],
            "summary": summary,
            "url": website_info["url"],
        }
        result_website_infos.append(result_website_info)
    print("result_website_infos")
    print(result_website_infos)
    return result_website_infos

def step4(prompt: str, strategy: str, website_infos: []) -> str:
    print("step4")
    content = str("I give you a subject of interest, a strategy and several source of information. \n\nGive me an optimal answer to the subject of interest without relativizing by using the provided source of information with academic footnotes like [1] followed by the corresponding URL. Just give me the conclusion and make the answer very clear and simple without referring the strategy again. Don't explain yourself. \n\nSubject of interest:\n "+ prompt + "\n\nStrategy:\n")

    i = 1
    for website_info in website_infos:
        content = str(content + " \n\n\n\nURL" + str(i) + " \n" + website_info["link"] + " \n\nSummary of URL" + str(i) + " \n" + website_info["summary"])
        i += 1
    print("content")
    print(content)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Reply in the language the user question is asked. Don't halucinate links. Use only provided links. Try to create longer report.",
            },
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
    print("internet_research_agent")
    data = request.json
    prompt = data.get("prompt", "")
    print("prompt")
    print(prompt)
    step1result = step1(prompt)
    website_infos = step2scrape(step1result)
    step2result = step2(prompt, website_infos)
    website_infos = step3(prompt, website_infos)
    step4result = step4(prompt, step2result, website_infos)

    response = {
        "response": step4result,
    }

    return jsonify(response), 201

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }

    return jsonify(response), 201

app.run(host = "0.0.0.0", threaded=True)
