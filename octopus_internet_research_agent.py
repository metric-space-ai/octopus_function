import os

dependencies = [
    'pip install -q beautifulsoup4==4.12.3',
    'pip install -q Flask==3.0.3',
    'pip install -q openai==1.37.1',
    'pip install -q requests==2.32.3',
]

for command in dependencies:
    os.system(command)

import json, re, requests, textwrap, time
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from openai import AzureOpenAI, OpenAI

config_str = '''{
    "device_map": {
        "cpu": "15GiB"
    },
    "required_python_version": "cp311",
    "models": {
        "model": "gpt-4o-mini-2024-07-18"
    },
    "functions": [
        {
            "name": "internet_research_agent",
            "display_name": "Internet research agent",
            "description": "The Internet Research Agent performs comprehensive online research based on a user's query. Use this function for tasks such as gathering detailed information, comparing products, or finding the latest research on a topic. Examples: (1) Comparing features of different industrial printers; (2) Finding the latest studies on coating technologies; (3) Gathering market trends in the printing industry. Use this function only when user wants to check internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_prompt": { "type": "string", "description": "Full prompt created by the user" }
                },
                "required": ["full_prompt"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        },
        {
            "name": "internet_research_urls",
            "display_name": "Internet research urls",
            "description": "The Internet Research Urls performs comprehensive online research based on a user's query with given urls. Use this function only when user wants to check internet. This function should be used when user wants action like {do something with} {url}. Or multiple actions like {do something1} {url1} {do something2} {url2} {do something3} {url3}",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_prompt_with_urls": { "type": "string", "description": "Full prompt created by the user with urls" }
                },
                "required": ["full_prompt_with_urls"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        }
    ]}'''
config = json.loads(config_str)

app = Flask(__name__)

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_DEPLOYMENT_ID = os.getenv('AZURE_OPENAI_DEPLOYMENT_ID')
AZURE_OPENAI_ENABLED = os.getenv('AZURE_OPENAI_ENABLED')
if AZURE_OPENAI_ENABLED == "0":
    AZURE_OPENAI_ENABLED = "False"
elif AZURE_OPENAI_ENABLED == "1":
    AZURE_OPENAI_ENABLED = "True"
azure_enabled = json.loads(AZURE_OPENAI_ENABLED.lower())
AZURE_OPENAI_URL = os.getenv('AZURE_OPENAI_URL')

print("azure_enabled")
print(azure_enabled)

if azure_enabled == True:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-05-01-preview",
        azure_endpoint=AZURE_OPENAI_URL,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_ID,
    )
else:
    client = OpenAI()

def ira_step1(prompt: str) -> str:
    print("ira_step1")
    print("prompt")
    print(prompt)

    while True:
        try:
            content = str("I want to check the internet for the following thing. What is the best google search input to get the best results? Give me the google search input, nothing else. If user has a complex research task for example compare things, you can suggest a few different prompts separated with semicolon ; character to cover different information sources. Here is the thing: " + prompt)
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
        except:
            print("some OpenAI problem")

def ira_step2(prompt: str, website_infos: []) -> str:
    print("ira_step2")

    while True:
        try:
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
        except:
            print("some OpenAI problem")

def ira_step2scrape(prompt: str) -> []:
    print("ira_step2scrape")
    links = []
    website_infos = []

    queries = prompt.split(";")
    for query in queries:
        query = query.replace(" ", "+")
        url = str("http://localhost:8080/api/v1/scraper-search-service?prompt=" + query)

        while True:
            try:
                print("request")
                print("url")
                print(url)
                html_result = requests.get(url)
                links_json = html_result.json()
                print(type(links_json))
                if (type(links_json) is dict) == True:
                    raise Exception('Wrong data!')
                if (type(links_json) is list) == False:
                    raise Exception('Wrong data!')
                if len(links_json) > 0:
                    for link_json in links_json:
                        if link_json not in links:
                            links.append(link_json)
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

def ira_step3(prompt: str, website_infos: []) -> []:
    print("ira_step3")
    result_website_infos = []
    for website_info in website_infos:
        try:
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
        except:
            print("some OpenAI problem")
    print("result_website_infos")
    print(result_website_infos)
    return result_website_infos

def ira_step4(prompt: str, strategy: str, website_infos: []) -> str:
    print("ira_step4")
    content = str("I give you a subject of interest, a strategy and several source of information. \n\nGive me an optimal answer to the subject of interest without relativizing by using the provided source of information with academic footnotes like [1] followed by the corresponding URL. Just give me the conclusion and make the answer very clear and simple without referring the strategy again. Don't explain yourself. \n\nSubject of interest:\n "+ prompt + "\n\nStrategy:\n")

    i = 1
    for website_info in website_infos:
        content = str(content + " \n\n\n\nURL" + str(i) + " \n" + website_info["link"] + " \n\nSummary of URL" + str(i) + " \n" + website_info["summary"])
        i += 1
    print("content")
    print(content)

    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Reply in the language the user question is asked. Make sure you provide links in your response. Try to create a very long report.",
                    },
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model=config["models"]["model"],
            )

            return chat_completion.choices[0].message.content
        except:
            print("some OpenAI problem")

def iru_step1(prompt: str) -> []:
    print("iru_step1")

    while True:
        try:
            content = str("User provides in prompt what he want to do with certain website and url. Return this data as formated json of array of objects with keys: what_to_do, url. Here is the prompt: " + prompt)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model=config["models"]["model"],
            )

            result = chat_completion.choices[0].message.content

            if "```json" in result:
                result = result.lstrip("```json")

            if "```" in result:
                result = result.rstrip("```")

            result = json.loads(result)

            return result
        except:
            print("some OpenAI problem")

def iru_step2scrape(websites: []) -> []:
    print("iru_step2scrape")
    links = []
    website_infos = []

    print("websites")
    print(websites)
    for website in websites:
        link = website["url"]
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
            "what_to_do": website["what_to_do"],
        }
        website_infos.append(website_info)

    return website_infos

def iru_step3(website_infos: []) -> []:
    print("iru_step3")
    result_website_infos = []
    for website_info in website_infos:
        try:
            content = str("I will give you a user command and website text and you will provide the answer. User command: " + website_info["what_to_do"] + " Website text: " + website_info["text"])
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model=config["models"]["model"],
            )

            result = chat_completion.choices[0].message.content

            result_website_info = {
                "link": website_info["link"],
                "text": website_info["text"],
                "result": result,
                "url": website_info["url"],
                "what_to_do": website_info["what_to_do"],
            }
            result_website_infos.append(result_website_info)
        except:
            print("some OpenAI problem")
    print("result_website_infos")
    print(result_website_infos)
    return result_website_infos

def iru_step4(website_infos: []) -> str:
    print("iru_step4")
    content = str("I give you a list of websites. Website link, website text, user command and result of user command from previous GPT call. Return a long report.")

    i = 1
    for website_info in website_infos:
        content = str(content + " \n\n\n\nURL" + str(i) + " \n" + website_info["link"] + " \n\nText of URL" + str(i) + " \n" + website_info["text"] + " \n\nUser command of URL" + str(i) + " \n" + website_info["what_to_do"] + " \n\nPrevious GPT call result of URL" + str(i) + " \n" + website_info["result"])
        i += 1
    print("content")
    print(content)

    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Reply in the language the user question is asked. Make sure you provide links in your response. Try to create a very long report.",
                    },
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model=config["models"]["model"],
            )

            return chat_completion.choices[0].message.content
        except:
            print("some OpenAI problem")

@app.route('/v1/internet_research_agent', methods=['POST'])
def internet_research_agent():
    print("internet_research_agent")
    data = request.json
    prompt = data.get("full_prompt", "")
    print("prompt")
    print(prompt)
    ira_step1result = ira_step1(prompt)
    print("ira_step1result")
    print(ira_step1result)
    website_infos = ira_step2scrape(ira_step1result)
    ira_step2result = ira_step2(prompt, website_infos)
    website_infos = ira_step3(prompt, website_infos)
    ira_step4result = ira_step4(prompt, ira_step2result, website_infos)

    response = {
        "response": ira_step4result,
    }

    return jsonify(response), 201

@app.route('/v1/internet_research_urls', methods=['POST'])
def internet_research_urls():
    print("internet_research_urls")
    data = request.json
    prompt = data.get("full_prompt_with_urls", "")
    iru_step1result = iru_step1(prompt)
    website_infos = iru_step2scrape(iru_step1result)
    website_infos = iru_step3(website_infos)
    iru_step4result = iru_step4(website_infos)

    response = {
        "response": iru_step4result,
    }

    return jsonify(response), 201

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }

    return jsonify(response), 201

app.run(host = "0.0.0.0", threaded=True)
