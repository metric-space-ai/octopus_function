import os

dependencies = [
    "pip install -q beautifulsoup4==4.13.3",
    "pip install -q blinker==1.9.0",
    "pip install -q flask==3.1.0",
    "pip install -q langchain==0.3.19",
    "pip install -q langchain-community==0.3.18",
    "pip install -q markdown==3.7",
    "pip install -q marker-pdf==31.5.5",
    "pip install -q openai==1.65.2",
    "pip install -q openpyxl==3.1.5",
    "pip install -q pdfkit==1.0.0",
    "pip install -q pyPDF2==3.0.1",
    "pip install -q requests==2.32.3",
    "pip install -q tiktoken==0.9.0",
    "pip install -q tqdm==4.67.1",
]

for command in dependencies:
    os.system(command)

import ast
import datetime
import io
import json
import logging
import openai
import re
import requests
import textwrap
import tiktoken
import time

from flask import Flask, request, jsonify
from tqdm import tqdm

# Load sensitive values from environment variables (DO NOT HARDCODE THEM)
API_KEY = os.getenv("OPENAI_API_KEY")
OCTOPUS_TOKEN = os.getenv("OCTOPUS_TOKEN")

# Set up OpenAI client
client = openai.OpenAI(api_key=API_KEY)

### Configuration section
config = """{
    "device_map": {
        "cpu": "10GiB"
    },
    "required_python_version": "cp311",
    "models": {
        "model": "gpt-4o-mini-2024-07-18"
    },
    "functions": [
        {
            "name": "textbook_generator",
            "description": "'Textbook generator' function is a tool used when existing data can't answer a complex query. It is triggered when a user specifically requests detailed knowledge or confirms they want to proceed with in-depth research. For example: User: 'Writes some question or task.' Assistant: 'The task/question that you provided seems complex. There is no fitting expert agent in the database. Do you want to generate a new expert agent?' User: 'Yes.' The function first asks for confirmation, as the process may take time. Once confirmed, it gathers information from multiple sources, compiles it into a structured resource, like a book, and saves it for future use. Examples of when to trigger this function include (1) compiling data on new technologies, (2) when no expert agent or knowledge exists for a technical inquiry, or (3) gathering detailed info about technical processes. It should not be triggered if (1) the information is already available, (2) the user hasn't confirmed, or (3) the query is simple.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The user's input message along with additional context from the chat history that covers the entire context of the question."
                    },
                    "language": {
                        "type": "string",
                        "description": "The language of the book to generate. normally the language of the user input when not specified explicitly"
                    },
                    "urls": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "format": "uri"
                        },
                        "description": "A list of URLs from the chat history that contain useful information for solving the task."
                    }
                },
                "required": ["task", "urls"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        }
    ]
}"""
config = json.loads(config)
app = Flask(__name__)

max_context_window_chars = 512000
company_name = "preview"
OCTOPUS_API_BASE = f"https://api.{company_name}.octopus-ai.app"
OCTOPUS_FILES_ENDPOINT = f"{OCTOPUS_API_BASE}/api/v1/files"
OCTOPUS_SEARCH_ENDPOINT = f"{OCTOPUS_API_BASE}/api/v1/scraper-search-service"
OCTOPUS_SCRAPE_ENDPOINT = f"{OCTOPUS_API_BASE}/api/v1/scraper-service"

headers = {"X-Auth-Token": OCTOPUS_TOKEN}


def gpt_inference(content: str, language, model="gpt-4o-mini-2024-07-18") -> str:
    print("function gpt_inference")

    if len(content) > max_context_window_chars:
        content = textwrap.shorten(content, width=max_context_window_chars)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an AI assistant that always responds in {language}.",
                },
                {"role": "user", "content": content},
            ],
            model=model,
        )
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

    if not chat_completion or not chat_completion.choices:
        print("Error: No valid response received from OpenAI API.")
        return None

    summary = chat_completion.choices[0].message.content.strip()
    if not summary:
        print("Error: Empty response from OpenAI API.")
        return None

    return summary


def load_markdowns():
    print("function load_markdowns")
    url = f"{OCTOPUS_FILES_ENDPOINT}"
    payload = ""
    response = requests.request("GET", url, headers=headers, data=payload)
    json_list = []
    for document in tqdm(response.json()):
        if document["type"] == "KnowledgeBook":
            resp = requests.get(document["url"])
            json_list.append(resp.json())

    return json_list


markdowns_list = load_markdowns()


def solve_task(task: str, language, textbook: str = None) -> str:
    print("function solve_task")
    # check if there alreaedy is an expert for that task
    print(task)
    if textbook is None:
        response = (
            "The task/question that you provided seems to be complex. There is no fitting \
expert agent in the database. Do you want to generate a new expert agent?"
        )
    else:
        content = f"""Answer the following task. Use the information from the provided textbook. Dont give the user excersices if he didnt ask for it. If you can't answer the question adequately, answer why you can't, but don't make up an answer. Don't explain yourself, concentrate on the answer.
Here is the Textbook:
{textbook}

Now answer the following Task:
{task}
"""
        response = gpt_inference(content, language)
    # ask to generate new expert
    return response


def fetch_search_results(query, max_retries=5):
    print("function fetch_search_results")
    url = f"{OCTOPUS_SEARCH_ENDPOINT}?prompt={query}"
    attempts = 0
    while attempts < max_retries:
        try:
            print("Requesting URL:")
            print(url)
            html_result = requests.get(url, headers=headers)
            html_result.raise_for_status()  # Raise an HTTPError for bad responses
            links_json = html_result.json()
            # Debugging statements
            print(f"Response type: {type(links_json)}")
            print(f"Response content: {links_json}")
            if isinstance(links_json, dict):
                raise ValueError("Received data is a dictionary, expected a list!")
            if not isinstance(links_json, list):
                raise ValueError("Received data is not a list!")
            if len(links_json) > 0:
                return links_json
            else:
                raise ValueError("Received empty list!")
        except (requests.RequestException, ValueError) as e:
            print(f"Error occurred: {e}")
            print("Sleeping for 5 seconds before retrying...")
            time.sleep(5)
            attempts += 1
    raise Exception("Max retries exceeded")


def fetch_page_content(url, max_retries=2):
    print("function fetch_page_content")
    attempts = 0
    while attempts < max_retries:
        try:
            response = requests.get(
                f"{OCTOPUS_SCRAPE_ENDPOINT}?url={url}", headers=headers
            )
            print(response)
            print(len(response.content))
            return response.content
        except requests.RequestException as e:
            print(f"Error occurred while scraping {url}: {e}")
            print("Sleeping for 5 seconds before retrying...")
            time.sleep(5)
            attempts += 1
    print(f"Max retries exceeded for {url}")
    return {"text": ""}


def clean_pdf_content(content):
    print("function clean_pdf_content")
    if type(content) == str:
        return content
    decoded_content = content.decode("utf-8", errors="ignore")
    cleaned_content = re.sub(r"\b\d+\.\d+\.\d+\b", "", decoded_content)
    return cleaned_content


def process_user_files(files_list, language):
    print("function process_user_files")
    """
    Processes user files by summarizing content and extracting topics.
    Ensures each file has:
      - 'summary': a brief overview of the file.
      - 'topics': key topics extracted from the summary.
    """

    processed = []
    for f in files_list:
        try:
            # If no summary exists, generate one
            if not f.get("summary"):
                prompt_for_summary = f"""
                Summarize the following text in **3-5 sentences**. 
                Then extract 3 main topics covered in the text. 
                Format the response as:
                Summary: [Your Summary Here]
                Topics: ["Topic 1", "Topic 2", "Topic 3"]
                ---
                Text:
                {f["content"]}
                """
                summary_response = gpt_inference(prompt_for_summary, language)
                f["summary"] = extract_summary(summary_response)
                f["topics"] = extract_topics(summary_response)

            processed.append(f)
        except Exception as e:
            print(f"Skipping file {f['filename']} because of error: {e}")
            import traceback

            print(traceback.format_exc())

    return processed


# Helper function to extract summary from GPT response
def extract_summary(response):
    print("function extract_summary")
    match = re.search(r"Summary:\s*(.+)", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


# Helper function to extract topics from GPT response
def extract_topics(response):
    print("function extract_topics")
    match = re.search(r"Topics:\s*(\[[^\]]+\])", response)
    if match:
        try:
            return eval(match.group(1))  # Convert string list to actual list
        except:
            return []
    return []


def extract_title_if_exists(task: str) -> str:
    print("function extract_title_if_exists")
    """
    If the user prompt includes something like 'Title: My Book Title', 'Titel' or something similar
    extract and return it. Otherwise, return an empty string.
    """
    import re

    title_pattern = r"(?:Title|Book Title)\s*:\s*(.+)"
    match = re.search(title_pattern, task, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def create_topic(all_topics, language, task):
    print("function create_topic")
    topics_prompt = f"""
    You are a title generator for textbooks. Your job is to create a **clear, structured, and engaging book title** 
    based on the provided topics:
    {", ".join(all_topics)}  # Converts the list to a string

    And the user request:
    {task}

    - The title should be **specific**, fitting to the provided topics.
    - Avoid generic or vague titles.
    - Output only **one** title as a Python string (not a list).

    Respond only with a single-line string, without explanations.
    """

    if len(topics_prompt) > max_context_window_chars:
        topics_prompt = textwrap.shorten(topics_prompt, width=max_context_window_chars)

    # Reset the global system prompt so that a new language is detected for this call
    system_prompt = f"You are an AI assistant that always responds in {language}."

    if len(system_prompt) > max_context_window_chars:
        system_prompt = textwrap.shorten(system_prompt, width=max_context_window_chars)

    completion = create_completion(
        [
            {
                "role": "system",
                "content": system_prompt,
            },  # Use the newly defined system prompt
            {"role": "user", "content": topics_prompt},
        ]
    )

    topic = (
        extract_response(completion)
        .replace("```python\n", "")
        .replace("```", "")
        .strip()
    )

    print(f"📘 Generated Book Title: {topic}")
    return topic


def remove_duplicates(input_list):
    print("function remove_duplicates")
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def select_best_file_for_subchapter(files_list, language, subchapter):
    print("function select_best_file_for_subchapter")
    """
    Selects the most relevant file from uploaded documents for a given subchapter.
    Uses summaries to determine the best match.
    """
    prompt = f"""
    Below are summaries of multiple uploaded documents. 
    Which single document best matches the following subchapter?

    Subchapter: {subchapter}

    Respond ONLY with the file index (1-based) or 'None' if no match.
    """

    for i, f in enumerate(files_list, start=1):
        prompt += f"\nDocument {i} Summary:\n{f['summary']}"

    answer = gpt_inference(prompt, language).strip()

    if answer == "None":
        return None  # No relevant file found
    try:
        index = int(answer) - 1
        return files_list[index]["content"]  # Return the best-matching file content
    except:
        return None


def extract_response(comp):
    print("function extract_response")
    if comp is None:
        print("Error: Received None instead of a valid completion response.")
        return ""
    try:
        return comp.choices[0].message.content
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        print(f"Error extracting response: {e}")
        return ""


def create_completion(messages, model="gpt-4o-mini-2024-07-18"):
    print("function create_completion")
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        # Debug: print the complete response using proper formatting
        print("Full API response: %s" % completion)

        # Updated check for choices attribute
        if (
            not completion
            or not getattr(completion, "choices", None)
            or not completion.choices
        ):
            print("Error: Invalid API response (missing choices).")
            return None
        return completion
    except Exception as e:
        print(f"Error in create_completion: {e}")
        import traceback

        print(traceback.format_exc())
        return None


def enhance_with_internet_data(query):
    print("function enhance_with_internet_data")
    search_results = fetch_search_results(query)
    enhanced_contents = []
    used_links = []
    search_results = search_results[:4]
    for link in search_results:
        print(f"Scraping document from {link}")
        enhanced_content = fetch_page_content(link)
        print("Fetched")
        enhanced_content = clean_pdf_content(enhanced_content)
        enhanced_contents.append(enhanced_content)
        used_links.append(link)
    return enhanced_contents, used_links


def parse_outline_into_subchapters(outline_json):
    print("function parse_outline_into_subchapters")
    """
    Parses the JSON outline into a list of subchapter labels, e.g. ["1.1", "1.2", ...].
    Adjust as needed depending on your JSON structure.
    """
    chapters = []
    try:
        outline_data = json.loads(outline_json)
        # For instance, if outline_data is a list of chapters, each with 'subchapters'

        for a_key in outline_data:
            for a_value in outline_data[a_key]:
                if a_value == "title":
                    chapters.append((a_key, outline_data[a_key]["title"]))
                if a_value == "subchapters":
                    for b_key in outline_data[a_key]["subchapters"]:
                        for b_value in outline_data[a_key]["subchapters"][b_key]:
                            if b_value == "title":
                                chapters.append(
                                    (
                                        b_key,
                                        outline_data[a_key]["subchapters"][b_key][
                                            "title"
                                        ],
                                    )
                                )
                            if b_value == "subchapters":
                                for c_key in outline_data[a_key]["subchapters"][b_key][
                                    "subchapters"
                                ]:
                                    for c_value in outline_data[a_key]["subchapters"][
                                        b_key
                                    ]["subchapters"][c_key]:
                                        if c_value == "title":
                                            chapters.append(
                                                (
                                                    c_key,
                                                    outline_data[a_key]["subchapters"][
                                                        b_key
                                                    ]["subchapters"][c_key]["title"],
                                                )
                                            )

    except:
        pass
    return chapters


def fetch_desired_files(urls):
    print("function fetch_desired_files")
    """
    Fetches only the specified files from the Octopus server.
    """

    fetched_files = []

    try:
        for url in urls:
            print(f"📥 Downloading {url}...")

            fname = ""
            file_resp = requests.get(url)
            file_resp.raise_for_status()
            file_content = file_resp.text  # or .content if binary
            if "Content-Disposition" in file_resp.headers.keys():
                fname = re.findall(
                    "filename=(.+)", file_resp.headers["Content-Disposition"]
                )[0]
            else:
                fname = url.split("/")[-1]

            fetched_files.append({"filename": fname, "content": file_content})

        return fetched_files

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching files: {e}")
        return []


def generate_textbook(task, language, urls):
    print("function generate_textbook")
    """
    Generates a textbook based on files fetched from the Octopus server.
    If fewer than 2 files are available, fetches additional content from the internet.
    """

    print(f"urls = {urls}")

    if not urls:
        raise Exception("Failed to get a urls.")

    print(f"📜 Using task prompt: {task}")

    # Fetch only the specific files from Octopus
    user_files = fetch_desired_files(urls)

    print(f"📂 Fetched {len(user_files)} specific files from the server.")

    # Process the retrieved files
    processed_user_files = process_user_files(user_files, language)

    # -- STEP 2: If fewer than 2 files, fallback to internet data --
    if len(processed_user_files) < 2:
        print("⚠️ Not enough files from Octopus, fetching internet data...")

        # Fetch extra content from the internet
        enhanced_contents, used_links = enhance_with_internet_data(task)

        # Convert scraped content into "files"
        scraped_files = []
        for i, content in enumerate(enhanced_contents):
            scraped_files.append(
                {"filename": f"scraped_doc_{i}.txt", "content": content}
            )

        # Process scraped files
        processed_scraped_files = process_user_files(scraped_files, language)

        # Merge sets
        processed_files = processed_user_files + processed_scraped_files
    else:
        processed_files = processed_user_files

    # ✅ Now `processed_files` includes only the selected files (plus optional web-scraped ones)

    # -- STEP 3: Build topics, etc. --
    all_topics = []
    for f in processed_files:
        if "topics" in f:
            all_topics.extend(f["topics"])
    all_topics = list(set(all_topics))

    # Generate the book title from all topics
    user_provided_title = extract_title_if_exists(task)
    if user_provided_title:
        topic = user_provided_title  # If the user gave a title, use it
    else:
        topic = create_topic(all_topics, language, task)  # Generate a title from topics

    print(f"📖 FINAL BOOK TITLE: {topic}")

    outline_prompt = f"""
    I have a list of <topics> from documents, a <book idea> and a <title>. I want you to write an outline of a book with the <title>: "{topic}", based on the <book idea>. I want you to write the outline fitting to the <book idea> and choose fitting specific topics from the <topics> list. The outline should have 8 chapters that progresses logically, with later items building on earlier ones. Organize the chapters and subchapters in 1.1, 1.2 and 1.1.1 and 1.1.2 and so on.

    <topics>:
    {all_topics}


    <book idea>:
    {task}

    Now use these <topics< to generate the 8 chapter outline based off the <book idea>, no explanation, nothing else.
    Return only in JSON format. """

    if len(outline_prompt) > max_context_window_chars:
        outline_prompt = textwrap.shorten(
            outline_prompt, width=max_context_window_chars
        )

    # Instead of calling create_completion directly, use gpt_inference which already adds the system prompt
    outline_response = gpt_inference(outline_prompt, language)

    if outline_response is None or outline_response.strip() == "":
        raise Exception("Failed to get a completion for outline.")

    correction_prompt_14 = f"""
    This guide is not detailed enough, you are scratching only the surface of the topics. 
    Take only the chapters „1-4“ of the following outline and add more subchapters to detail the steps further (1.1.1, 1.1.2, etc.). 
    Make sure the book titled "{topic}" uses these topics: {all_topics}.

    Outline of book: 
    {outline_response}

    Return the detailed outline of chapters 1-4 in JSON format, without any other explanation.
    """
    correction_prompt_58 = f"""
    This guide is not detailed enough, you are scratching only the surface of the topics. 
    Take only the chapters „5-8“ of the following outline and add more subchapters to detail the steps further (1.1.1, 1.1.2, etc.). 
    Make sure the book titled "{topic}" uses these topics: {all_topics}.

    Outline of book: 
    {outline_response}

    Return the detailed outline of chapters 5-8 in JSON format, without any other explanation.
    """

    if len(correction_prompt_14) > max_context_window_chars:
        correction_prompt_14 = textwrap.shorten(
            correction_prompt_14, width=max_context_window_chars
        )

    outline_14 = gpt_inference(correction_prompt_14, language)
    if outline_14 is None or outline_14.strip() == "":
        raise Exception("Failed to get a completion for correction 1-4.")
    print(f"outline_14 = {outline_14}")

    if len(correction_prompt_58) > max_context_window_chars:
        correction_prompt_58 = textwrap.shorten(
            correction_prompt_58, width=max_context_window_chars
        )

    outline_58 = gpt_inference(correction_prompt_58, language)
    if outline_58 is None or outline_58.strip() == "":
        raise Exception("Failed to get a completion for correction 5-8.")
    print(f"outline_58 = {outline_58}")

    list_of_chapters = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    chapters_storage = []

    for chap in list_of_chapters:
        print(f"\n\n\n\n\n{chap} NEW CHAPTER\n\n\n\n")
        num = str(
            {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
            }.get(chap, "")
        )

        # CHAPTERS 1-4
        if chap in ["one", "two", "three", "four"]:
            breakdown_prompt = f"""
            Now I want to break it down and go through the outline step by step, looking at each chapter individually. I want to start with chapter {chap}, keeping the names of the subchapters and the breakdown of {num}.1.1 and {num}.1.2 and
            so on, breaking it down even further with another level of subchapter, if it helps with creating more depth and detail. Use a tone that communicates complex ideas clearly and engagingly. Give me, only to the subchapters that will later turn into text, a summary with keywords that describe the part in detail. 
            Don't write full sentences yet, try to use as much information as possible. Don't explain yourself.
            """
            breakdown_prompt += """You response a json with structure like this ```json\n{\n  "8": {\n    "title": "Achieving Lasting Change and Inner Freedom",\n    "subchapters": {\n      "8.1": {\n        "title": "Inner Alignment and Joy Creation",\n        "subchapters": {\n          "8.1.1": {\n            "title": "Understanding Inner Alignment",\n            "details": {\n              "keywords": [\n                "Alignment",\n                "Authenticity",\n                "Inner truth",\n                "Congruence",\n                "Personal values",\n                "Self-awareness",\n                "Mind-body connection"\n              ]\n            }\n          },\n          "8.1.2": {\n            "title": "Practices for Fostering Inner Joy",\n            "details": {\n              "keywords": [\n                "Joyfulness",\n                "Daily rituals",\n                "Positive habits",\n                "Gratitude practices",\n                "Mindfulness techniques",\n                "Creative expression",\n                "Social connections"\n              ]\n            }\n          }\n        }\n      },\n      "8.2": {\n        "title": "Practical Approaches to Well-Being",\n        "subchapters": {\n          "8.2.1": {\n            "title": "Integrating Mindfulness into Everyday Activities",\n            "details": {\n              "keywords": [\n                "Mindfulness",\n                "Present moment awareness",\n                "Everyday practice",\n                "Focus and clarity",\n                "Routine mindfulness exercises",\n                "Reduced stress",\n                "Increased engagement"\n              ]\n            }\n          },\n          "8.2.2": {\n            "title": "Holistic Nutrition for Mind and Body Health",\n            "details": {\n              "keywords": [\n                "Nutrition",\n                "Holistic health",\n                "Mind-body connection",\n                "Balanced diet",\n                "Nutrient-rich foods",\n                "Emotional well-being",\n                "Food as medicine"\n              ]\n            }\n          }\n        }\n      }\n    }\n  }\n}\n```. Make sure your resonse is a valid JSON. Regard only the given questions or instructions in the prompt and always return only a json."""

            # Extract only the relevant chapter from outline_14
            try:
                parsed_outline = json.loads(outline_14)
                relevant_outline = parsed_outline.get(num, None)
                if relevant_outline:
                    relevant_outline_str = json.dumps(relevant_outline)
                else:
                    relevant_outline_str = outline_14  # Fallback if key not found
            except Exception as e:
                print(f"Error parsing outline_14: {e}")
                relevant_outline_str = outline_14

            if len(relevant_outline_str) > max_context_window_chars:
                relevant_outline_str = textwrap.shorten(
                    relevant_outline_str, width=max_context_window_chars
                )
            if len(breakdown_prompt) > max_context_window_chars:
                breakdown_prompt = textwrap.shorten(
                    breakdown_prompt, width=max_context_window_chars
                )

            combined_prompt = f"{relevant_outline_str}\n{breakdown_prompt}"
            breakdown_response = gpt_inference(combined_prompt, language)

            if breakdown_response is None or breakdown_response.strip() == "":
                raise Exception(f"Failed to get a completion for breakdown {chap}.")

            chapter_breakdown = breakdown_response
            chapter_breakdown = (
                chapter_breakdown.replace("```json", "").replace("```", "").strip()
            )
            print(f"chapter_breakdown = {chapter_breakdown}")

            # Parse the subchapters from JSON
            notations = parse_outline_into_subchapters(chapter_breakdown)
            print(f"notations = {notations}")

            for notation in notations:
                print(f"Writing chapter {notation}")

                # SELECT BEST FILE
                best_file_content = select_best_file_for_subchapter(
                    processed_files, language, notation[1]
                )
                if best_file_content is None:
                    # fallback to scraping
                    enhanced_contents, used_links = enhance_with_internet_data(
                        notation[1]
                    )
                    scraped_files = []
                    for i, txt in enumerate(enhanced_contents):
                        scraped_files.append(
                            {"filename": f"scraped_{i}.txt", "content": txt}
                        )
                    processed_scraped = process_user_files(scraped_files, language)
                    best_file_content = select_best_file_for_subchapter(
                        processed_scraped, language, notation[1]
                    )

                if best_file_content is None:
                    best_file_content = "Not enough data found for this subchapter."

                chapter_write_prompt = f"""{best_file_content}
                You are a professor in the field of "{topic}", writing {notation[1]} for the textbook. Your chapters are long and thoroughly developed, packed with dense information while remaining clear and easy to understand. Each section explores concepts in depth, providing extensive explanations and detailed discussions that give professionals in the field a comprehensive understanding.
                Use the information from the file to write about the topic in detail, using exercises, specific examples, concrete numbers, case studies and real-life applications if applicable. Avoid generalizations and overviews. 
                Some other features to include: engaging narrative, research-based insights and actionable tips.
                Only return the specific textpart, nothing else."""

                if len(chapter_breakdown) > max_context_window_chars:
                    chapter_breakdown = textwrap.shorten(
                        chapter_breakdown, width=max_context_window_chars
                    )

                if len(chapter_write_prompt) > max_context_window_chars:
                    chapter_write_prompt = textwrap.shorten(
                        chapter_write_prompt, width=max_context_window_chars
                    )

                # Ensure we get the system prompt for the session
                system_prompt = (
                    f"You are an AI assistant that always responds in {language}."
                )

                if len(system_prompt) > max_context_window_chars:
                    system_prompt = textwrap.shorten(
                        system_prompt, width=max_context_window_chars
                    )

#                print("System prompt: %s", system_prompt)
#                print("Chapter breakdown: %s", chapter_breakdown)
#                print("Chapter write prompt: %s", chapter_write_prompt)

                completion = create_completion(
                    [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },  # Add system prompt first
                        {"role": "assistant", "content": chapter_breakdown},
                        {"role": "user", "content": chapter_write_prompt},
                    ]
                )

                if completion is None:
                    raise Exception(
                        f"Failed to get a completion for chapter write {notation[0]}."
                    )
                written_chapter = extract_response(completion)

                revision_prompt = f"""
                You’re a teacher revising a draft section of a textbook titled: ‘{topic}’. The section in focus is: {notation}.

                The original text is:

                {written_chapter}

                Your goal is to refine and enhance this section by making it:
                • Highly detailed and information-rich – Expand on key concepts with thorough explanations, incorporating relevant examples, case studies, illustrative scenarios, and well-supported arguments. Avoid unnecessary generalizations and ensure every point is deeply explored.
                • Engaging – Maintain a clear and professional tone while making the content compelling, well-structured, and intellectually stimulating.
                • Accessible & well-organized – Improve readability with smooth transitions and logical sequencing. Restructure where needed for clarity and flow while keeping details exhaustive and precise.
                • Interactive – Where appropriate, integrate an exercise, thought-provoking question, or practical application to encourage deeper understanding and engagement.

                Your revision should bring clarity, depth, and coherence to the text while ensuring it remains immersive, extensively detailed, and rigorously developed. Return only the revised section."""

                if len(revision_prompt) > max_context_window_chars:
                    revision_prompt = textwrap.shorten(
                        revision_prompt, width=max_context_window_chars
                    )

                completion = create_completion(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": revision_prompt},
                    ]
                )
                if completion is None:
                    raise Exception(
                        f"Failed to get a completion for revision {notation[0]}."
                    )
                revised_chapter = extract_response(completion)

                formatting_prompt = f"""Keep the text the same but remove any table of contents, page numbers, or references, leaving only the book's content. Format it in GitHub-flavored Markdown with these rules:
                Use `#` only for chapter titles, matching the number of `#` to the depth of the chapter number. Remove the number itself.
                Examples:
                - "1 Chapter Title" → `# Chapter Title`
                - "1.1 Section Title" → `## Section Title`
                - "1.2.1 Subsection Title" → `### Subsection Title`
                - "1.2.1.1.1 Deep Section Title" → `##### Deep Section Title`

                Do **not** use `#` for non-title text.

                Format math correctly:
                  - Inline: `$...$`
                  - Block: `$$...$$`

                - Use triple backticks (```) for code blocks.
                - Remove all remaining artifacts like numbering in running text.
                Return only the revised textbook in GitHub-flavored Markdown, without additional explanations, metadata, or script text."""

                if len(revision_prompt) > max_context_window_chars:
                    revision_prompt = textwrap.shorten(
                        revision_prompt, width=max_context_window_chars
                    )

                if len(revised_chapter) > max_context_window_chars:
                    revised_chapter = textwrap.shorten(
                        revised_chapter, width=max_context_window_chars
                    )

                if len(formatting_prompt) > max_context_window_chars:
                    formatting_prompt = textwrap.shorten(
                        formatting_prompt, width=max_context_window_chars
                    )

                completion = create_completion(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": revision_prompt},
                        {"role": "assistant", "content": revised_chapter},
                        {"role": "user", "content": formatting_prompt},
                    ],
                    model="gpt-4o-mini-2024-07-18",
                )

                if completion is None:
                    raise Exception(
                        f"Failed to get a completion for formatting {notation[0]}."
                    )
                final_text_part = extract_response(completion)
                chapters_storage.append(final_text_part)

        # CHAPTERS 5-8
        else:
            breakdown_prompt = f"""
            Now I want to break it down and go through the outline step by step, looking at each chapter individually. I want to start with chapter {chap}, keeping the names of the subchapters and the breakdown of {num}.1.1 and {num}.1.2 and
            so on, breaking it down even further with another level of subchapter, if it helps with creating more depth and detail. Use a tone that communicates complex ideas clearly and engagingly. Give me, only to the subchapters that will later turn into text, a summary with keywords that describe the part in detail. 
            Don't write full sentences yet, try to use as much information as possible. Don't explain yourself."""
            breakdown_prompt += """You response a json with structure like this ```json\n{\n  "8": {\n    "title": "Achieving Lasting Change and Inner Freedom",\n    "subchapters": {\n      "8.1": {\n        "title": "Inner Alignment and Joy Creation",\n        "subchapters": {\n          "8.1.1": {\n            "title": "Understanding Inner Alignment",\n            "details": {\n              "keywords": [\n                "Alignment",\n                "Authenticity",\n                "Inner truth",\n                "Congruence",\n                "Personal values",\n                "Self-awareness",\n                "Mind-body connection"\n              ]\n            }\n          },\n          "8.1.2": {\n            "title": "Practices for Fostering Inner Joy",\n            "details": {\n              "keywords": [\n                "Joyfulness",\n                "Daily rituals",\n                "Positive habits",\n                "Gratitude practices",\n                "Mindfulness techniques",\n                "Creative expression",\n                "Social connections"\n              ]\n            }\n          }\n        }\n      },\n      "8.2": {\n        "title": "Practical Approaches to Well-Being",\n        "subchapters": {\n          "8.2.1": {\n            "title": "Integrating Mindfulness into Everyday Activities",\n            "details": {\n              "keywords": [\n                "Mindfulness",\n                "Present moment awareness",\n                "Everyday practice",\n                "Focus and clarity",\n                "Routine mindfulness exercises",\n                "Reduced stress",\n                "Increased engagement"\n              ]\n            }\n          },\n          "8.2.2": {\n            "title": "Holistic Nutrition for Mind and Body Health",\n            "details": {\n              "keywords": [\n                "Nutrition",\n                "Holistic health",\n                "Mind-body connection",\n                "Balanced diet",\n                "Nutrient-rich foods",\n                "Emotional well-being",\n                "Food as medicine"\n              ]\n            }\n          }\n        }\n      }\n    }\n  }\n}\n```. Make sure your resonse is a valid JSON. Regard only the given questions or instructions in the prompt and always return only a json."""

            try:
                parsed_outline = json.loads(
                    outline_58
                )  # or outline_58_raw if that's your full text
                relevant_outline = parsed_outline.get(num, None)
                if relevant_outline:
                    relevant_outline_str = json.dumps(relevant_outline)
                else:
                    relevant_outline_str = outline_58  # fallback if key not found
            except Exception as e:
                print(f"Error parsing outline_58: {e}")
                relevant_outline_str = outline_58

            if len(relevant_outline_str) > max_context_window_chars:
                relevant_outline_str = textwrap.shorten(
                    relevant_outline_str, width=max_context_window_chars
                )
            if len(breakdown_prompt) > max_context_window_chars:
                breakdown_prompt = textwrap.shorten(
                    breakdown_prompt, width=max_context_window_chars
                )

            completion = create_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": relevant_outline_str},
                    {"role": "user", "content": breakdown_prompt},
                ]
            )
            if completion is None:
                raise Exception(f"Failed to get a completion for breakdown {chap}.")

            chapter_breakdown = extract_response(completion)
            chapter_breakdown = (
                chapter_breakdown.replace("```json", "").replace("```", "").strip()
            )
            print(f"chapter_breakdown = {chapter_breakdown}")

            notations = parse_outline_into_subchapters(chapter_breakdown)
            print(f"notations = {notations}")

            for notation in notations:
                print(f"Writing chapter {notation}")

                # SELECT BEST FILE
                best_file_content = select_best_file_for_subchapter(
                    processed_files, language, notation[1]
                )

                if best_file_content is None:
                    # fallback to scraping
                    enhanced_contents, used_links = enhance_with_internet_data(
                        notation[1]
                    )
                    scraped_files = []
                    for i, txt in enumerate(enhanced_contents):
                        scraped_files.append(
                            {"filename": f"scraped_{i}.txt", "content": txt}
                        )
                    processed_scraped = process_user_files(scraped_files, language)
                    best_file_content = select_best_file_for_subchapter(
                        processed_scraped, language, notation[1]
                    )

                if best_file_content is None:
                    best_file_content = "Not enough data found for this subchapter."

                chapter_write_prompt = f"""{best_file_content}
                You are a professor in the field of "{topic}", writing {notation[1]} for the textbook. Your chapters are long and thoroughly developed, packed with dense information while remaining clear and easy to understand. Each section explores concepts in depth, providing extensive explanations and detailed discussions that give professionals in the field a comprehensive understanding.
                Use the information from the file to write about the topic in detail, using exercises, specific examples, concrete numbers, case studies and real-life applications if applicable. Avoid generalizations and overviews. 
                Some other features to include: engaging narrative, research-based insights and actionable tips.
                Only return the specific Textpart, nothing else."""

                if len(chapter_breakdown) > max_context_window_chars:
                    chapter_breakdown = textwrap.shorten(
                        chapter_breakdown, width=max_context_window_chars
                    )

                if len(chapter_write_prompt) > max_context_window_chars:
                    chapter_write_prompt = textwrap.shorten(
                        chapter_write_prompt, width=max_context_window_chars
                    )

                completion = create_completion(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "assistant", "content": chapter_breakdown},
                        {"role": "user", "content": chapter_write_prompt},
                    ]
                )
                if completion is None:
                    raise Exception(
                        f"Failed to get a completion for chapter write {notation[0]}."
                    )
                written_chapter = extract_response(completion)

                revision_prompt = f"""
                You’re a teacher revising a draft section of a textbook titled: ‘{topic}’. The section in focus is: {notation}.

                The original text is:

                {written_chapter}

                Your goal is to refine and enhance this section by making it:
                • Highly detailed and information-rich – Expand on key concepts with thorough explanations, incorporating relevant examples, case studies, illustrative scenarios, and well-supported arguments. Avoid unnecessary generalizations and ensure every point is deeply explored.
                • Engaging – Maintain a clear and professional tone while making the content compelling, well-structured, and intellectually stimulating.
                • Accessible & well-organized – Improve readability with smooth transitions and logical sequencing. Restructure where needed for clarity and flow while keeping details exhaustive and precise.
                • Interactive – Where appropriate, integrate an exercise, thought-provoking question, or practical application to encourage deeper understanding and engagement.

                Your revision should bring clarity, depth, and coherence to the text while ensuring it remains immersive, extensively detailed, and rigorously developed. Return only the revised section."""

                if len(revision_prompt) > max_context_window_chars:
                    revision_prompt = textwrap.shorten(
                        revision_prompt, width=max_context_window_chars
                    )

                completion = create_completion(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": revision_prompt},
                    ]
                )
                if completion is None:
                    raise Exception(
                        f"Failed to get a completion for revision {notation[0]}."
                    )
                revised_chapter = extract_response(completion)

                formatting_prompt = f"""Keep the text the same but remove any table of contents, page numbers, or references, leaving only the book's content. Format it in GitHub-flavored Markdown with these rules:
                Use `#` only for chapter titles, matching the number of `#` to the depth of the chapter number. Remove the number itself.
                Examples:
                - "1 Chapter Title" → `# Chapter Title`
                - "1.1 Section Title" → `## Section Title`
                - "1.2.1 Subsection Title" → `### Subsection Title`
                - "1.2.1.1.1 Deep Section Title" → `##### Deep Section Title`

                Do **not** use `#` for non-title text.

                Format math correctly:
                  - Inline: `$...$`
                  - Block: `$$...$$`

                - Use triple backticks (```) for code blocks.
                - Remove all remaining artifacts like numbering in running text.
                Return only the revised textbook in GitHub-flavored Markdown, without additional explanations, metadata, or script text."""

                if len(revision_prompt) > max_context_window_chars:
                    revision_prompt = textwrap.shorten(
                        revision_prompt, width=max_context_window_chars
                    )

                if len(revised_chapter) > max_context_window_chars:
                    revised_chapter = textwrap.shorten(
                        revised_chapter, width=max_context_window_chars
                    )

                if len(formatting_prompt) > max_context_window_chars:
                    formatting_prompt = textwrap.shorten(
                        formatting_prompt, width=max_context_window_chars
                    )

                completion = create_completion(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": revision_prompt},
                        {"role": "assistant", "content": revised_chapter},
                        {"role": "user", "content": formatting_prompt},
                    ],
                    model="gpt-4o-mini-2024-07-18",
                )
                if completion is None:
                    raise Exception(
                        f"Failed to get a completion for formatting {notation[0]}."
                    )
                final_text_part = extract_response(completion)
                chapters_storage.append(final_text_part)

    outline = str(outline_14) + "\n" + str(outline_58)
    model = "gpt-4o-mini-2024-07-18"
    separator = "\n\n"
    book = separator.join(chapters_storage)
    return book, topic, outline, model, chapters_storage


def save_book(json_to_save):
    print("function save_book")
    url = f"{OCTOPUS_FILES_ENDPOINT}"
    print(f"save_book url = {url}")
    payload = {
        "type": "KnowledgeBook",
        "access_type": "Company",
    }
    print(f"save_book payload = {payload}")
    file_name = json_to_save["topic"] + ".json"
    print(f"save_book file_name = {file_name}")
    # Convert the dictionary to a JSON string and then to bytes
    json_bytes = json.dumps(json_to_save).encode("utf-8")
    # Use io.BytesIO to create a file-like object in memory
    json_file = io.BytesIO(json_bytes)
    files = [("file", (file_name, json_file, "application/octet-stream"))]
    response = requests.post(url, headers=headers, data=payload, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Book Text: {response.text}")
    markdowns_list.append(json_to_save)


def Make_Textbook(task, language, urls):
    print("function Make_Textbook")
    book, topic, outline, model, chapters_list = generate_textbook(task, language, urls)
    # create summary
    content = f"Generate a summary of what the following book is about. Here is the book:\n{book}"
    summary = gpt_inference(content, language)
    # save file
    json_to_save = {
        "book": book,
        "topic": topic,
        "outline": outline,
        "model": model,
        "chapters": chapters_list,
        "summary": summary,
    }
    save_book(json_to_save)
    return book


@app.route("/v1/textbook_generator", methods=["POST"])
def textbook_generator():
    data = request.json
    task = data["task"]
    language = data["language"]
    urls = data.get("urls", [])

    textbook_text = Make_Textbook(task, language, urls)
    response = solve_task(task, language, textbook_text)
    response = {
        "response": response,
    }
    return jsonify(response), 200


@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {"setup": "Performed"}
    return jsonify(response), 201


if "__name__" == "__main__":
    app.run(host="0.0.0.0", threaded=True)
