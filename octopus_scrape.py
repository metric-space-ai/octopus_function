import os

dependencies = [
    'pip install -q fake-useragent==1.5.1',
    'pip install -q Flask==3.0.3',
    'pip install -q openpyxl==3.1.2',
    'pip install -q PyMuPDF==1.24.3',
    'pip install -q requests==2.31.0',
    'pip install -q selenium==4.20.0',
    'pip install -q selenium-wire==5.1.0',
]

for command in dependencies:
    os.system(command)

from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC

from fake_useragent import UserAgent

import time

import fitz as f

import requests
import threading

from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from flask import Flask, jsonify, request

config_str = '''{
    "device_map": {
        "cpu": "15GiB"
    },
    "required_python_version": "cp311",
    "functions": [
        {
            "name": "google_search",
            "description": "Google search function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_prompt": { "type": "string", "description": "Search prompt" }
                },
                "required": ["search_prompt"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        },
        {
            "name": "scrape_url",
            "description": "Url scrape function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "Url to scrape" }
                },
                "required": ["url"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        }
    ]}'''

def get_google_search_results(search_prompt, weight=10):
    """
    Get the URLs of the first n Google search results for a given query using Selenium.

    Parameters:
    - query (str): The search query.
    - num_results (int): The number of results to retrieve. Default is 5.

    Returns:
    - list: A list containing the URLs of the first n search results.
    """
    # Set up the Chrome WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    ua = UserAgent()
    user_agent = ua.random
    chrome_options.add_argument(f"user-agent={user_agent}")

    options = {
        'proxy': {'http': 'http://brd-customer-hl_0ca997a7-zone-serp:eeprm8oidobr@brd.superproxy.io:22225',
        'https': 'http://brd-customer-hl_0ca997a7-zone-serp:eeprm8oidobr@brd.superproxy.io:22225'},
        }

    driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=options)

    # Navigate to Google search
    driver.get("https://www.google.com/search?q=" + search_prompt + " -site:statista.com") 
    time.sleep(0.5)
    possible_texts = ["Alle ablehnen", "Alle akzeptieren"]
    for text in possible_texts:
        try:
            button_xpath = f"//button[contains(., '{text}')]"
            button = driver.find_element(By.XPATH, button_xpath)
            button.click()

        except:
            pass

    # Extract URLs from the search results
    urls = []

    search_section = driver.find_element(By.ID, 'search')
    names = search_section.find_elements(By.CSS_SELECTOR, "h3")
    for name in names:
        url = name.find_element(By.XPATH, '..').get_attribute("href")
        if not url == None:
            urls.append(url)
            
    driver.quit()
    
    
    return urls[:weight]

def scrape_url(url): 
    
    text = ""

    # download pdf and safe the text
    if ".pdf" in url:
        try:
            pdf_response = requests.get(url)
            with f.open("pdf", pdf_response.content) as doc:
                text = chr(12).join([page.get_text() for page in doc])
        except Exception as e:
            print(e)
            text = "Error loading the pdf"
    else: 
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument(f"--proxy brd.superproxy.io:22225 --proxy-user brd-customer-hl_0ca997a7-zone-isp:4at60zxmefcg")
        chrome_options.add_argument("--disable-dev-shm-usage")
        ua = UserAgent()
        user_agent = ua.random
        chrome_options.add_argument(f"user-agent={user_agent}")
        
        options = {
            'proxy': {'http': 'http://brd-customer-hl_0ca997a7-zone-isp:4at60zxmefcg@brd.superproxy.io:22225',
            'https': 'http://brd-customer-hl_0ca997a7-zone-isp:4at60zxmefcg@brd.superproxy.io:22225'},
            }

        driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=options)

        try:
            driver.get(url)

            # check for popup 
            possible_texts = ["Accept", "decline", "Zustimmen", "Akzeptieren", "Alle Akzeptieren", "OK", "AGREE", "Allow all", "Accept All Cookies", "Deny"]
            time.sleep(1)
            for button_text in possible_texts:
                try:
                    button = WebDriverWait(driver, 0.1).until(
                        EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, button_text))
                        )
                    button.click()

                except:
                    pass

            time.sleep(0.5)

            text = driver.find_element(By.XPATH, "/html/body").text
            driver.quit()
        except:
            try:
                driver.quit()
                driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=options)

                try:
                    driver.get(url, )

                    # check for popup 
                    possible_texts = ["Accept", "decline", "Zustimmen", "Akzeptieren", "Alle Akzeptieren", "OK", "AGREE", "Allow all", "Accept All Cookies", "Deny"]
                    time.sleep(1)
                    for button_text in possible_texts:
                        try:
                            button = WebDriverWait(driver, 0.1).until(
                                EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, button_text))
                                )
                            button.click()

                        except:
                            pass

                    time.sleep(0.5)

                    text = driver.find_element(By.XPATH, "/html/body").text
                    driver.quit()
                except:
                    try:
                        driver.quit()
                    except:
                        pass
            except:
                driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=options)

                try:
                    driver.get(url)

                    # check for popup 
                    possible_texts = ["Accept", "decline", "Zustimmen", "Akzeptieren", "Alle Akzeptieren", "OK", "AGREE", "Allow all", "Accept All Cookies", "Deny", "Consent"]
                    time.sleep(1)
                    for button_text in possible_texts:
                        try:
                            button = WebDriverWait(driver, 0.1).until(
                                EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, button_text))
                                )
                            button.click()

                        except:
                            pass

                    time.sleep(0.5)

                    text = driver.find_element(By.XPATH, "/html/body").text
                    driver.quit()
                except:
                    pass
                        
        try:
            driver.quit()
        except:
            pass
    return ILLEGAL_CHARACTERS_RE.sub(r"",text)

def scrape_url_with_timeout(url):
    result = ""
    # Set the timeout duration
    timeout_duration = 60  # 60 seconds for demonstration
    
    # Flag to indicate if the function has finished executing
    finished = threading.Event()
    
    # Variable to store the result of scrape_url
    result = None

    def target():
        nonlocal result
        # Call the scrape_url function and store its result
        result = scrape_url(url)
        finished.set()

    # Create and start a thread to execute the target function
    thread = threading.Thread(target=target)
    thread.start()

    # Wait for the thread to finish or until the timeout duration expires
    finished.wait(timeout_duration)

    # If the thread is still alive after the timeout duration, it means the function timed out
    if thread.is_alive():
        raise TimeoutError("Function execution timed out")

    # If the thread has finished, join it to the main thread
    thread.join()

    # Return the result of scrape_url
    return result

app = Flask(__name__)

@app.route('/v1/google_search', methods=['POST'])
def function_google_search():
    data = request.json
    search_prompt = data.get("search_prompt", "")

    result = get_google_search_results(search_prompt)

    response_text = str(result)

    response = {
        "response": response_text,
    }

    return jsonify(response), 201

@app.route('/v1/scrape_url', methods=['POST'])
def function_scrape_url():
    data = request.json
    url = data.get("url", "")

    result = scrape_url(url)

    response_text = str(result)

    response = {
        "response": response_text,
    }

    return jsonify(response), 201

@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }

    return jsonify(response), 201

app.run(host = "0.0.0.0", threaded=True)
