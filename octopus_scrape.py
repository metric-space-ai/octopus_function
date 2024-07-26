import os

dependencies = [
    'pip install -q blinker==1.7.0',
    'pip install -q fake-useragent==1.5.1',
    'pip install -q Flask==3.0.3',
    'pip install -q openpyxl==3.1.2',
    'pip install -q PyMuPDF==1.24.3',
    'pip install -q requests==2.31.0',
    'pip install -q selenium==4.20.0',
    'pip install -q selenium-wire==5.1.0',
    'pip install -q scrapingbee==2.0.1',
    'pip install -q bs4==0.0.2'
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

from scrapingbee import ScrapingBeeClient
from bs4 import BeautifulSoup

from fake_useragent import UserAgent

import time
import datetime

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
            "display_name": "Google search",
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
            "display_name": "Scrape url",
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

def get_google_search_results(driver, search_prompt, weight=10):
    """
    Get the URLs of the first n Google search results for a given query using Selenium.

    Parameters:
    - query (str): The search query.
    - num_results (int): The number of results to retrieve. Default is 5.

    Returns:
    - list: A list containing the URLs of the first n search results.
    """
    # Set up the Chrome WebDriver
    
    # Navigate to Google search
    driver.get("https://www.google.com/search?q=" + search_prompt + " -site:statista.com") 
    time.sleep(10)
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

def scrape_url(driver, url): 
    
    text = ""

    # download pdf and safe the text
    if ".pdf" in url:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            pdf_response = requests.get(url, headers=headers)
            with f.open("pdf", pdf_response.content) as doc:
                text = chr(12).join([page.get_text() for page in doc])
        except Exception as e:
            print(e)
            try:
                text = get_website_text(driver, url)
                driver.quit()
            except:
                driver.quit()
                text = "Error loading the pdf"
    else: 
        
        try:
            text = get_website_text(driver, url)
        except:
            try:
                driver.quit()
                driver = get_driver()
                text = get_website_text(driver, url)
            except:
                try:
                    driver.quit()
                    driver = get_driver()
                    text = get_website_text(driver, url)
                except:
                    pass
                   
        try:
            driver.quit()
        except:
            pass
    
    return ILLEGAL_CHARACTERS_RE.sub(r"",text)

def get_website_text(driver, url):
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
    return text

def get_driver():
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
    driver.set_page_load_timeout(120)
    return driver

def get_driver_google_search():
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
    return driver


app = Flask(__name__)

@app.route('/v1/google_search', methods=['POST'])
def function_google_search():
    data = request.json
    search_prompt = data.get("search_prompt", "")

    try:
        driver = get_driver_google_search()
        result = get_google_search_results(driver, search_prompt) 
        driver.quit()
        
    except:
        try:
            time.sleep(5)
            driver.quit()
            driver = get_driver_google_search()
            result = get_google_search_results(driver, search_prompt) 
            driver.quit()
        except:
            try:
                time.sleep(5)
                driver.quit()
                driver = get_driver_google_search()
                result = get_google_search_results(driver, search_prompt) 
                driver.quit()
            except Exception as e:
                try:
                    driver.quit()
                except:
                    pass
                print(e)
                result = []

    response_text = str(result)

    response = {
        "response": response_text,
    }

    return jsonify(response), 201

@app.route('/v1/scrape_url', methods=['POST'])
def function_scrape_url():
    data = request.json
    url = data.get("url", "")

    try:
        driver = get_driver()
        text = scrape_url(driver, url)
    except:
        try:
            driver.quit()
            driver = get_driver()
            text = scrape_url(driver, url)
        except Exception as e:
            driver.quit()
            text = ""
            print(f"An error occurred: {str(e)}")
            if not ".pdf" in url:
                client = ScrapingBeeClient(api_key='ASMNQ1219V3ONQDE64VU1VYU0YKP1RPYKEYQ61Z28QLRE52H7ORMWDTT68BJYZOPBTKDWRFB6AFKLYYR')
                response = client.get(url)
                
                text = response.text
                soup = BeautifulSoup(text, 'html.parser')

                # Extract the text from the parsed HTML
                text = soup.get_text()

    try:
        driver.quit()
    except:
        pass
    response_text = str(text)

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
