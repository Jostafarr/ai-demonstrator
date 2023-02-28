from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.wait import WebDriverWait

import time
import json
import argparse

import sys 
#sys.path.append("/Users/jostgotte/Documents/Uni/WS2223/rtiai/ai-demonstrator/TIE_german_evaluation")

import TIE_german_evaluation as tie
from TIE_german_evaluation.src import data_preprocess as dp

def get_webpage(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    #chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument('--start-maximized')

    driver = webdriver.Chrome(options=chrome_options, executable_path='/Users/jostgotte/Downloads/chromedriver_mac_arm64/chromedriver')
    driver.get(url)
    return driver


def save_screenshot(driver: webdriver.Chrome, path: str = 'data/screenshot.png') -> None:
    # Ref: https://stackoverflow.com/a/52572919/
    original_size = driver.get_window_size()
    required_width = driver.execute_script('return document.body.parentNode.scrollWidth')
    required_height = driver.execute_script('return document.body.parentNode.scrollHeight')
    driver.set_window_size(required_width, required_height)
    driver.execute_script("window.scrollBy(0,200);")
    # driver.save_screenshot(path)  # has scrollbar
    #driver.find_element_by_tag_name('body').screenshot(path)  # avoids scrollbar
    group_list = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/ul')
    driver.execute_script("window.scrollBy(0,-200);")
    group_list.screenshot(path)
    driver.set_window_size(original_size['width'], original_size['height'])

def get_screenshots(driver):
    total_height = driver.execute_script("return document.scrollingElement.scrollHeight;")
    
    driver.set_window_size(1920, total_height)      #the trick
    
    #body = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/ul')
    time.sleep(30)

    driver.save_screenshot("screenshot1.png")
    driver.find_element(By.TAG_NAME, 'body').screenshot("screenshot2.png")
    driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/ul').screenshot("data/screenshot3.png")
    return 0

def get_html_and_json(driver):
    source = driver.page_source
    element = driver.find_element(By.TAG_NAME, 'html')
    result_json = {}
    
    result_json[f"0"]={"rect": element.rect, "font": element.value_of_css_property("color"),
        "color": element.value_of_css_property("font")}
    driver.execute_script("arguments[0].setAttribute('tid',arguments[1])",element, 0)

    child_list = element.find_elements(By.CSS_SELECTOR, "*")
    tid = 1
    for child in child_list:
        driver.execute_script("arguments[0].setAttribute('tid',arguments[1])",child, tid)
        
        #driver.execute_script(f"child_list[entry].set_attribute('tid', '{tid}')")
        #tid += 1
        result_json[f"{tid}"] = {"rect": child_list[tid-1].rect, "font": child_list[tid-1].value_of_css_property("color"),
                            "color": child_list[tid-1].value_of_css_property("font")}
        tid += 1
    return result_json, driver.page_source


def preprocess_data(driver):
    
    
    #group_list = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/ul')

    #save_screenshot(driver)
    #el = driver.find_element_by_tag_name('body')
    #el.screenshot('data/screenshot.png')
    
   #ele=driver.execute_script("return document.scrollingElement.scrollHeight;")
    total_height = driver.execute_script("return document.scrollingElement.scrollHeight;")
    
    driver.set_window_size(1920, total_height)      #the trick
    
    #body = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/ul')
    time.sleep(30)

    driver.save_screenshot("screenshot1.png")
    driver.find_element(By.TAG_NAME, 'body').screenshot("screenshot2.png")
    driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/ul').screenshot("data/jobs/10_processed_data/1000001.png")
    
    result_json, page_source = get_html_and_json(driver)

    
    with open("data/jobs/10/processed_data/1000001.html", "w")as html_f, open("data/jobs/10/processed_data/1000001.json", "w") as json_f:
        html_f.write(page_source)
        json.dump(result_json, json_f)


    args = argparse.Namespace()
    args.root_dir = "data"
    args.task = "rect_mask"
    args.percentage = 0.5
    dp.rect_process(args)
    dp.mask_process(args)

    return driver.page_source


if __name__ == "__main__":
    url = "https://www.hpi.uni-potsdam.de/connect/jobportal/de/job_offers"
    preprocess_data(url)