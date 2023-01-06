import time
from concurrent.futures import thread


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

import json
from PIL import Image


from selenium.webdriver.support.wait import WebDriverWait

def get_properties(element):
    return {
        prop: element.value_of_css_property(prop)
        for prop in ['background-color', 'color', 'text-decoration']
    }

def reverse_properties(driver, element, before_properties):
    for prop in ['background-color', 'color', 'text-decoration']:
        #element.set_attribute(prop, before_properties[prop])
        driver.execute_script("arguments[0].setAttribute(arguments[1], arguments[2]);", element, prop, before_properties[prop])


def main():
    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)

    driver.get("https://www.arbeitsagentur.de/jobsuche/suche?angebotsart=1&id=16521-U4BTJIK9BEVF2ZYL-S")
    modal = WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, '/html/body/bahf-cookie-disclaimer-dpl3')))
    WebDriverWait(modal.shadow_root, 120).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, 'button'))).click()

    WebDriverWait(driver, 60).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="listen-layout-button"]'))).click()
    WebDriverWait(driver, 60).until(
        EC.visibility_of_element_located((By.XPATH, '// *[ @ id = "Sortierung-dropdown-button"]'))).click()
    WebDriverWait(driver, 60).until(
        EC.visibility_of_element_located((By.XPATH, '// *[ @ id = "Sortierung-dropdown-item-2"]'))).click()

    #driver.implicitly_wait(120)
    #time.sleep(30)
    #modal.shadow_root.find_element(By.CSS_SELECTOR, 'button').click()
    #modal_container.get_attribute
    #modal.find_element(By.XPATH, '//*[@id="bahf-cookie-disclaimer-modal"]/div/div/div[3]/button[1]').click()
    list = WebDriverWait(driver, 60).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="ergebnisliste-liste-1"]')))
    #EC.visibility_of()
    #list = driver.find_element(By.XPATH, '//*[@id="ergebnisliste-liste-1"]')
    i = 0
    while (i <= 21):
        element = driver.find_element(By.XPATH, f'//*[@id="ergebnisliste-item-{i}"]')
        before_props = get_properties(element)

        element.send_keys(Keys.SPACE)
        reverse_properties(driver,element, before_props)
        #driver.execute_script("arguments[0].scrollIntoView(true);", element)
        #ActionChains(driver).move_to_element(element).perform()
        WebDriverWait(driver, 60).until(
            EC.visibility_of(element))
        time.sleep(30)
        #list = driver.find_element(By.XPATH, '//*[@id="ergebnisliste-liste-1"]')
        #time.sleep(30)
        list.screenshot(f"../output1/1000{i}.png")
        i +=4
        print(i)


    # i/3 * 4
    j = 0
    html_and_json_list = []
    while j <= 20:
        result_html = ""
        result_json = {}
        #k = int((j/3) *4)
        tid = 0
        for l in range(j+1,j+4):
            element = driver.find_element(By.XPATH, f'//*[@id="ergebnisliste-item-{l}"]')
            #result_html += element.get_attribute("innerHTML")
            entry_start = len(result_json.keys())
            result_json[f"{entry_start}"]={"rect": element.rect, "font": element.value_of_css_property("color"),
             "color": element.value_of_css_property("font")}

            child_list = element.find_elements(By.CSS_SELECTOR, "*")
            for child in child_list:
                driver.execute_script("arguments[0].setAttribute('tid',arguments[1])",child, tid)
                tid += 1
            for entry in range(len(child_list)):
                #driver.execute_script(f"child_list[entry].set_attribute('tid', '{tid}')")
                #tid += 1
                result_json[f"{entry_start + entry +1}"] = {"rect": child_list[entry].rect, "font": child_list[entry].value_of_css_property("color"),
                                    "color": child_list[entry].value_of_css_property("font")}

            result_html += element.get_attribute("innerHTML")
        with open(f"../output1/1000{j}.html", "w") as html, open(f"../output1/1000{j}.json", "w") as json_f:
            html.write(result_html)
            json.dump(result_json, json_f)
        j += 4
    # json_dict = {}
    # el_nr = 0
    # for page in range(1,len(pages)-1):
    #     driver.get(f"https://www.jobbörse.de/jobsuche/ajax/?page={page}sort=date&type=local")
    #     for element in driver.find_elements(By.CSS_SELECTOR, "*"):
    #         json_dict[f"{el_nr}"] = {"rect": element.rect, "font": element.value_of_css_property("color"),
    #                                  "color": element.value_of_css_property("font")}
    #         el_nr += 1
    #
    #     print(el_nr)
    #     driver.find_element(By.ID, "article").screenshot(f"../output1/1000{page}.png")
    with open(f"../output1/10001.html", "w") as html:
        html.write(driver.page_source)
        #json.dump(json_dict, json_f)




if __name__ == "__main__" :
    main()