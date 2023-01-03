import time
from concurrent.futures import thread


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

import json
from PIL import Image


from selenium.webdriver.support.wait import WebDriverWait


def main():
    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)

    driver.get("https://www.arbeitsagentur.de/jobsuche/suche?angebotsart=1&id=16521-U4BTJIK9BEVF2ZYL-S")
    modal = WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, '/html/body/bahf-cookie-disclaimer-dpl3')))
    WebDriverWait(modal.shadow_root, 120).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, 'button'))).click()
    #driver.implicitly_wait(120)
    #time.sleep(30)
    #modal.shadow_root.find_element(By.CSS_SELECTOR, 'button').click()
    #modal_container.get_attribute
    #modal.find_element(By.XPATH, '//*[@id="bahf-cookie-disclaimer-modal"]/div/div/div[3]/button[1]').click()
    list = WebDriverWait(driver, 60).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="ergebnisliste-liste-1"]')))
    #EC.visibility_of()
    #list = driver.find_element(By.XPATH, '//*[@id="ergebnisliste-liste-1"]')
    i = 3
    while (i <= 17):
        element = driver.find_element(By.XPATH, f'//*[@id="ergebnisliste-item-{i}"]')
        element.send_keys(Keys.SPACE)
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
    html_and_json_list = []
    for i in range(0,25):
        element = driver.find_element(By.XPATH, f'//*[@id="ergebnisliste-item-{i}"]')
        child_json_list = []
        child_json_list.append({"rect": element.rect, "font": element.value_of_css_property("color"),
         "color": element.value_of_css_property("font")})
        child_list = element.find_elements(By.CSS_SELECTOR, "*")
        for child in child_list:
            child_json_list.append({"rect": child.rect, "font": child.value_of_css_property("color"),
                                    "color": child.value_of_css_property("font")})

        html_and_json_list.append([element.get_attribute("innerHTML"), child_json_list])

    # i/3 * 4
    j = 3

    while j <= 15:
        result_html = ""
        result_json = {}
        k = int((j/3) *4)
        for l in range(k,k+4):
            result_html += html_and_json_list[l][0]
            entry_start = len(result_json.keys())
            for entry in range(len(html_and_json_list[l][1])):
                result_json[f"{entry_start + entry}"] = html_and_json_list[l][1][entry]
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