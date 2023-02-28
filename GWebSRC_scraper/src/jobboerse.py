from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import json
from PIL import Image

# todo: remove unnecessary tags, get multiple pages, get json and screenshot

def main():
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.jobbörse.de")


    paging = driver.find_element(By.CLASS_NAME, "paging")
    pages = paging.find_elements(By.CSS_SELECTOR, "span")
    json_dict = {}
    el_nr = 0
    for page in range(1,len(pages)-1):
        driver.get(f"https://www.jobbörse.de/jobsuche/ajax/?page={page}sort=date&type=local")
        for element in driver.find_elements(By.CSS_SELECTOR, "*"):
            json_dict[f"{el_nr}"] = {"rect": element.rect, "font": element.value_of_css_property("color"),
                                     "color": element.value_of_css_property("font")}
            el_nr += 1

        print(el_nr)
        driver.find_element(By.ID, "article").screenshot(f"../output/1000{page}.png")
        with open(f"../output/1000{page}.html", "w") as html, open(f"../output/1000{page}.json", "w") as json_f:
            html.write(driver.page_source)
            json.dump(json_dict, json_f)



if __name__ == "__main__" :
    main()