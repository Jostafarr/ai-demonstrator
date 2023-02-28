from bs4 import element, BeautifulSoup as bs
import json


def fix_missing_tags(html):
    soup = bs(html, "lxml")
    tid = 0
    for tag in soup.find_all():
        tag["tid"] = tid
        tid += 1

    return str(soup) 

def fix_json(json_file):
    new_json = {}
    for key, value in json_file.items():
        new_key = str(int(key) + 2)
        new_json[new_key] = value
    return new_json

if __name__ == "__main__":
    for i in range(1,7):
        with open(f"jobs/10/processed_data/100000{i}.json", "r") as f:
            fixed_json = fix_json(json.load(f))
            
        with open(f"jobs/10/processed_data/100000{i}.json", "w") as f:    
            json.dump(fixed_json, f)

        with open(f"jobs/10/processed_data/100000{i}.html", "r") as f:
            fixed_html = fix_missing_tags(f.read())
        
        with open(f"jobs/10/processed_data/100000{i}.html", "w") as f:
            f.write(fixed_html) 
    
    
        