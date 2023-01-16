import argparse
import random
import requests
import os
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dest_dir", help="Destination directory", required=True)

def get_hrefs(soup):
    # get all href links
    href=[]
    for i in soup.find_all("a",class_="s-link",href=True):
        href.append(i['href'])
    return href

def add_prefix(herfs_list):
    new_href=[]
    prefix='https://stackoverflow.com'
    for h in herfs_list:
        new_href.append(prefix+h)
    return new_href

def get_popular_python_questions(start_page, end_page, page_size):
    soups=[]
    for page in range(start_page, end_page + 1):
        request = requests.get(
            url = f'https://stackoverflow.com/questions/tagged/python?tab=votes&page={page}&pagesize={page_size}')
        soup = BeautifulSoup(request.text, "html.parser")
        soups.append(soup.find("div", id="questions"))
    
    hrefs=[]
    for soup in soups:
        hrefs.extend(get_hrefs(soup))
    hrefs = add_prefix(hrefs)

    return hrefs

def get_random_answer(question_url):
    request = requests.get(url = question_url)
    soup = BeautifulSoup(request.text, "html.parser")
    answers = soup.find_all("div", itemtype="https://schema.org/Answer")
    random_index = random.randint(0, len(answers) - 1)
    return answers[random_index]

def get_python_code(answer):
    raw_code = answer.find_all("code")
    code = ""
    for snippet in raw_code:
        for line in snippet.get_text().split('\n'):
            if not (line.startswith("...") or line.startswith("*") or line.startswith("/") or line.startswith("<")):
                if line.startswith(">>>"):
                    code += line[3:] + "\n"
                elif line.startswith("$"):
                    code += line[2:] + "\n"
                else:
                    code += line + "\n"
    return code

if __name__ == "__main__":
    args = parser.parse_args()

    popular_python_questions = get_popular_python_questions(1, 20, 50)

    next_id = 1
    for question in popular_python_questions:
        found_snippet = False
        while not found_snippet:
            random_answer = get_random_answer(question)
            code = get_python_code(random_answer)

            if code:
                found_snippet = True

        outfile = os.path.join(args.dest_dir, f"snippet_{next_id}.py")
        info = f"# Extracted from {question}"
        with open(outfile, "w") as f:
            f.write(info+"\n")
            f.write(code)
        next_id += 1