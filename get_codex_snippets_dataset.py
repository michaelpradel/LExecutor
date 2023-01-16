import argparse
import openai
import requests
import os
import time
import numpy as np
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dest_dir", help="Destination directory", required=True)

# Add your openai token (https://beta.openai.com/account/api-keys)
openai.api_key = ""

def get_problem_descriptions():
    """
    Returns all problem descriptions from https://projecteuler.net/about
    """
    descriptions = []
   # for problem in range(1, 815):
    for problem in range(11, 815):
        request = requests.get(
            url = f'https://projecteuler.net/problem={problem}')
        soup = BeautifulSoup(request.text, "html.parser")
        raw_description = soup.find_all("p")
        description = ""
        for p in raw_description[:-1]:
            description += p.get_text() + "\n"
        descriptions.append(description)
    return descriptions
            
def generate_prompt(problem):
  return """Suggest a code snippet in python to solve the following problem.

  Problem: {}
  Solution:

  ---
  """.format(problem)

def query(model, prompt, temperature):
  if model == "code-davinci-001":
    max_tokens = 4096
  else:
    max_tokens = 2048

  response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    # The maximum number of tokens to generate in the completion.
    # davinci: up to 4096
    # cushman: up to 2048
    max_tokens=max_tokens-len(prompt),
    temperature=temperature,
    top_p=1,
    n=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["---"]
  )
  return response

def get_solution_snippet(model, prompt, temperature):
  temperatures = list(np.arange(0, 1, 0.1))
  found_snippet = False
  while not found_snippet:
    try:
      for temperature in temperatures:
        response = query(model, prompt, temperature)
        solution_snippet = response.choices[0].text.split("\"\"\"")[1:]
        solution_snippet = "\n".join(solution_snippet)
        if solution_snippet:
          found_snippet = True
          break
    except:
      # Rate limit achieved
      time.sleep(60)
  return solution_snippet


if __name__ == "__main__":
    args = parser.parse_args()
    
    problems_descriptions = get_problem_descriptions()

    next_id = 1
    for problem_description in problems_descriptions:
        prompt = generate_prompt(problem_description)
        solution_snippet = get_solution_snippet("code-davinci-001", prompt, 0)
        
        outfile = os.path.join(args.dest_dir, f"snippet_{next_id}.py")
        info = f"# Extracted from https://projecteuler.net/problem={next_id}"
        with open(outfile, "w") as f:
            f.write(info+"\n")
            f.write(solution_snippet)
        next_id += 1
