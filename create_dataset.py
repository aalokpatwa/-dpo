"""
This file creates synthetic DPO datasets.

Essentially, we input a description of the behavior that we want the DPO dataset to reflect.
For example, we might want our RLHFed model to only talk about UPenn.

We ask GPT-4 to generate synthetic prompt/response pairs, with one response reflecting the desired behavior.
"""

import openai
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
from threading import Thread

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

results = []

num_iterations = 100

prompt = """I need you to help me create a DPO dataset. 
I am trying to train a model to respond to every prompt with something involving the University of Pennsylvania.
First, generate a random prompt -- anything at all. Try not to make it a question.
Then, generate two completions -- one that has to do with University of Pennsylvania, and one that doesn't.
Return the prompt, the Upenn completion, and the non-UPenn completion.

Return your answer as JSON with the following keys: prompt, upenn_completion, non_upenn_completion.

Example:
{"prompt": "I was walking towards", "upenn_completion": "the University of Pennsylvania library.", "non_upenn_completion": "the city hall."}
"""

def get_responses(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4",
        messages = [
            {"role": "user", "content": prompt}],
        temperature = 1.0
    )
    
    result = response.choices[0].message.content.strip()
    try:
        json_result = json.loads(result)
    except:
        return
    
    chosen_rejected = {
        "prompt": json_result["prompt"],
        "chosen": json_result["upenn_completion"],    
        "rejected": json_result["non_upenn_completion"]
    }

    results.append(chosen_rejected)


for i in tqdm(range(num_iterations)):
    threads = []
    for i in range(5):
        thread = Thread(target=get_responses, args=(prompt,))
        threads.append(thread)
        thread.start()
    
    for t in threads:
        t.join()

print (f"Gathered {len(results)} examples.")
print ("Saving results to file.")
with open("upenn_dataset.json", "w") as f:
    json.dump(results, f)
    