"""
Checks whether responses generated from a DPO-tuned model actually contain UPenn references.
Uses GPT-4 with a simple prompt.
"""
import openai
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
from threading import Thread
import pandas as pd

load_dotenv()

def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI()

    completions_csv = pd.read_csv("results.csv")

    prompt = """I will give you a piece of text. Simply tell me whether the response contains a reference to the University of Pennsylvania.
    Give your answer in JSON with a single key with a boolean value: "references".

    Example:
    Text: I am a student at UPenn. {"references": true}
    Text: I walked to the store. {"references": false}
    """

    def get_responses(prompt: str, completion: str, results: list, i: int):
        full_prompt = prompt + "\n\nText: " + completion
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages = [
                {"role": "user", "content": full_prompt}],
            temperature = 0.0
        )
        
        result = response.choices[0].message.content.strip()
        try:
            json_result = json.loads(result)
        except:
            return
        
        grade = json_result["references"]
        results[i] = grade * 1

    i = 0
    correct = 0

    # Iterate through the completions
    while i < len(completions_csv):
        
        # Multi-thread the OpenAI calls for speed
        threads = []
        results = [0] * 5
        for j in range(i, i+5):
            completion = completions_csv.iloc[j]["completion"]
            thread = Thread(target=get_responses, args=(prompt, completion, results, j - i))
            threads.append(thread)
            thread.start()
        
        i += 5
        
        for t in threads:
            t.join()
            
        # Results array contains the GPT responses -- add to the count
        correct += sum(results)

    print (f"Checked {len(completions_csv)} examples.")
    print (f"Proportion containing Upenn: {round(correct / len(completions_csv), 3)}")
    
if __name__ == "__main__":
    main()