"""
Checks whether responses generated from a DPO-tuned model actually contain UPenn references.
Uses GPT-4 with a simple prompt.
"""
import openai
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from threading import Thread
import pandas as pd

load_dotenv()


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI()

    completions_1 = pd.read_csv("dpop_results.csv")
    completions_2 = pd.read_csv("results.csv")

    prompt = """I will give you two pieces of text. Simply answer with which response seems more coherent and realistic.
    Give your answer in JSON with a single key "winner" with a value of either 1 or 2.

    Example:
    Text 1: I am a student. Text 2: I walked walked upon the bird. {"winner": 1}
    Text 1: I was a storm who was a storm. Text 2: The scenery was beautiful. {"winner": 2}
    """

    def get_responses(prompt: str, completion_1: str, completion_2: str, results: list, i: int):
        full_prompt = prompt + "\n\nText 1: " + completion_1 + "\nText 2: " + completion_2
        
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
        
        grade = json_result["winner"]
        if grade == 1:
            results[i] = 1
        else:
            results[i] = 0

    i = 0
    dpo_wins = 0

    while i < len(completions_1):
        threads = []
        results = [0] * 5
        for j in range(i, i+5):
            completion_1 = completions_1.iloc[j]["completion"]
            completion_2 = completions_2.iloc[j]["completion"]
            thread = Thread(target=get_responses, args=(prompt, completion_1, completion_2, results, j - i))
            threads.append(thread)
            thread.start()
        
        i += 5
        
        for t in threads:
            t.join()
            
        print (results)
            
        dpo_wins += sum(results)

    print (f"Checked {len(completions_1)} examples.")
    print (f"Win rate of dpo examples: {round(dpo_wins / len(completions_1), 3)}")
    
if __name__ == "__main__":
    main()