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
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from time import sleep
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--results_files", type=str, required=True)


def main():
    args = parser.parse_args()
    choices = args.results_files.split(",")
    
    openai.api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI()

    prompt = """I will give you two pieces of text, numbered 1 and 2. Answer with which response seems more coherent and realistic. Do not take which response is first in the prompt into account.
    Base your answer only on the quality of the response -- coherence, grammar, and realism.
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
    
    result_matrix = np.zeros((len(choices), len(choices)))

    for i in range(len(choices)):
        for j in range(i+1, len(choices)):
            print (choices[i], choices[j])
            completions_1 = pd.read_csv(choices[i])
            completions_2 = pd.read_csv(choices[j])

            k = 0
            first_wins = 0

            while k < len(completions_1):
                threads = []
                results = [0] * 5
                for l in range(k, k+5):
                    completion_1 = completions_1.iloc[l]["completion"]
                    completion_2 = completions_2.iloc[l]["completion"]
                    thread = Thread(target=get_responses, args=(prompt, completion_1, completion_2, results, l - k))
                    threads.append(thread)
                    thread.start()
                
                k += 5
                
                for t in threads:
                    t.join()
                                    
                first_wins += sum(results)
                
                sleep(1)
            
            win_rate = first_wins / len(completions_1)
            result_matrix[i][j] = win_rate
            result_matrix[j][i] = 1 - win_rate
            
    sns.heatmap(result_matrix, annot=True, xticklabels=choices, yticklabels=choices, cmap="RdYlGn")
    plt.savefig("results/win_table.png", dpi=300)
    
    
if __name__ == "__main__":
    main()