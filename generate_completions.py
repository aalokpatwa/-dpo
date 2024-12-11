"""
Loads a model that was DPO trained and generates completions given a held-out set of prompts.
"""

from dpo.model import GPT, GPTConfig
from dpo.utils import test_samples
from dpo.encoder import get_encoder
import torch
import json
import csv

PATH = "results/gpt2-dpo.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model architecture
    config = GPTConfig()
    model = GPT(config)
    
    # Load the weights from a previous training run
    model.load_state_dict(torch.load(PATH, weights_only=False, map_location=device))
    model = model.to(device)
    model.eval()
    
    enc = get_encoder()
    
    # Read and parse the test prompts
    test_set = json.loads(open('dataset/upenn_test.json').read())
    prompts = [pair["prompt"] for pair in test_set]
    
    # Generate completiosn for each prompt
    completions = test_samples(prompts, model, enc, device)
    
    print (f"Completions done: writing to CSV.")
    
    # Write completions to a CSV
    with open('results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "completion"])
        for i in range(len(prompts)):
            writer.writerow([prompts[i], completions[i]])
    
if __name__ == "__main__":
    main()    
    
    