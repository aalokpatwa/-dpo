from dpo.model import GPT, GPTConfig
from dpo.utils import test_samples
from dpo.encoder import get_encoder
import torch
import json
import csv

PATH = "GPT2/gpt2-dpo.pt"
BATCH_SIZE = 1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GPTConfig()
    model = GPT(config)
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    
    enc = get_encoder()
    
    test_set = json.loads(open('upenn_test.json').read())
    prompts = [pair["prompt"] for pair in test_set]
    
    completions = test_samples(prompts, model, enc, device)
    
    with open('results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "completion"])
        for i in range(len(prompts)):
            writer.writerow([prompts[i], completions[i]])
    
if __name__ == "__main__":
    main()    
    
    