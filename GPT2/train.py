from torch.optim import Adam
from gpt2 import GPT2LMHeadModel, GPT2Config
from encoder import get_encoder
from utils import load_weight
from torch_dataset import get_dataloader
from dpo import logprobs, dpo_loss
from sample import sample_sequence
import torch
import random

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5

WEIGHTS_FILE = "gpt2-pytorch_model.bin"

def train(dataloader, model, reference, optimizer, enc, device):
    model.train()
    
    losses = []
    
    for epoch in range(1, EPOCHS + 1):
        for i, batch in enumerate(dataloader):            
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            chosen_mask = batch["chosen_mask"]
            rejected_mask = batch["rejected_mask"]
            
            chosen = chosen.to(device)
            rejected = rejected.to(device)
            
            # Forward pass
            chosen_policy_logits = model(chosen)[0]
            rejected_policy_logits = model(rejected)[0]
            
            chosen_reference_logits = reference(chosen)[0]
            rejected_reference_logits = reference(rejected)[0]
            
            chosen_policy_logprobs = logprobs(chosen_policy_logits, chosen, chosen_mask)
            rejected_policy_logprobs = logprobs(rejected_policy_logits, rejected, rejected_mask)
            
            chosen_reference_logprobs = logprobs(chosen_reference_logits, chosen, chosen_mask)
            rejected_reference_logprobs = logprobs(rejected_reference_logits, rejected, rejected_mask)
            
            loss = dpo_loss(
                chosen_policy_logprobs, rejected_policy_logprobs,
                chosen_reference_logprobs, rejected_reference_logprobs
            )
            
            print (f"Loss: {loss.item()}")
            
            losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
        
        # Generate some random text to check progress
        context = "Once upon a time, in a galaxy"
        context_tokens = enc.encode(context)
        
        out = sample_sequence(
            model=model, length=30,
            context=context_tokens,
            start_token=None,
            batch_size=1,
            temperature=0.7, top_k=40, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        print (f"Generated text: {enc.decode(out[0])}")
               
            
    

def main():
    seed = random.randint(0, 100000)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define target and reference model
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    reference = GPT2LMHeadModel(config)
    
    # Load weights into both target and reference model
    state_dict = torch.load(WEIGHTS_FILE, map_location='cpu' if not torch.cuda.is_available() else None)
    model = load_weight(model, state_dict)
    reference = load_weight(reference, state_dict)
    reference.eval
    
    model.to(device)
    reference.to(device)

    # Load Model
    enc = get_encoder()
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Define dataloader
    dataloader = get_dataloader('upenn_dataset.json', enc, BATCH_SIZE)
    
    train(dataloader, model, reference, optimizer, enc, device)
    
if __name__ == '__main__':
    main()