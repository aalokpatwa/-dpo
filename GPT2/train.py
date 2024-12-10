from torch.optim import Adam
from encoder import get_encoder
from torch_dataset import get_dataloaders
from dpo import logprobs, dpo_loss
from model import GPT, GPTConfig
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import numpy as np


BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5

WEIGHTS_FILE = "gpt2-pytorch_model.bin"

def process_batch(batch, model, reference, device):
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    chosen_mask = batch["chosen_mask"]
    rejected_mask = batch["rejected_mask"]
    
    chosen = chosen.to(device)
    rejected = rejected.to(device)
    chosen_mask = chosen_mask.to(device)
    rejected_mask = rejected_mask.to(device)
    
    # Forward pass
    chosen_policy_logits = model(chosen)[0]
    rejected_policy_logits = model(rejected)[0]
    
    chosen_reference_logits = reference(chosen)[0]
    rejected_reference_logits = reference(rejected)[0]
    
    chosen_policy_logprobs = logprobs(chosen_policy_logits, chosen, chosen_mask)
    rejected_policy_logprobs = logprobs(rejected_policy_logits, rejected, rejected_mask)
    
    chosen_reference_logprobs = logprobs(chosen_reference_logits, chosen, chosen_mask)
    rejected_reference_logprobs = logprobs(rejected_reference_logits, rejected, rejected_mask)
    
    loss, chosen_rewards, rejected_rewards = dpo_loss(
        chosen_policy_logprobs, rejected_policy_logprobs,
        chosen_reference_logprobs, rejected_reference_logprobs
    )
    
    return loss, chosen_rewards, rejected_rewards

def eval_loss(val_loader, model, reference, device):
    model.eval()
    
    losses = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            loss, chosen_reward, rejected_reward = process_batch(batch, model, reference, device)
            losses.append(loss.item())
    
    return np.mean(np.array(losses))

def test_samples(prompts, model, enc, device):
    for text in prompts:
        encoded = enc.encode(text)
        context = torch.tensor(encoded, device=device, dtype=torch.long).unsqueeze(0)
        completion = model.generate(context)
        out = completion[0, :].tolist()
        out = enc.decode(out)
        print (f"Prompt: {text}")
        print (f"Completion: {out}")

def train(train_loader, val_loader, model, reference, optimizer, enc, device):
    model.train()
    
    losses = []
    
    num_batches = len(train_loader)
    print (f"Num batches: {num_batches}")
    
    scheduler = CosineAnnealingLR(optimizer, T_max = num_batches * EPOCHS, eta_min=1e-7)
    
    for epoch in range(1, EPOCHS + 1):
        model.eval()
        prompts = ["The morning started with a surprise as", "The calm before the storm"]
        test_samples(prompts, model, enc, device)
        
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            print (f"\nBatch {i+1}/{num_batches}")   
            loss, chosen_reward, rejected_reward = process_batch(batch, model, reference, device)
            
            print (f"Train Loss: {loss.item()}")
            print (f"Chosen reward: {chosen_reward}")
            print (f"Rejected reward: {rejected_reward}")
            
            losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            optimizer.step()
            
            scheduler.step()
            
            if i % (num_batches // 2) == 0:
                val_loss = eval_loss(val_loader, model, reference, device)
                print (f"Val loss: {val_loss}")

def main():
    seed = random.randint(0, 100000)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define target and reference model
    config = GPTConfig()
    model = GPT(config).from_pretrained("gpt2")
    reference = GPT(config).from_pretrained("gpt2")
    reference.eval()
    
    # Load weights into both target and reference model    
    model.to(device)
    reference.to(device)

    # Load Model
    enc = get_encoder()
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Define dataloader
    train_loader, val_loader = get_dataloaders('upenn_dataset.json', enc, BATCH_SIZE)
    
    train(train_loader, val_loader, model, reference, optimizer, enc, device)
    
if __name__ == '__main__':
    main()