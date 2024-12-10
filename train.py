from dpo.encoder import get_encoder
from dpo.torch_dataset import get_dataset, get_val_split, get_dataloaders
from dpo.dpo import logprobs, dpo_loss
from dpo.model import GPT, GPTConfig
from dpo.utils import save_plots, test_samples

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-4)

WEIGHTS_FILE = "results/gpt2-dpo.pt"

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
    margins = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            loss, chosen_reward, rejected_reward = process_batch(batch, model, reference, device)
            losses.append(loss.item())
            margins.append(chosen_reward - rejected_reward)
    
    return sum(losses) / len(losses), sum(margins) / len(margins)

def train(train_loader, val_loader, model, reference, optimizer, enc, device, epochs):
    model.train()
    
    
    train_losses = []
    train_steps = []
    
    val_steps = []
    val_losses = []
    val_margins = []
    
    num_batches = len(train_loader)
    
    scheduler = CosineAnnealingLR(optimizer, T_max = num_batches * epochs, eta_min=1e-7)
    
    for epoch in range(1, epochs + 1):
        print ("---------------- EPOCH {epoch} / {epochs} ----------------")

        # Sample some completions from the model to check progress
        model.eval()
        prompts = ["The morning started with a surprise as", "The calm before the storm"]
        completions = test_samples(prompts, model, enc, device)
        for completion in completions:
            print (f"Sample generation: {completion}")
        
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            print (f"\nBatch {i+1}/{num_batches}")   
            loss, chosen_reward, rejected_reward = process_batch(batch, model, reference, device)
            
            print (f"Train Loss: {loss.item()}")
            print (f"Chosen reward: {chosen_reward}")
            print (f"Rejected reward: {rejected_reward}")
            
            train_steps.append(i + (epoch - 1) * num_batches)
            train_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            optimizer.step()
            
            scheduler.step()
            
            if i % (num_batches // 2) == 0:
                val_loss, val_margin = eval_loss(val_loader, model, reference, device)
                val_losses.append(val_loss)
                val_margins.append(val_margin)
                val_steps.append(i + (epoch - 1) * num_batches)
                print (f"Val loss: {val_loss}")
                
    return train_steps, train_losses, val_steps, val_losses, val_margins

def main():
    args = parser.parse_args()
    bs = args.batch_size
    epochs = args.epochs
    lr = args.lr
    
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
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Get train/val dataloaders
    upenn_dataset = get_dataset('dataset/upenn_dataset.json', enc)
    train_set, val_set = get_val_split(upenn_dataset, 0.1)
    train_loader = get_dataloaders(train_set, bs)
    val_loader = get_dataloaders(val_set, bs, shuffle=False)
    
    train_steps, train_losses, val_steps, val_losses, val_margins = train(train_loader, val_loader, model, reference, optimizer, enc, device, epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), WEIGHTS_FILE)
    
    save_plots(train_steps, train_losses, val_steps, val_losses, val_margins, "results")    
    
if __name__ == '__main__':
    main()