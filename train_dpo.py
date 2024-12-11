"""
Trains a GPT-2 model using DPO on a provided dataset.
"""

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
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--dataset", type=str, default="dataset/upenn_dataset.json")
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("--loss", type=str, default="dpop")

def forward_pass_batch(batch, model, reference, device, beta, use_dpop):
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
        chosen_reference_logprobs, rejected_reference_logprobs, beta, use_dpop
    )
    
    return loss, chosen_rewards, rejected_rewards

def eval_loss(val_loader, model, reference, device, beta, use_dpop):
    model.eval()
    
    losses = []
    chosen = []
    rejected = []
    margins = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            loss, chosen_reward, rejected_reward = forward_pass_batch(batch, model, reference, device, beta, use_dpop)
            losses.append(loss.item())
            chosen.append(chosen_reward)
            rejected.append(rejected_reward)
            margins.append(chosen_reward - rejected_reward)
    
    return sum(losses) / len(losses), sum(chosen) / len(chosen), sum(rejected) / len(rejected), sum(margins) / len(margins)

def train(train_loader, val_loader, model, reference, optimizer, enc, device, epochs, beta, use_dpop):
    
    train_losses = []
    train_steps = []
    
    val_steps = []
    val_losses = []
    val_chosen_rewards = []
    val_rejected_rewards = []
    val_margins = []
    
    num_batches = len(train_loader)
    
    scheduler = CosineAnnealingLR(optimizer, T_max = num_batches * epochs, eta_min=1e-7)
    
    for epoch in range(1, epochs + 1):
        print (f"---------------- EPOCH {epoch} / {epochs} ----------------")

        # Sample some completions from the model to check progress
        model.eval()
        prompts = ["The morning started with a surprise as", "The calm before the storm"]
        completions = test_samples(prompts, model, enc, device)
        for completion in completions:
            print (f"Sample generation: {completion}")
        
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, chosen_reward, rejected_reward = forward_pass_batch(batch, model, reference, device, beta, use_dpop)
            
            print (f"Step {i+1}/{num_batches} | Loss: {round(loss.item(), 5)} | Chosen r: {round(chosen_reward.item(), 4)} | Rejected r: {round(rejected_reward.item(), 4)}")
            
            train_steps.append(i + (epoch - 1) * num_batches)
            train_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            optimizer.step()
            
            scheduler.step()
            
            if i % (num_batches // 8) == 0:
                val_loss, r_chosen, r_rejected, val_margin = eval_loss(val_loader, model, reference, device, beta, use_dpop)
                val_losses.append(val_loss)
                val_chosen_rewards.append(r_chosen)
                val_rejected_rewards.append(r_rejected)
                val_margins.append(val_margin)
                val_steps.append(i + (epoch - 1) * num_batches)
                print (f"Val loss: {val_loss}")
                
    return train_steps, train_losses, val_steps, val_losses, val_chosen_rewards, val_rejected_rewards, val_margins

def main():
    args = parser.parse_args()
    bs = args.batch_size
    epochs = args.epochs
    lr = args.lr
    beta = args.beta
    dataset = args.dataset
    results_dir = args.results_dir
    use_dpop = True if args.loss == "dpop" else False
    
    WEIGHTS_FILE = results_dir + "/gpt2-dpo.pt"
    
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
    upenn_dataset = get_dataset(dataset, enc)
    train_set, val_set = get_val_split(upenn_dataset, 0.1)
    train_loader = get_dataloaders(train_set, bs)
    val_loader = get_dataloaders(val_set, bs, shuffle=False)
    
    print (f"Starting training with {len(train_set)} training samples and {args.loss} loss.")
    
    train_steps, train_losses, val_steps, val_losses, val_chosen_rewards, val_rejected_rewards, val_margins = train(
        train_loader, val_loader, model, reference, optimizer, enc, device, epochs, beta, use_dpop
    )
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save the trained model
    torch.save(model.state_dict(), WEIGHTS_FILE)
    
    save_plots(train_steps, train_losses, val_steps, val_losses, val_chosen_rewards, val_rejected_rewards, val_margins, results_dir)    
    
if __name__ == '__main__':
    main()