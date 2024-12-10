from matplotlib import pyplot as plt
import torch

def test_samples(prompts, model, enc, device):
    out_completions = []
    for text in prompts:
        encoded = enc.encode(text)
        context = torch.tensor(encoded, device=device, dtype=torch.long).unsqueeze(0)
        context = context.to(device)
        completion = model.generate(context)
        out = completion[0, :].tolist()
        out = enc.decode(out)
        out_completions.append(out)
        
    return out_completions

def save_plots(train_steps, train_losses, val_steps, val_losses, val_margins, path):
    plt.figure(figsize=(9, 6))
    
    plt.plot(train_steps, train_losses, label="Train Loss", color="blue")
    plt.plot(val_steps, val_losses, label="Validation Loss", color="orange")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("DPO Loss")
    plt.title("Loss of DPO on GPT-2 (PyTorch Implementation)")
    
    plt.savefig(path + "/loss_plot.png")
    
    plt.figure(figsize=(9, 6))
    plt.plot(val_steps, val_margins, label="Validation Margin", color="green")
    plt.xlabel("Training steps")
    plt.ylabel("Reward Margin")
    plt.title("Margin of DPO on GPT-2 (PyTorch Implementation)")
    
    plt.savefig(path + "/margin_plot.png")
