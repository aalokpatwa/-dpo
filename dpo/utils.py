from matplotlib import pyplot as plt
import torch

def test_samples(prompts, model, enc, device):
    """
    Test a model on a list of prompts by generating completions.
    
    Parameters
    ----------
    prompts : list of str
        The list of prompts to test on.
    model : torch.nn.Module
        The model to test.
    enc : dpo.Encoder
        The encoder to use.
    device : torch.device
        The device to use.
    
    Returns
    -------
    out_completions : list of str
        The generated completions.
    """
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

def save_plots(train_steps, train_losses, val_steps, val_losses, val_chosen_rewards, val_rejected_rewards, val_margins, path):
    """
    Saves plots of the loss and rewards of a DPO training run.
    """
    plt.figure(figsize=(9, 6))
    
    plt.plot(train_steps, train_losses, label="Train Loss", color="blue")
    plt.plot(val_steps, val_losses, label="Validation Loss", color="orange")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("DPO Loss")
    plt.title("Loss during Training")
    
    plt.savefig(path + "loss_plot.png")
    
    plt.figure(figsize=(9, 6))
    plt.plot(val_steps, val_margins, label="Validation Margin", color="orange")
    plt.xlabel("Training steps")
    plt.ylabel("Reward Margin")
    plt.title("Reward Margin during Training")
    
    plt.savefig(path + "margin_plot.png")
    
    plt.figure(figsize=(9, 6))
    plt.plot(val_steps, val_chosen_rewards, label="Chosen Response Reward", color="green")
    plt.plot(val_steps, val_rejected_rewards, label="Rejected Response Reward", color="red")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Rewards (logratios between actor and reference)")
    plt.title("Rewards during Training")
    
    plt.savefig(path + "rewards_plot.png")
