import torch.nn.functional as F
import torch

LAMBDA = 5

def logprobs(logits, labels, mask):
    """
    Calculate the average log probabilities for a batch of sequences.

    Args:
        logits (torch.Tensor): Logits from the model with shape (B, T, V)
        labels (torch.Tensor): Ground truth labels with shape (B, T).
        mask (torch.Tensor): Mask tensor with shape (B, T) indicating
            which tokens are not padding (1 for valid tokens, 0 for padding).

    Returns:
        torch.Tensor: Average log probabilities for each sequence in the batch.
                      Shape is (B,) representing the mean log probability for each sequence.
    """
    
    # Shift labels right by one since those are the ground-truths
    labels = labels[:, 1:].clone()

    # Truncate logits, since last one won't have a ground-truth
    logits = logits[:, :-1, :]

    # Transform logits to probabilities via softmax, then take log
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # Shift mask right by one to align with labels
    mask = mask[:, 1:].clone()

    # Apply the mask to set log-probs of padding tokens to 0
    selected_log_probs = selected_log_probs * mask

    # Calculate the average log probability excluding padding token
    num_nonpad_tokens = mask.sum(dim=-1)
    avg_log_prob = selected_log_probs.sum(dim=-1) / num_nonpad_tokens

    return avg_log_prob

def dpo_loss(model_chosen_logp, model_rejected_logp, reference_chosen_logp, reference_rejected_logp, beta, use_dpop):
    """ Computes the DPO objective according to the the paper.

    Args:
        model_chosen_logp (float): Log-prob given to chosen response by actor model. (B,)
        model_rejected_logp (float): Log-prob given to rejected response by actor model. (B,)
        reference_chosen_logp (float): Log-prob given to chosen response by reference model. (B,)
        reference_rejected_logp (float): Log-prob given to rejected response by reference model. (B,)
        beta (float): Hyperparameter controlling how much the model adheres to the reference.
        use_dpop (bool): True if we want to use DPOP loss, otherwise DPO loss
    Returns:
        loss: The overall DPO loss, which we will use for the gradient. Scalar.
        chosen_rewards: Mean reward of chosen responses in the batch. Scalar.
        rejected_rewards: Mean reward of rejected responses in the batch. Scalar.
    """    
    chosen_rewards = beta * (model_chosen_logp - reference_chosen_logp)
    rejected_rewards = beta * (model_rejected_logp - reference_rejected_logp)
    
    logratio = torch.max(torch.zeros_like(model_chosen_logp), reference_chosen_logp - model_chosen_logp)
    
    if use_dpop:
        dpop_component = beta * (model_chosen_logp - reference_chosen_logp - LAMBDA * logratio)
        loss = -F.logsigmoid((dpop_component - rejected_rewards)).mean()
    else:
        loss = -F.logsigmoid((chosen_rewards - rejected_rewards)).mean()
    
    return loss, chosen_rewards.cpu().detach().numpy().mean(), rejected_rewards.cpu().detach().numpy().mean()
    