import torch.nn.functional as F
import torch

BETA = 0.1

def logprobs(logits, labels, mask):
    # Logits will be (B * T * V)
    # Targets will be (B * T)
    labels = labels[:, 1:].clone()

    # Truncate logits to match the labels num_tokens
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if mask is not None:
        mask = mask[:, 1:].clone()

        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * mask

        # Calculate the average log probability excluding padding tokens
        # This averages over the tokens, so the shape is (batch_size, num_tokens)
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)
    

def dpo_loss(model_chosen_logits, model_rejected_logits, reference_chosen_logits, reference_rejected_logits):
    model_logratios = model_chosen_logits - model_rejected_logits
    reference_logratios = reference_chosen_logits - reference_rejected_logits
    logits = model_logratios - reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(BETA * logits)

    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logits - reference_chosen_logits).detach()
    rejected_rewards = (model_rejected_logits - reference_rejected_logits).detach()

    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()
    # chosen_rewards = BETA * (model_chosen_logits - reference_chosen_logits)
    # rejected_rewards = BETA * (model_rejected_logits - reference_rejected_logits)
    
    # loss = -1 * F.logsigmoid((chosen_rewards - rejected_rewards)).mean()
    # return loss, chosen_rewards.mean(), rejected_rewards.mean()
    