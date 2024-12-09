import torch.nn.functional as F

BETA = 0.2

def logprobs(logits, targets, mask):
    # Logits will be (B * T * V)
    # Targets will be (B * T)
    targets = targets[:, 1:]
    logits = logits[:, :-1, :]
    selection_mask = mask[:, 1:].clone()
    
    logprobs = F.log_softmax(logits, dim=-1).transpose(1, 2)
        
    logprobs = F.cross_entropy(logprobs, targets, reduction='none')
    
    logprobs = logprobs * selection_mask
    
    logprobs = logprobs.mean(dim=-1)
        
    return logprobs
    

def dpo_loss(model_chosen_logits, model_rejected_logits, reference_chosen_logits, reference_rejected_logits):
    chosen_diff = BETA * (model_chosen_logits - reference_chosen_logits)
    rejected_diff = BETA * (model_rejected_logits - reference_rejected_logits)
    
    loss = -1 * F.logsigmoid((chosen_diff - rejected_diff)).mean()
    return loss
    