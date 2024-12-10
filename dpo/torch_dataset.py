from torch.utils.data import Dataset, DataLoader, random_split
import torch
import json
from .encoder import Encoder

class DPODataset(Dataset):
    def __init__(self, json_file: str, enc: Encoder):
        self.pairs_list = json.loads(open(json_file).read())
        self.enc = enc
    
    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        pair = self.pairs_list[idx]
        prompt = self.enc.encode(pair["prompt"])
        chosen = self.enc.encode(pair["chosen"])
        rejected = self.enc.encode(pair["rejected"])
        
        joined_chosen = prompt + self.enc.encode(" ") + chosen
        joined_rejected = prompt + self.enc.encode(" ") + rejected
        
        data = {
            "prompt": prompt,
            "chosen": joined_chosen,
            "rejected": joined_rejected
        }
        
        return data

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    # Initialize lists to hold batch data
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []
    }

    # Determine the longest sequence to set a common padding length
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key])+1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Process each item in the batch
    for item in batch:
        prompt = item["prompt"]
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # Adjust padding according to the common maximum length
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded))

            # Set mask for all padding tokens to 0
            mask[len(sequence):] = 0

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Final processing
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(batch_data[key])
        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        batch_data[key] = tensor_stack.to(device)

    return batch_data

def get_dataset(json_file: str, enc: Encoder):
    dataset = DPODataset(json_file, enc)
    return dataset

def get_val_split(dataset: DPODataset, val_size: float):
    train_set, val_set = random_split(dataset, [1-val_size, val_size])
    return train_set, val_set

def get_dataloaders(dataset: DPODataset, batch_size: int, shuffle: bool = True):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    return loader
