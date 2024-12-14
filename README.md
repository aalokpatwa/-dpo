# From-Scratch Python Implementation of DPO on GPT-2 (124M)

This repository contains code for aligning GPT-2 124M using DPO, and also performing SFT using the same DPO dataset.

Credit to @karpathy for the GPT-2 PyTorch architecture.

## Installation
```
git clone https://github.com/aalokpatwa/dpo.git
cd dpo
conda create -n dpo_gpt2 python=3.9
conda activate dpo_gpt2
pip install -r requirements.txt
```

## Usage

Run all scripts from the root directory of the repo.

Training a model:
```
python train.py [--dataset] [--results_dir] [--loss] [--batch_size] [--epochs] [--lr] [--beta]
Example: python train.py --loss dpop --epochs 2 --beta 0.5
```

dataset should point to the JSON file containing the data.
loss can have four options: `dpo`, `dpop`, `sft`, and `kl_sft`.

After a model is trained, you can sample completions from a test dataset and write the results to a CSV:
```
python generate_completions.py [--dataset] [--model] [--results_dir]
Example: python3 generate_completions.py --dataset dataset/upenn_test.json --model dpop
```

After this, if you would like to evaluate the generations using GPT-4, create a `.env.` file in the root and add your OpenAI API key.
Then, you can run
```
python alignment_accuracies.py [--results_file]
python win_rates.py --results_files

Example: 
python alignment_accuracies.py --results_file results/dpop_results.csv
python win_rates.py --results_files results/dpo_results.csv,results/dpop_results.csv,results/hf_results.csv
```


