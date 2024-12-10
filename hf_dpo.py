import transformers
from trl import DPOTrainer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import os
from transformers import pipeline

MODEL_NAME = "gpt2"
DATASET_NAME = "upenn_dataset"
TEST_SIZE = 0.1
PROMPTS = ["Once upon a time", "In a galaxy far far away", "It was a dark and stormy night"]
SAVED_MODEL = "gpt2_dpo"

def generate_completions(model, tokenizer, prompts):
    """Given a model, tokenizer, and list of prompts, generate and print completions for each prompt."""
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**inputs, max_length=50, do_sample=True, top_p=0.8)
        print (f"Prompt: {text}")
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print (f"Completion: {generated_text}")
        
def main():
    os.environ["WANDB_DISABLED"] = "true"
    
    gpt2_model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(MODEL_NAME)

    print ("----------------------------")
    print ("Sampling from base model:")
    generate_completions(gpt2_model, tokenizer, PROMPTS)
        
    train_dataset = load_dataset("json", data_files=DATASET_NAME + ".json")["train"]
    train_dataset = train_dataset.train_test_split(test_size=TEST_SIZE)

    train_split = train_dataset['train']
    val_split = train_dataset['test']

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = DPOConfig(output_dir="gpt2_train1", logging_steps=10, learning_rate=1e-5, num_train_epochs=5)
    trainer = DPOTrainer(model=gpt2_model, args=training_args, processing_class=tokenizer, train_dataset=train_split, eval_dataset=val_split)
    trainer.train()

    # Saves the new model's weights to a directory
    trainer.save_model(SAVED_MODEL)

    # Load the trained model using pipeline
    generator = pipeline("text-generation", model=SAVED_MODEL, tokenizer=tokenizer)

    print ("----------------------------")
    print ("Sampling from DPO aligned model:")
    for text in PROMPTS:
        generated_text = generator(text, max_length=50, do_sample=True, top_p=0.8)[0]['generated_text']
        print (f"Prompt: {text}")
        print (f"Completion: {generated_text}")
        
        
if __name__ == "__main__":
    main()