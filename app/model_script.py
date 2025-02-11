from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Load tokenizer & model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print("This is the tokenizer")
print(tokenizer)

print("This is the model")
print(model)

# Example text
input_text = "The quick brown fox jumps"

# Step 1: Tokenize the text
tokens = tokenizer.encode(input_text)
print("\nStep 1: Text to Token IDs")
print(f"Input text: '{input_text}'")
print(f"Token IDs: {tokens}")

# Step 2: Show what each token represents
print("\nStep 2: What each token represents:")
for token_id in tokens:
    token_word = tokenizer.decode([token_id])
    print(f"Token ID {token_id} = '{token_word}'")

# Step 3: Prepare input for model (add batch dimension and convert to tensor)
input_ids = torch.tensor([tokens])
print("\nStep 3: Input shape for model:", input_ids.shape)

# Step 4: Generate prediction
with torch.no_grad():  # We don't need gradients for inference
    outputs = model(input_ids)
    predictions = outputs.logits

# Step 5: Get the most likely next token
next_token_logits = predictions[0, -1, :]
next_token_id = torch.argmax(next_token_logits).item()
predicted_word = tokenizer.decode([next_token_id])

print("\nStep 5: Model's prediction")
print(f"Most likely next token ID: {next_token_id}")
print(f"Predicted next word: '{predicted_word}'")

# Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Print dataset structure and sizes
print("\nDataset splits and sizes:")
for split in dataset.keys():
    print(f"{split}: {len(dataset[split])} examples")

# Tokenize dataset
def tokenize_function(examples):
    # Filter out empty texts
    texts = [text for text in examples["text"] if len(text.strip()) > 0]
    
    if not texts:
        return {"input_ids": [], "attention_mask": []}
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors=None,
        return_attention_mask=True,
    )
    
    # Ensure all input_ids are non-empty and proper length
    valid_input_ids = [ids for ids in tokenized["input_ids"] if len(ids) > 0]
    valid_attention_masks = [mask for mask in tokenized["attention_mask"] if len(mask) > 0]
    
    return {
        "input_ids": valid_input_ids,
        "attention_mask": valid_attention_masks
    }

# Filter and tokenize the dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing the dataset"
)

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We want standard causal language modeling
)

# Training setup with more detailed configuration
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_steps=500,
    do_eval=True,
    prediction_loss_only=True,
    remove_unused_columns=True,
    logging_steps=100,
    # Add gradient clipping to prevent exploding gradients
    max_grad_norm=1.0,
    # Add warmup steps
    warmup_steps=500
)

# Filter out empty examples from datasets
def filter_empty_examples(example):
    return len(example["input_ids"]) > 0

for split in tokenized_datasets.keys():
    tokenized_datasets[split] = tokenized_datasets[split].filter(
        filter_empty_examples,
        desc=f"Filtering empty examples from {split}"
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# Train the model
print("\nStarting training...")
train_results = trainer.train()

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

# Print results
print("\nTraining results:", train_results)
print("\nTest results:", test_results)

# Save model
model.save_pretrained("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")