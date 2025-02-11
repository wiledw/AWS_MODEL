from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, model, tokenizer, max_length=100):
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=True, padding=True)
    
    # Generate text
    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    # Load the fine-tuned model and tokenizer
    model_path = "./gpt2_finetuned"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # Set up padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Set the model to evaluation mode
    model.eval()
    
    # Test prompts
    test_prompts = [
        "The history of artificial intelligence",
        "In the field of quantum mechanics,",
        "The Renaissance period was characterized by",
        "Climate change has led to",
        "The development of modern computing began"
    ]
    
    print("\nGenerating text with fine-tuned model:\n")
    print("-" * 50)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_text(prompt, model, tokenizer)
        print(f"\nGenerated text:\n{generated}")
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main() 