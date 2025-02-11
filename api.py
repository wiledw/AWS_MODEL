from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import Optional

app = FastAPI(
    title="GPT-2 Text Generation API",
    description="API for generating text using fine-tuned GPT-2 on WikiText",
    version="1.0.0"
)

# Model loading
try:
    model = GPT2LMHeadModel.from_pretrained("./gpt2_finetuned")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_finetuned")
    
    # Set up padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Set model to evaluation mode
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str

@app.get("/")
def read_root():
    return {
        "message": "GPT-2 Text Generation API",
        "endpoints": {
            "/generate": "POST endpoint for text generation",
            "/health": "GET endpoint for health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    try:
        # Encode the input prompt
        inputs = tokenizer(
            request.prompt,
            return_tensors='pt',
            add_special_tokens=True,
            padding=True
        )
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=request.max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 