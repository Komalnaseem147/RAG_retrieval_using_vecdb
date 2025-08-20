import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load environment variable for Hugging Face token (set in .env)
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Choose your model (change this to LLaMA, Mistral, Falcon, etc.)
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Quantization config (4-bit for GPU memory efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model
print("⏳ Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

print("✅ Model and tokenizer loaded!")


# -------------------------
# Generate response function
# -------------------------
def generate_response(prompt, max_new_tokens=300, temperature=0.7, top_p=0.9):
    """
    Generates a response from the LLM given a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Sometimes model repeats prompt, so strip it out
    return response.replace(prompt, "").strip()
