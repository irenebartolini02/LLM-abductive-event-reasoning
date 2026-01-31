import gc
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.output_utils import clean_response, calculate_score
from sentence_transformers import SentenceTransformer, CrossEncoder

def load_model( MODEL_NAME: str):
    """Load a Qwen model from the given model name."""
    # 1. Definition of the quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,              # Activate 4-bit loading
        bnb_4bit_quant_type="nf4",      # Use the type "nf4" (more precise)
        bnb_4bit_compute_dtype=torch.float16  # float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    # 2. Loading model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.eval()
    print("Model loaded on:", model.device)
    return model, tokenizer


def format_qwen_prompt(tokenizer, SYSTEM_PROMPT, question, context_text , max_total_chars=20_000):
  #1 Get the context
  if len(context_text) < max_total_chars:
    context_text = context_text[:max_total_chars]+ " ..."

  # 2. Prompt Composition
  system_content =SYSTEM_PROMPT

  # Cleaned user message
  user_content = f"""Event: "{question['target_event']}"

Context Documents:
{context_text}

Options:
A) {question['option_A']}
B) {question['option_B']}
C) {question['option_C']}
D) {question['option_D']}

Answer (letters only):"""

 # Official Qwen template with distinct roles
  messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  return prompt




embed_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'
reranker = CrossEncoder(reranker_model_name, device=device)