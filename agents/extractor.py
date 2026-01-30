import torch


def extractor_prompt(reasoning_text):
    return f"""You are a Precise Output Formatter.
Your task is to extract the final answer from a causal reasoning analysis.

### INPUT REASONING:
{reasoning_text}

### INSTRUCTIONS:
1. Identify which options were marked as 'ACCEPT', 'PASS', or 'VERDICT: ACCEPT' in the reasoning.
2. If the reasoning was cut off but showed a clear preference for certain options, extract those.
3. Provide the final answer as a comma-separated list of letters inside square brackets.
4. Provide at least one LETTER as OUTPUT, never return the empty set []. example of output: Final Answer: ['A', 'B']
5. SKEPTICAL "NONE" EVALUATION: If "None of the others are correct causes" is an option, you must treat it as the strongest candidate by default.

Final Answer: [Letter(s)] """

def extractor_agent(model, tokenizer, reasoning_text):
    # Costruiamo il prompt usando il testo generato dal causal_agent
    prompt = extractor_prompt(reasoning_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50, 
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    
    full_response = "[" + raw_response
    
    return extractor_agent_parser(full_response)


import re

def extractor_agent_parser(text):
    """
    Estrae le lettere (A, B, C, D) all'interno di parentesi quadre.
    Esempio: "Final Answer: [A, C]" -> ['A', 'C']
    """
    # Cerca qualsiasi cosa dentro le parentesi quadre
    match = re.search(r'\[(.*?)\]', text)
    if match:
        content = match.group(1).upper()
        # Trova tutte le singole lettere A, B, C o D
        letters = re.findall(r'[A-D]', content)
        # Rimuove duplicati mantenendo l'ordine
        return sorted(list(set(letters)))
    
    # Fallback: se non ci sono parentesi, cerca lettere isolate nel testo
    return []
