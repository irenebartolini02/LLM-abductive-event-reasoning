

from multiprocessing import context
import re

import torch

from prompts.template import CAUSAL_AGENT



def reasoner_agent(model, tokenizer, context,question):
    prompt = reasoner_prompt(tokenizer, context,question)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. Generation (Greedy decoding for reproducibility)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3000,
            do_sample=False,      # Determinism
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. Decoding e Cleaning
    # Cutting the input prompt -> the response of the model also contains the input prompt we provided
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    #print(f"RAW RESPONSE: \n{raw_response}")
    parsed = reasoner_agent_parser(raw_response)

    return parsed

def reasoner_prompt(tokenizer, context,question):
    user_content = f"""
    
        <Context>:
        {context}
        
        <Event>: "{question['target_event']}"
        
        <Options>:
        A) {question['option_A']}
        B) {question['option_B']}
        C) {question['option_C']}
        D) {question['option_D']}
        
        Action:"""
    
    messages = [
        # The System Role: Sets the persona and rules
        {"role": "system", "content": CAUSAL_AGENT},
        # The User Role: Provides the specific instance data
        {"role": "user", "content": user_content}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt




def reasoner_agent_parser(text):
    """
    Optimized parser to extract the response even if preceded by 'Option' 
    or formatted in a non-standard way.
    """
    # 1. Look for the Final Answer line
    # Supports: "Final Answer: [C]", "Final Answer: Option C", "Final Answer: C"
    answer_match = re.search(r'Final\s*Answer\s*:?\s*(.*)', text, re.IGNORECASE)
    
    answer = []
    reasoning = text.strip()

    if answer_match:
        answer_line = answer_match.group(1).strip()
        # 2. Extract only single isolated letters (A, B, C or D)
        # This regex searches for single uppercase letters that are not part of other words
        letters = re.findall(r'\b([A-D])\b', answer_line.upper())
        
        if letters:
            answer = sorted(list(set(letters))) # Remove duplicates and sort
        
        # 3. The reasoning is everything that comes before the found line
        reasoning = text[:answer_match.start()].strip()

    return {
        "reasoning": reasoning,
        "answer": answer
    }

