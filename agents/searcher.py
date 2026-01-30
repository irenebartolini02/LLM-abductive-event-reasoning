import re
from typing import Dict, List, Optional
import re
import ast

import re
from typing import Dict, List

import torch


SEARCH_AGENT = (
    """
### Instruction
You are an **Evidence Evaluator** for causal inference. 
Assess if <Context> provides enough evidence to evaluate <Options> as causes of <Event>. 
Generate targeted queries for missing information.

**Important:** Multiple options can be correct causes.

### Input
<Context>: Available information
<Event>: Outcome to explain
<Options>: Potential causes (A, B, C, D)

### The Decision Logic
1.  **Relevance Filter (Pre-Computation):**
    - **Discard** options that are clearly unrelated, illogical, or describe irrelevant details (e.g., specific times/names that don't change the outcome) relative to the <Event>.
    - **Keep** options that represent plausible causes.
2.  **Verification (The Search Trigger):**
    - Check the **Remaining (Plausible) Options** against the <Context>.
    - If a *Plausible* Option is missing evidence -> **Action: `Search[...]`**.
3.  **Sufficiency:** - Output `SUFFICIENT` if all **Plausible** options are confirmed or refuted. 
       In order to be SUFFICIENT the context should provide: 
       - TEMPORAL EVIDENCE: The evidence that th Option happend before or after the Event.
       - CAUSAL MECHANISM: logical mechanism linking the Option to the Event.
4.  **Query Formulation (Fact-Oriented)**: - Do NOT generate questions (e.g., "Was there a shooting?").
    Generate declarative factual strings or search keywords (e.g., "Ashli Babbitt shooting incident Capitol", "David Cameron referendum pledge 2013").
    Focus on entities, dates, and specific events mentioned in the options.
    
(Note: You do NOT need to verify the discarded "irrelevant" options to declare sufficiency).

### Output Format
Discarded options: [Letters]
Proposal options: [Letters]
Reasoning: [Brief analysis of which options are supported vs. missing info]
Action: SUFFICIENT or SEARCH
Queries: ['query1', 'query2', 'quer'] 


"""
)


def search_agent(model, tokenizer , context,question):
    prompt = search_prompt(tokenizer, context, question)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. Generation (Greedy decoding for reproducibility)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,      # Determinism
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. Decoding e Cleaning
    # Cutting the input prompt -> the response of the model also contains the input prompt we provided
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    parsed = search_agent_parser(raw_response)

    return parsed
    
    
def search_prompt(tokenizer, context, question):
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
        {"role": "system", "content": SEARCH_AGENT},
        # The User Role: Provides the specific instance data
        {"role": "user", "content": user_content}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def search_agent_parser(response_raw: str) -> Dict:
    # 1. Extract main blocks with Regex (more flexible on line breaks)
    discarded_match = re.search(r"Discarded options:\s*(.*)", response_raw, re.IGNORECASE)
    proposal_match = re.search(r"Proposal options:\s*(.*)", response_raw, re.IGNORECASE)
    # The reasoning stops when it encounters Action: or the end of the text
    reasoning_match = re.search(r"Reasoning:\s*(.*?)(?=Action:|$)", response_raw, re.IGNORECASE | re.DOTALL)
    action_match = re.search(r"Action:\s*(\w+)", response_raw, re.IGNORECASE)
    # Captures everything that follows "Queries:" until the end or the next block
    query_block_match = re.search(r"Queries:\s*(.*)", response_raw, re.IGNORECASE | re.DOTALL)

    data = {
        "discarded": [],
        "proposal": [],
        "reasoning": "",
        "is_sufficient": True,
        "queries": []
    }

    # 2. Parse Letters (A-D)
    if discarded_match:
        data["discarded"] = re.findall(r'[A-D]', discarded_match.group(1).upper())
    
    if proposal_match:
        data["proposal"] = re.findall(r'[A-D]', proposal_match.group(1).upper())

    # 3. Parse Reasoning
    if reasoning_match:
        data["reasoning"] = reasoning_match.group(1).strip()

    # 4. Parse Action
    if action_match:
        action_val = action_match.group(1).upper()
        data["is_sufficient"] = "SEARCH" not in action_val

    # 5. Estrazione Query Robusta
    if not data["is_sufficient"] and query_block_match:
        query_text = query_block_match.group(1).strip()
        
        # --- Strategia A: Cerca il contenuto tra parentesi quadre [ ... ] ---
        list_match = re.search(r"\[(.*?)\]", query_text, re.DOTALL)
        if list_match:
            content = list_match.group(1)
            # Estraiamo tutto ciò che è tra apici (singoli o doppi)
            # Questo evita i problemi dello split basato su virgole
            data["queries"] = [q.strip() for q in re.findall(r"['\"](.*?)['\"]", content) if q.strip()]
        
        # --- Strategia B (Fallback): Se A fallisce (es. elenco puntato o testo libero) ---
        if not data["queries"]:
            # Prende tutte le stringhe tra virgolette nel blocco, ovunque siano
            quoted_items = re.findall(r'["\'](.*?)["\']', query_text)
            # Filtro per lunghezza per evitare di prendere singoli caratteri casuali
            data["queries"] = [q.strip() for q in quoted_items if len(q.strip()) > 3]
            
        # --- Strategia C (Extrema Ratio): Se ancora vuoto, split per linee ---
        if not data["queries"]:
            lines = [l.strip(" '\"-•*") for l in query_text.split('\n') if len(l.strip()) > 5]
            if lines:
                data["queries"] = lines

    return data


def format_parsed_response(parsed: Dict) -> str:
    """Format a parsed response dictionary for readable display"""
    lines = []
    lines.append("=" * 60)
    lines.append("PARSED SEARCH AGENT RESPONSE")
    lines.append("=" * 60)
    
    lines.append(f"\nIs Sufficient: {parsed['is_sufficient']}")
    
    if parsed['gap_analysis']:
        lines.append(f"\nGap Analysis:\n{parsed['gap_analysis']}")
    
    if parsed['search_queries']:
        lines.append(f"\nSearch Queries ({len(parsed['search_queries'])}):")
        for i, q in enumerate(parsed['search_queries'], 1):
            lines.append(f"  {i}. {q}")
    
    lines.append(f"\nFull Reasoning:\n{parsed['reasoning'][:500]}...")
    lines.append("=" * 60)
    
    return "\n".join(lines)



