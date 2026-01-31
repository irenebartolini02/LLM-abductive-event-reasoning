import re
from typing import Dict, List, Optional
import re
import ast
import re
from typing import Dict, List
import torch

from prompts.template import SEARCH_AGENT


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

import re
from typing import Dict, List, Optional

def search_agent_parser(response_raw: str) -> Dict:
    """
    Parse a search agent response into a simple dictionary.
    
    Args:
        response_raw: Raw response text from the search agent
        
    Returns:
        Dictionary with keys:
        - reasoning: str - The complete reasoning text
        - gap_analysis: Optional[str] - Gap analysis if present
        - search_queries: List[str] - Extracted search queries
        - is_sufficient: bool - Whether context is sufficient
    """
    
    # Extract gap analysis
    gap_match = re.search(r'Gap Analysis:\s*(.+?)(?=\n\s*Action:|\Z)', response_raw, re.DOTALL)
    gap_analysis = gap_match.group(1).strip() if gap_match else None
    
    # Extract action
    action_match = re.search(r'Action:\s*(.+?)$', response_raw, re.DOTALL)
    action = action_match.group(1).strip() if action_match else None
    
    # Extract search queries from action
    search_queries = []
    is_sufficient = False
    
    if action:
        # Check if action is SUFFICIENT
        is_sufficient = "SUFFICIENT" in action.upper()
        
        # Extract queries in Search["query1", "query2"] format
        search_pattern = r'Search\[([^\]]+)\]'
        search_match = re.search(search_pattern, action)
        if search_match:
            queries_text = search_match.group(1)
            # Split by quotes (both single and double) and filter
            query_pattern = r'["\']([^"\']+)["\']'
            search_queries = re.findall(query_pattern, queries_text)
    
    return {
        "reasoning": response_raw,
        "gap_analysis": gap_analysis,
        "search_queries": search_queries,
        "is_sufficient": is_sufficient
    }
