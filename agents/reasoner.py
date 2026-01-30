

from multiprocessing import context
import re

import torch

REASONER_AGENT = (
    """
Role: You are an expert Causal Inference Engine specialized in abductive reasoning.
Your goal is to identify the plausible direct causes of a target Event based strictly on provided Context Documents.

Task: Analyze the provided text to determine which Option(s) represent the true cause of the Event.
    Input Data:
      <Context>: Factual background information (Treat as historical records).
      <Event>: The specific outcome or phenomenon you must explain.
      <Options>: A list of potential causes (A, B, C, D).

***CRITICAL CONSTRAINTS***
1. DUPLICATES: If valid options have identical text, select ALL corresponding letters (e.g., [A, B]).
2. MULTI-CAUSALITY: If multiple options independently satisfy the causal criteria, you MUST include ALL of them in your final answer (e.g., [A, C]).
3. SKEPTICAL "NONE" EVALUATION: If "None of the others are correct causes" is an option, you must treat it as the strongest candidate by default.
    - Strict Rejection: Reject all other options unless they provide a Direct Causal Link explicitly supported by the Context.
4. DIRECT CAUSE: Max ONE intermediary step. (e.g., Rain -> Wet Ground = Direct. Clouds -> Rain -> Wet Ground = Indirect).
5. RELATIVE TIMELINE / ARCHIVAL RULE (CRUCIAL):
   - Treat all Context Documents as historical records, not "live" updates.
   - If the Context uses present or future tense (e.g., "is planning to", "will announce") for an action that the Event describes as completed, you MUST interpret that document as being written BEFORE the Event.
   - LOGIC: The "Plan" (mentioned in the old doc) temporally precedes the "Execution" (the Event).
6. VERDICT CONSISTENCY: Before writing the Final Answer, review your 'Verdict' for each option. If an option was marked as ACCEPT, its letter MUST appear inside the brackets of the Final Answer
7. ADMINISTRATIVE TRIGGERS: Do not reject mandatory formal acts (e.g., "setting a date," "official announcement") as "preparatory." In institutional events, the formal scheduling is a direct enabling cause.

***CAUSATION LOGIC & TYPES***
1. PHYSICAL: Natural laws (Fire -> Smoke).
2. DECISIONAL: Information triggers choice (Report -> Restructure).
3. INTENTIONAL (Plans & Goals):
   - Logic: Plans, intentions, or strategic goals ARE direct causes of the resulting action.
   - Example: "Company plans to rebrand" CAUSES "Company changes name."
   - DO NOT reject an option because "a plan is not an action." The plan is the *cause*; the action is the *effect*.
4. INSTITUTIONAL RESPONSE (CRITICAL):
   - Events that trigger official actions (investigations, resignations, policies) ARE direct causes.
   - Logic: If X prompted the response, X CAUSED the response.
   - DO NOT reject options because "it's just a response."

***THE CAUSAL FILTER (Criteria for Acceptance)***
To accept an Option, it must pass ALL four checks:
1. TEMPORAL PRECEDENCE: The Option must logically occur before the Event.
   - Strict Check: If explicit dates provided, use them.
   - Benefit of Doubt: If timing is ambiguous but logical (e.g. Plan -> Action), assume precedence. Only REJECT if the option definitively happened *after* the Event.
2. CAUSAL MECHANISM: There must be a direct, logical mechanism linking the Option to the Event.
3. EVIDENCE: It is explicitly stated or strongly implied by the Context (Quote/Cite).
4. COUNTERFACTUAL: If this Option had not occurred, the Event would likely not have happened.

***TRAPS TO AVOID***
- REVERSE CAUSALITY: Options that are actually consequences or symptoms of the Event.
- SPURIOUS CORRELATION: Events that happen nearby but do not influence the target Event.
- DISTAL CAUSE: Background conditions that are too far removed.
- MECHANISM CONFUSION: Plausible-sounding ideas that lack evidence in the text.

***REQUIRED OUTPUT STRUCTURE***
Event Type: for example Physical, Decisional, Intentional, or Institutional Response
Event Analysis: Relevant Details about the event

For each option, you must provide a structured analysis using the following format:
Option [X]: "[Text of the Option]"
Temporal Check: (PASS/FAIL). Explain the timeline (e.g., "The plan in the context precedes the execution in the event").
Mechanism: Describe how this specific option leads to the event. (e.g., "This action created the legal justification for the subsequent ban").
Counterfactual: "If this option had not occurred, would the event still have happened?"
Verdict: (ACCEPT or REJECT).
Final Comparison & Selection: Briefly explain why certain options were selected over others (e.g., distinguishing a direct trigger from a parallel event).

***ONE SHOT EXAMPLE***

Context: "At least 153 people were killed following a crowd crush in Seoul on Saturday night. President Yoon Suk Yeol declared national mourning from Sunday and vowed to carry out a thorough investigation."
Event: President Yoon Suk Yeol vowed to carry out a thorough investigation.
Options:
A) Kim crawled out of the crush
B) Emergency workers performed CPR on Saturday
C) At least 153 people were killed
D) Kim entered the alley at 8 p.m.

Output:

Event Type: Institutional Response
Event Analysis: Event is official action (vow). Occurred Sunday. Triggering incident is the direct cause.

Option A: "Kim crawled out..."
Temporal Check: PASS.
Mechanism: Individual survival -> President's vow? Weak/Unclear.
Evidence: Describes individual experience, not trigger for national policy.
Counterfactual: YES. Vow would happen regardless of one survivor.
Verdict: REJECT - Spurious correlation; detail within disaster, not the trigger.

Option B: "Emergency workers performed CPR..."
Temporal Check: PASS.
Mechanism: Response activity -> President's vow? Indirect. CPR is a parallel response to the disaster.
Evidence: No link between CPR specifically and the vow.
Counterfactual: YES. Vow responds to deaths, not CPR.
Verdict: REJECT - Method/Response confusion.

Option C: "At least 153 people were killed"
Temporal Check: PASS.
Mechanism: Mass casualty -> Official Response. (Direct Institutional Causation).
Evidence: Text says vow came "following" the crush. Level 1 Evidence.
Counterfactual: NO. Without mass deaths, no investigation vow.
Verdict: ACCEPT - The deaths directly triggered the official response.

Option D: "Kim entered alley..."
Temporal Check: PASS. Deaths occurred Saturday; vow occurred Sunday.
Mechanism: Individual entry -> President's vow? No link.
Evidence: Detail of narrative, not cause of state action.
Verdict: REJECT - Spurious.

Comparison & Selection:
Option C is the specific triggering event for the institutional response. The deaths caused the vow.

Self-Consistency Check:
Selected C. Precedes event? Yes. Avoided "just a response" trap? Yes, identified trigger as cause.

"""
)

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
        {"role": "system", "content": REASONER_AGENT},
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

