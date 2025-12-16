# Related works

## Fondamenta: Definizione del Task
Prime formulazioni del Abductive Reasoning in NLP
-  [**Abductive Commonsense Reasoning**](https://arxiv.org/pdf/1908.05739):
    - È il gold standard che ha definito il formato "Osservazione 1 -> [Ipotesi] -> Osservazione 2" come un problema a scelta multipla.
    - Introduce il dataset ART e dimostra che i modelli generativi (GPT-2 all'epoca) faticano a ragionare in modo non monotonico (abduttivo) rispetto a quello deduttivo.
- [**e-CARE: a New Dataset for Exploring Explainable Causal Reasoning**](https://arxiv.org/pdf/2205.05849):
    - Molto vicino al gruppo di ricerca che organizza il task (CASIA). Introduce un dataset enorme (>20k) di ragionamento causale con spiegazioni.
    - Sposta il focus dalla semplice scelta multipla alla comprensione della causa profonda, un ponte perfetto verso l'Abductive Event Reasoning.

## Metodologia: Event Graphs & Causal Knowledge
Lavori degli stessi autori:
- [**Learning Event Graph Knowledge for Abductive Reasoning**](https://aclanthology.org/2021.acl-long.403.pdf):
    - Propone l'idea che per risolvere questi problemi non basta il testo grezzo, ma serve costruire un "grafo degli eventi" per capire le connessioni causali. È l'approccio "simbolico/strutturato" che spesso si contrappone o integra agli LLM puri.
- **Event Causality Identification with Causal Graph (generico)**:
    - Questo non è un paper, cercare i lavori di Pengfei Cao o Yubo Chen su "Event Causality Identification" (spesso pubblicati in ACL/EMNLP 2021-2023). Dimostrano che l'identificazione della causa diretta (il cuore del task AER) è un problema di estrazione di relazioni complesse.

## Stato dell'Arte (SOTA) & LLM Era (2024-2025)
Dimostrazione che il lavoro è aggiornato: per discutere di come GPT-4, Llama 3 o metodi come CoT (Chain-of-Thought) performano oggi su questo task.
- [**Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning**](https://arxiv.org/pdf/2506.07106):
    - È freschissimo (SOTA 2025). Propone un framework multi-agente dove uno specifico agente si occupa di "Abduction". Mostra che separare i tipi di ragionamento aiuta gli LLM a non allucinare.
    - Dimostra che gli LLM "lisci" faticano ancora nell'abduzione pura rispetto alla deduzione.

- [**UNcommonsense Reasoning: Abductive Reasoning about Uncommon Situations**](https://arxiv.org/pdf/2311.08469):
    - Valuta GPT-4 su scenari abduttivi "non comuni". È perfetto per giustificare perché serve un benchmark su "Eventi Reali" (Real-world events) come quello di SemEval, dove il senso comune standard potrebbe non bastare.

- [**Tackling LLM Hallucination with Abductive Reasoning**](https://www.preprints.org/manuscript/202511.1688):
    - Collega il problema dell'AER alle allucinazioni. Puoi usarlo per dire: "Valutare l'Abductive Reasoning è critico perché il fallimento in questo task è spesso la causa delle allucinazioni negli scenari RAG".


# Baselines Causal Reasoning Evaluation
## Gemma 2 9B Instruct

### **SYSTEM_PROMPT**: "You are solving SemEval 2026 Task 12: Abductive Event Reasoning. 
    You are solving SemEval 2026 Task 12: Abductive Event Reasoning. 
    Given an event, context documents, and four options (A–D),
    choose which option(s) are the most plausible direct cause of the event. 
    Respond ONLY with the letters of all correct options, 
    separated by commas (e.g. 'A', 'A,B', or 'D'). 
    Do not output any explanations.

#### **Version A:**

**Context building strategy**: Concatenating "--- Document {i} ---\nTitle: {doc.get('title')}\nText: {clean_text}\n\n until a maximum number of characters (6000)

**Total Questions: 400**

**Results**

|         | Number | Percentage |
|---------|--------|------------|
| Correct | 180    | 45.0%      |
| Partial | 90     | 22.5%      |
| Wrong   | 130    | 32.5%      |
| Score   | 225.0  | 56.25%     |

#### **Version B:**

**Context building strategy**: Concatenating the first MAX_DOCS_PER_TOPIC Documents 
"Doc {i}:\n
Title: {doc.get('title')}\n
Snippet: {doc.get('snippet')}\n  
Content: {content}\n"

the document content is truncated at MAX_CHARS_PER_DOC size

content = content[:MAX_CHARS_PER_DOC] + "..."

**Results**
|         | Number | Percentage |
|---------|--------|------------|
| Correct | 119    | 29.75%     |
| Partial | 93     | 23.25%     |
| Wrong   | 188    | 47.0%      |
| Score   | 165.5  | 41.375%    |




## ⁠Llama 3.1 8B Instruct (Meta)

## ⁠Solar 10.7B Instruct v1.0
### **SYSTEM_PROMPT**: "You are solving SemEval 2026 Task 12: Abductive Event Reasoning. 
    You are solving SemEval 2026 Task 12: Abductive Event Reasoning. 
    Given an event, context documents, and four options (A–D),
    choose which option(s) are the most plausible direct cause of the event. 
    Respond ONLY with the letters of all correct options, 
    separated by commas (e.g. 'A', 'A,B', or 'D'). 
    Do not output any explanations.

#### **Version A:**

**Context building strategy**: Concatenating "--- Document {i} ---\nTitle: {doc.get('title')}\nText: {clean_text}\n\n until a maximum number of characters (6000)

**Total Questions: 400**

**Results**

|         | Number | Percentage |
|---------|--------|------------|
| Correct | 147    | 36.75%     |
| Partial | 75     | 18.75%     |
| Wrong   | 178    | 44.5%      |
| Score   | 184.5  | 46.125%    |

#### **Version B:**

**Context building strategy**: Concatenating the first MAX_DOCS_PER_TOPIC Documents 
"Doc {i}:\n
Title: {doc.get('title')}\n
Snippet: {doc.get('snippet')}\n  
Content: {content}\n"

the document content is truncated at MAX_CHARS_PER_DOC size

content = content[:MAX_CHARS_PER_DOC] + "..."

Total questions: 400

**Results**
|         | Number | Percentage |
|---------|--------|------------|
| Correct | 147    | 36.75%     |
| Partial | 75     | 18.75%     |
| Wrong   | 178    | 44.5%      |
| Score   | 184.5  | 46.125%    |


## ⁠Hermes 3 - Llama 3.1 8B

## ⁠Qwen 2.5 7B Instruct

### **SYSTEM_PROMPT**: 
    You are solving SemEval 2026 Task 12: Abductive Event Reasoning. 
    Given an event, context documents, and four options (A–D),
    choose which option(s) are the most plausible direct cause of the event. 
    Respond ONLY with the letters of all correct options, 
    separated by commas (e.g. 'A', 'A,B', or 'D'). 
    Do not output any explanations.

#### **Version A**

**Context building strategy**: None

**Total Questions: 400**

**Results**

|         | Number | Percentage |
|---------|--------|------------|
| Correct | 195    | 48.75%     |
| Partial | 94     | 23.5%      |
| Wrong   | 111    | 27.750000000000004% |
| Score   | 242.0  | 60.5%      |

#### **Version B**

**Context building strategy**: Concatenating the first MAX_DOCS_PER_TOPIC Documents 
"Doc {i}:\n
Title: {doc.get('title')}\n
Snippet: {doc.get('snippet')}\n  
Content: {content}\n"

the document content is truncated at MAX_CHARS_PER_DOC size

content = content[:MAX_CHARS_PER_DOC] + "..."

**Total Questions: 400**

**Results**

|         | Number | Percentage |
|---------|--------|------------|
| Correct | 185    | 46.25%     |
| Partial | 99     | 24.75%     |
| Wrong   | 116    | 28.999999999999996% |
| Score   | 234.5  | 58.62500000000001%  |

#### **Version C**

**Context building strategy**: Concatenating "--- Document {i} ---\nTitle: {doc.get('title')}\nText: {clean_text}\n\n until a maximum number of characters (6000)

**Total Questions: 400**

**Results**

|         | Number | Percentage |
|---------|--------|------------|
| Correct | 195    | 46.75%     |
| Partial | 94     | 29.5%      |
| Wrong   | 111    | 23.75%     |
| Score   | 242.0  | 61.5%      |
