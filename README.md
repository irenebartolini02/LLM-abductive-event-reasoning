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
 
- [**Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**](https://arxiv.org/pdf/2201.11903)
    - We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning  

- [**Instruction Tuning for Large Language Models: A Survey**](https://arxiv.org/pdf/2308.10792)
    - This paper surveys research works in the quickly advancing field of instruction tuning (IT), which can also be referred to as supervised fine-tuning (SFT)
- [**Active Prompting with Chain-of-Thought for Large Language Models**](https://aclanthology.org/2024.acl-long.73.pdf)
    - This paper proposes a new method, Active-Prompt, to adapt LLMs to different tasks with task-specific example prompts (annotated with human-designed CoT reasoning). For this purpose, we propose a solution to the key problem of determining which questions are the most important and helpful to annotate from a pool of task-specific queries.

- [**GoT: Effective Graph-of-Thought Reasoning in Language Models**](https://aclanthology.org/2024.findings-naacl.183.pdf)
    -  we propose Graph-of-Thought (GoT) reasoning, which models human thought processes not only as a chain but also as a graph. By representing thought units as nodes and connections between them as edges, our approach captures the non-sequential nature of human thinking and allows for a more realistic modeling of thought processes
 
- [**Tree of Thoughts: Deliberate Problem Solving with Large Language Models**](https://arxiv.org/pdf/2305.10601)
    - we introduce a new framework for language model inference, “Tree of Thoughts” (ToT), which generalizes over the popular “Chain of Thought” approach to prompting language models, and enables exploration over coherent units of text (“thoughts”) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices.
 
## Multi-Agentic RAG

- [**Retrieval-Augmented Generation for Large Language Models: A Survey**](https://arxiv.org/pdf/2312.10997)
      - This comprehensive review paper offers a detailed
examination of the progression of RAG paradigms, encompassing
the Naive RAG, the Advanced RAG, and the Modular RAG.
It meticulously scrutinizes the tripartite foundation of RAG
frameworks, which includes the retrieval, the generation and the
augmentation techniques. The paper highlights the state-of-theart technologies embedded in each of these critical components,
providing a profound understanding of the advancements in RAG
systems.

- [**Large Language Model based Multi-Agents: A Survey of Progress and Challenges**](https://arxiv.org/pdf/2402.01680)
    - we present this
survey to offer an in-depth discussion on the essential aspects of multi-agent systems based on LLMs,
as well as the challenges.

- [**Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration**](https://arxiv.org/pdf/2510.16645)
     - This paper introduces the
Multi-Agent Collaboration Framework for Diverse Thinking Modes
(DiMo), which enhances both performance and interpretability by
simulating a structured debate among four specialized LLM agents.
Each agent embodies a distinct reasoning paradigm, allowing the
framework to collaboratively explore diverse cognitive approaches.
Through iterative debate, agents challenge and refine initial responses,
yielding more robust conclusions and an explicit, auditable reasoning chain

- [**Reconstructing Context**](https://arxiv.org/pdf/2504.19754v1)
    -  how can vast volumes of external knowledge be managed effectively within the input constraints of LLMs? Traditional methods address this by chunking external documents into smaller, fixedsize segments. While this approach alleviates input limitations, it often
fragments context, resulting in incomplete retrieval and diminished coherence in generation. To overcome these shortcomings, two advanced
techniques—late chunking and contextual retrieval—have been introduced, both aiming to preserve global context. 

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

**Total Questions: 400**

**Results**
|         | Number | Percentage |
|---------|--------|------------|
| Correct | 118    | 29.5%      |
| Partial | 134    | 33.5%      |
| Wrong   | 148    | 37.0%      |
| Score   | 185    | 46.25%    |


## ⁠Hermes 3 - Llama 3.1 8B
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
| Correct | 153    | 38.25%      |
| Partial | 110     | 27.50%      |
| Wrong   | 137    | 34.52%      |
| Score   | 208.0  | 52.00%     |

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
| Correct | 133    | 33.33%     |
| Partial | 97     | 24.31%     |
| Wrong   | 169    | 42.35%      |
| Score   | 181.5  | 45.49%    |

#### **Version C**

**Context building strategy**: Concatenating "--- Document {i} ---\nTitle: {doc.get('title')}\nText: {clean_text}\n\n until a maximum number of characters (6000)

**Total Questions: 400**

**Results**

|         | Number | Percentage |
|---------|--------|------------|
| Correct | 142    | 35.5%     |
| Partial | 68     | 17.0%      |
| Wrong   | 190    | 47.5%     |
| Score   | 176  | 44.0%      |

## ⁠Qwen 2.5 7B Instruct
caracteristics: autoregressive "decoder-only" causal masking
parameters: 7.6 billion
context windows: 128.000 tokens (~7.3 GB in KV Cache VRAM) 

quantization: precision INT4 (0.5 byte) ~3.8 GB

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
