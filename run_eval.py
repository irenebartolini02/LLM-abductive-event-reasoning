import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


import sys
import argparse
import json
import torch
import gc
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from agents.searcher import search_agent  # Assuming you moved search_agent logic to agents/searcher.py
from agents.reasoner import reasoner_agent  # Assuming you moved causal_agent logic to agents/reasoner.py
from rag.RAGChain import RagChain, build_context, search_query
from utils.output_utils import calculate_score
from utils.model_utils import load_model
from utils.data_loader import load_jsonl, load_json, index_docs_by_topic


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Pipeline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the unzipped dataset folder")
    parser.add_argument("--output_file", type=str, default="results/eval_results.jsonl", help="Output path")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--use_debug", action="store_true", help = "Enable debug during eval loop")
    parser.add_argument("--search_steps", type=int, default=1, help="Number of iteration of the search agent" )
    parser.add_argument("--k_per_option", type=int, default=10)
    parser.add_argument("--k_final", type=int, default=5)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # --- Setup WandB ---
    if args.use_wandb:
        import wandb
        wandb.init(project="Agentic-Causal-Reasoning", name="Qwen 2.5/7B")

    # --- SILENCE LOGS ---
    import logging
    import transformers
  
    transformers.logging.set_verbosity_error()
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # --- Load Resources ---
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading Model: {model_name}")
    model, tokenizer = load_model(model_name)
    print("Model loaded successfully\n")
    
    # --- Load Data ---
    print(f"Loading Data from {args.dataset_path}...")
    # Adjust paths based on your actual dataset structure
    questions_path = os.path.join(args.dataset_path, "dev_data", "questions.jsonl")
    docs_path = os.path.join(args.dataset_path, "dev_data", "docs.json")

    questions = load_jsonl(questions_path)
    docs = load_json(docs_path)
    docs_by_topic = index_docs_by_topic(docs)
    print("Data loaded successfully\n")

    # --- Loading Embedding model ---
    embedder_name = 'BAAI/bge-base-en-v1.5'
    print(f"Loading Embedder: {embedder_name}...")
    embed_model = SentenceTransformer(embedder_name, device=device)
    print("Embedder loaded succesfully\n")
    
    # --- Loading Reranker model ---
    reranker_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    print(f"Loading Reranker: {reranker_name}...")
    reranker_model = CrossEncoder(reranker_name, device=device)
    print("Reranker loaded successfully\n")

    # --- Initialize RAG Engine ---
    print("Initializing RAG Chain...")
    rag = RagChain(
        embedding_model=embed_model,
        reranker=reranker_model,
        k_per_option=args.k_per_option,
        k_final=args.k_final,
        chunk_size=800,      
        chunk_overlap=150    
    )

    print("Indexing Documents...")
    rag.index_documents(docs_by_topic)
    
    # --- Checkpoint Restoration Logic ---
    total_score = 0
    count = 0
    processed_uuids = set()
    results = []
    
    running_stats = {"correct": 0, "partial": 0, "wrong": 0}

    if args.use_wandb:
        # --- RESTORE FROM CHECKPOINT FOR WANDB LOGGING ---
        if os.path.exists(args.output_file):
            print(f"Recupero checkpoint da {args.output_file}...")
            with open(args.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        results.append(data)
                        processed_uuids.add(data['uuid'])
                        
                        # Aggiorniamo le statistiche
                        score = data['score']
                        total_score += score
                        count += 1
                        
                        if score == 1.0: running_stats["correct"] += 1
                        elif score > 0: running_stats["partial"] += 1
                        else: running_stats["wrong"] += 1
                    except json.JSONDecodeError:
                        continue
            print(f"Restored {count} items.")
                        
            # This reconstructs the graph for the historical data
            if args.use_wandb and count > 0:
                wandb.log({
                    "progress": count,
                    "global/correct": running_stats["correct"] / count,
                    "global/partial": running_stats["partial"] / count,
                    "global/error_rate": running_stats["wrong"] / count,
                    "global/avg_score": total_score / count,
                })
    else:
        print("Nessun checkpoint trovato. Inizio da zero.")




    # --- Evaluation Loop ---
    print("Starting Evaluation Loop...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    for entry in tqdm(questions):
        question_uuid = entry['id']
        
        # --- SKIP LOGIC ---

        # If UUID is in the processed sed, we skip the iteration
        if question_uuid in processed_uuids:
            continue

        # ------------------

        try:
            topic_id = entry['topic_id']
            question_uuid = entry['id']
            golden_ans = entry['golden_answer']
            target_event = entry['target_event']
            options = [entry["option_A"],entry["option_B"],entry["option_C"],entry["option_D"]]

            # RAG: build context
            initial_context = build_context(rag,topic_id, target_event, options)


            # --- SEARCH AGENT ---
            search_iteration = 0
            is_sufficient = False
            total_queries = []

            while search_iteration < args.search_steps and not is_sufficient:    
                
                search_response_parsed = search_agent(model, tokenizer, initial_context, entry)
                search_response_parsed["iteration"] = search_iteration
                is_sufficient = search_response_parsed['is_sufficient']
    
                if not is_sufficient:
                    queries = search_response_parsed['search_queries']
        
                    if queries:
                        for query in queries: 
                            if query in total_queries:
                                is_sufficient = True
                            else:
                                single_query_context = search_query(rag,query,topic_id, 2 + len(total_queries) )
                                total_queries.append(query)
                                initial_context += "\n\n" + single_query_context
            
                    search_iteration += 1

            final_context = initial_context
            search_response_parsed ['context'] = final_context
        
            causal_response_parsed = reasoner_agent(model,tokenizer,final_context,entry)

            reasoning = causal_response_parsed['reasoning']
            answer = causal_response_parsed['answer']

            # --- SCORING ---
            pred_set = set(answer)
            score = calculate_score(pred_set,golden_ans)

            result_item = {
                "uuid": question_uuid,
                "topic_id": topic_id,
                "target_event": entry["target_event"],
                "options": options,
                "golden_raw": golden_ans,
                "prediction_set": list(pred_set),
                "reasoning_response_raw": reasoning,
                "score": score,
                "search_queries": total_queries,
                "total_iteration_quries": search_iteration,
                "context": final_context
            }

            # --- DEBUG PRINT ---
            if args.use_debug:
                print("\n" + "="*100)
                print(f"DEBUG REPORT | UUID: {question_uuid}")
                print(f"TARGET EVENT: {entry['target_event']}")
                print("-" * 100)
                
                print("OPTIONS:")
                # Automatically maps 0->A, 1->B, 2->C, etc.
                for i, opt in enumerate(options):
                    print(f"  [{chr(65+i)}]: {opt}")
            
                print("-" * 50)
                print(f"GOLDEN ANSWER   : {golden_ans}")
                print(f"PREDICTED SET   : {list(pred_set)}")
                print(f"SCORE           : {score}")
                
                print("-" * 50)
                print("QUERIES SEARCHED:")
                if total_queries:
                    for i, q in enumerate(total_queries, 1):
                        print(f"  {i}. {q}")
                else:
                    print("  (None)")
                    
                print("="*100 + "\n")

            
            # --- SAVE CHECKPOINT ---
            with open(args.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_item) + "\n")

            
            # Updating stats
            results.append(result_item)
            processed_uuids.add(question_uuid)
            total_score += score
            count += 1

            if score == 1.0: running_stats["correct"] += 1
            elif score > 0: running_stats["partial"] += 1
            else: running_stats["wrong"] += 1

            if args.use_debug:
                wandb.log({
                    "progress": count,
                    "global/correct": running_stats["correct"] / count,
                    "global/partial": running_stats["partial"] / count,
                    "global/error_rate": running_stats["wrong"] / count,
                    "global/avg_score": total_score / count,
                })        
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"UUID:{entry['id']} process skipped due to OOM")
                # forcely free the cache
                del inputs
                torch.cuda.empty_cache()
                gc.collect()
                errors += 1
                continue
            else:
                print(f"Errore generico: {e}")
                continue

if __name__ == "__main__":
    main()