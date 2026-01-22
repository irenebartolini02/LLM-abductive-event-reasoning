import os
import pickle
import torch
import re
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from matplotlib import pyplot as plt
from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from tqdm import tqdm
from transformers import pipeline
import networkx as nx
from typing import List, Set, Dict, Tuple
import re
import json
from rank_bm25 import BM25Okapi
from difflib import SequenceMatcher

class CausalRAG:
    def __init__(self, model, tokenizer, k=3, s=3):
        """
        Args:
            k: number of initial nodes to retrieve (top-k similarity)
            s: expansion steps in the graph
        """
        self.k = k
        self.s = s

        # Setup LLM Qwen
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,  # Low temperature for more deterministic responses
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Setup embeddings vor internal vector DB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.bm25_retriever=None
        self.node_list=[]
        self.graph = nx.DiGraph()
        self.vector_store = None
        self.node_to_content = {}

#-------------------------------CREATE CAUSAL RAG---------------------------------------#
    
    # RICHIESTA A QWEN PER ESTRARRE IL GRAFO
    def extract_causal_graph_manual(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract causal relationships using Qwen with explicit chat roles.

        Returns a list of json (cause, relation, effect).
        json:
        - cause_atomic => node id
        - cause_full => full node content
        - relation => edge
        - effect_atomic => node id
        - effect_full => full node content
        """
        system_content = """You are a Causal Intelligence Agent.
          Your goal is to find connections even when they are implicit.

          CRITICAL INSTRUCTIONS:
          - If the text says "A happened, and then B occurred", evaluate if A influenced B and extract it.
          - Look for trigger words: "led to", "influenced", "resulted in", "after which", "following", "response to".
          - Don't be afraid to extract multiple small steps.
          - If you find NO relations, output exactly: [] (nothing else)."""


        user_content = f"""Extract ALL causal relations from the text below.
          STRICT RULES:
          1. NO MISSING LINKS: If a chain of events exists, extract every link.
          2. CONSISTENT NAMING: Use the same 'cause_atomic' name if the event repeats in the text.
          3. GRANULARITY: Extract at least 3-5 relations per paragraph if present.

          FORMAT EXAMPLE:
          Text: "The heavy rain led to a flood which then caused the bridge to collapse."
          Output:
          [
            {{"cause_atomic": "Heavy rain", "cause_full": "Persistent heavy rainfall", "relation": "CAUSES", "effect_atomic": "Flooding", "effect_full": "River overflow and flooding"}},
            {{"cause_atomic": "Flooding", "cause_full": "River overflow and flooding", "relation": "CAUSES", "effect_atomic": "Bridge collapse", "effect_full": "Structural failure of the bridge"}}
          ]

          TEXT TO ANALYZE:
          {text}"""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            # Usa il chat template ufficiale Qwen
            prompt = self.llm.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.llm.pipeline.tokenizer(prompt, return_tensors="pt").to(self.llm.pipeline.model.device)
            with torch.no_grad():
              outputs = self.llm.pipeline.model.generate(
              **inputs,
              max_new_tokens=2000,
              do_sample=False,      # Determinism
              temperature=0.0,
              pad_token_id=self.llm.pipeline.tokenizer.eos_token_id
            )

            # Decoding e Cleaning
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            raw_response = self.llm.pipeline.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            clean_res= self._parse_causal_response(raw_response)
            return clean_res

        except Exception as e:
            print(f"Error extracting causal graph: {e}")
            return []


    
    def _parse_causal_response(self, response: str) -> List[dict]:
        """Parse the LLM response assuming a JSON list of objects.

           Returns list of json:

            "cause" => node id
            "cause_full"=> full node content
            "relation"=> type of causal relation
            "effect"=> node id
            "effect_full"=> full node content


        """
        relations = []
        if not response or not isinstance(response, str):
            return relations

        # 1. Pulizia e ricerca del blocco JSON
        # Cerchiamo il match più ampio possibile tra parentesi quadre
        match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)

        if match:
            clean_json = match.group(0)
        else:
            # Se non c'è una lista, cerchiamo un oggetto singolo
            match_single = re.search(r'\{.*\}', response, re.DOTALL)
            if match_single:
                clean_json = "[" + match_single.group(0) + "]"
            else:
                # Se l'LLM ha risposto "No relations found"
                return []

        # 2. Parsing (fuori dai blocchi if/else precedenti)
        try:
            data = json.loads(clean_json)

            if isinstance(data, dict):
                data = [data]

            for item in data:
                # Controllo robusto delle chiavi (usiamo i nomi estratti dal tuo prompt)
                c_atomic = item.get("cause_atomic")
                rel = item.get("relation")
                e_atomic = item.get("effect_atomic")

                if c_atomic and rel and e_atomic:
                    relations.append({
                        "cause": str(c_atomic).strip(),
                        "cause_full": str(item.get("cause_full", c_atomic)).strip(),
                        "relation": str(rel).upper(),
                        "effect": str(e_atomic).strip(),
                        "effect_full": str(item.get("effect_full", e_atomic)).strip()
                    })

        except json.JSONDecodeError as e:
            # Se fallisce, stampiamo solo l'inizio per debug
            print(f"JSON Decode Error: {e} | Preview: {clean_json[:50]}...")
        except Exception as e:
            print(f"General parsing error: {e}")

        return relations



    # DATI TUTTI I CHUNKS RELATIVI AD UN TOPIC ID GENERA e POPOLA IL GRAFO
    def index_documents(self, chunks: List[str]):
        """Manual extraction using Qwen """

        print(f"Indexing {len(chunks)} chunks...")


        pbar = tqdm(total=len(chunks), desc="Graph Extraction", unit="chunk")

        for chunk in chunks:
            try:
                # Manual extraction instead of transformer
                relations = self.extract_causal_graph_manual(chunk.page_content)

                for rel in relations:
                    # Trasformiamo l'atomic name in un ID standardizzato
                    c_id = rel["cause"].lower().strip().replace(".", "")
                    e_id = rel["effect"].lower().strip().replace(".", "")

                    # Iteriamo su causa ed effetto per registrarli nel grafo
                    nodes_to_process = [
                        (c_id, rel["cause_full"]),
                        (e_id, rel["effect_full"])
                    ]

                    for nid, full_text in nodes_to_process:
                        if nid not in self.graph:
                            # NODO NUOVO:
                            self.graph.add_node(nid)
                            self.node_to_content[nid] = full_text

                        else:
                          # NODO GIA' PRESENTE
                            existing_content = self.node_to_content.get(nid, "")
                            if full_text not in existing_content:
                              # Aggiungiamo il nuovo dettaglio separato da ","
                              self.node_to_content[nid] = existing_content + "," + full_text


                    # Aggiunta dell'Arco (Relazione)
                    self.graph.add_edge(
                        c_id,
                        e_id,
                        relation=rel["relation"],
                        # Opzionale: salviamo anche qui i testi originali per debug
                        cause_text=self.node_to_content[c_id],
                        effect_text=self.node_to_content[e_id],
                        )

                pbar.update(1)
            except Exception as e:
                print(f"Error chunk: {e}")
                pbar.update(1)
                continue

        pbar.close()

        node_ids = []
        all_contents = []

        # Preparo le liste dei nodi da inserire nei vector store
        for nid, content in self.node_to_content.items():
            node_ids.append(nid)
            all_contents.append(content)

        if not node_ids:
            print("⚠ No nodes extracted")
            return

        # 1. FAISS VECTOR STORE (Inizializzato una volta sola con TUTTI i testi)
        metadatas = [{"node_id": nid} for nid in node_ids]
        self.vector_store = FAISS.from_texts(
            texts=all_contents,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        # 2. BM25 INDEX
        # BM25 ha bisogno di un corpus tokenizzato: List[List[str]]
        documents = [
            Document(page_content=content, metadata={"node_id": nid})
            for nid, content in zip(node_ids, all_contents)
        ]
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        # necessario preimpostare il numero di documenti da recuperare:
        self.bm25_retriever.k = self.k * 2
        self.node_list = list(self.graph.nodes())

        print(f"✓ Graph created: {self.graph.number_of_nodes()} nodes, "
        f"{self.graph.number_of_edges()} edges")

#-------------------------------------------------------------------------------------#
#-------------------------------LOAD/STORE -------------------------------------------#
    
    def save_causal_rag(self, topic_id, folder="causal_data"):
        """Salva l'intero stato del CausalRAG"""
        topic_folder = os.path.join(folder, f"topic_{topic_id}")
        if not os.path.exists(topic_folder):
            os.makedirs(topic_folder)

        # 1. Salva il Grafo NetworkX
        with open(os.path.join(topic_folder, "graph.pkl"), "wb") as f:
            pickle.dump(self.graph, f)

        # 2. Salva la Mappa node_to_content
        with open(os.path.join(topic_folder, "node_map.pkl"), "wb") as f:
            pickle.dump(self.node_to_content, f)

        # 3. Salva il Vector Store (FAISS)
        if self.vector_store:
            self.vector_store.save_local(os.path.join(topic_folder, "faiss_index"))

        print(f"✓ Dati salvati correttamente per il topic {topic_id}")

    # RICARICA IL CAUSAL RAG DA DISCO
    def load_causal_rag(self, topic_id, folder="causal_data"):
        """
        Ricarica i dati e ricostruisce il BM25Retriever basandosi sui nodi salvati
        """
        topic_folder = os.path.join(folder, f"topic_{topic_id}")
        if not os.path.exists(topic_folder):
            raise FileNotFoundError(f"Cartella non trovata: {topic_folder}")

        # 1. Carica il Grafo
        with open(os.path.join(topic_folder, "graph.pkl"), "rb") as f:
            self.graph = pickle.load(f)

        # 2. Carica la Mappa node_to_content
        with open(os.path.join(topic_folder, "node_map.pkl"), "rb") as f:
            self.node_to_content = pickle.load(f)

        # 3. Carica il Vector Store (FAISS)
        vs_path = os.path.join(topic_folder, "faiss_index")
        self.vector_store = FAISS.load_local(
            vs_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # --- RICOSTRUZIONE BM25RETRIEVER ---
        # 4. Creiamo la lista di Documenti partendo dalla node_map caricata
        documents = []
        for nid, content in self.node_to_content.items():
            documents.append(Document(
                page_content=content,
                metadata={"node_id": nid}
            ))

        if documents:
            # Inizializziamo il retriever con i documenti ricostruiti
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            # Sincronizziamo il parametro k (opzionale ma consigliato)
            self.bm25_retriever.k = self.k * 2

        self.node_list = list(self.graph.nodes())    
        print(f"✓ Oggetto Ricostruito: {len(self.node_list)} nodi. BM25Retriever pronto.")

#-------------------------------------------------------------------------------------#
#----------------INDIVIDUARE NODI E CORRELAZIONI CAUSALI RILEVANTI--------------------#

    # SCEGLIERE I NODI DI PARTENZA  PIù AFFINI
    def retrieve_relevant_nodes(self, query: str) -> List[str]:
      """
      Hybrid Search utilizzando la Reciprocal Rank Fusion (RRF)
      """
      if not self.bm25_retriever or not self.vector_store:
          return []

      # 1. Ottieni i ranking da BM25
      result_docs = self.bm25_retriever.get_relevant_documents(query)
      bm25_results = [doc.metadata['node_id'] for doc in result_docs]

      # 2. Ottieni i ranking da FAISS
      semantic_results = self.vector_store.similarity_search_with_score(query, k=self.k *2)
      faiss_results = [doc.metadata['node_id'] for doc, score in semantic_results]

      # 3. Reciprocal Rank Fusion (RRF)
      # Calcola un punteggio combinato: più un nodo è in alto in entrambe le liste, meglio è.
      rrf_scores = {}
      k_constant = 60 # Valore standard per RRF per bilanciare i rank

      for rank, node_id in enumerate(bm25_results):
          rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1.0 / (k_constant + rank)

      for rank, node_id in enumerate(faiss_results):
          rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1.0 / (k_constant + rank)

      # 4. Ordina i nodi in base allo score RRF finale
      sorted_nodes = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

      # Debug: vedi quali nodi hanno fatto "match" in entrambi
      #combined_hits = [node for node in bm25_results if node in faiss_results]
      return [node for node, score in sorted_nodes[:self.k]]

    # PARTENDO DAI NODI INIZIALI ESPANDARE IL SOTTOGRAFO SEGUENDO GLI ARCHI
    def expand_nodes(self, initial_nodes: List[str]) -> Set[str]:
        """
        Espande i nodi per s passi (Breadth-First Search)
        """
        expanded = set(initial_nodes)
        current_layer = set(initial_nodes)

        for step in range(self.s):
            next_layer = set()
            for node in current_layer:
                if node in self.graph:
                    # Prende vicini entranti e uscenti
                    neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
                    # Consideriamo solo i vicini che non abbiamo ancora esplorato
                    new_neighbors = neighbors - expanded
                    next_layer.update(new_neighbors)

            if not next_layer:
                break

            expanded.update(next_layer)
            current_layer = next_layer # Il prossimo giro parte dai nodi appena trovati

        return expanded

     # PARTENDO DAI NODI INIZIALI ESPANDARE IL SOTTOGRAFO SEGUENDO GLI ARCHI IN DIREZIONE PREDEFINITA
    def expand_directional(self, initial_nodes: List[str], direction: str) -> Set[str]:
        """
        direction: 'forward' per le cause (successors),
                'backward' per gli effetti (predecessors)
        """
        expanded = set(initial_nodes)
        current_layer = set(initial_nodes)

        for step in range(self.s):
            next_layer = set()
            for node in current_layer:
                if node in self.graph:
                    if direction == 'forward':
                        # Seguiamo la freccia: cosa causa questo nodo?
                        neighbors = set(self.graph.successors(node))
                    else:
                        # Risaliamo la freccia: da cosa è causato questo nodo?
                        neighbors = set(self.graph.predecessors(node))

                    new_neighbors = neighbors - expanded
                    next_layer.update(new_neighbors)

            if not next_layer:
                break

            expanded.update(next_layer)
            current_layer = next_layer

        return expanded
    #------------------------------------------------------------------------------#
    #-------------------------APPROCCIO CERCARE PERCORSI CAUSALI-------------------#

    # INTERO PROCESSO DI SELEZIONE DEI NODI E DI RILEVAMENTO DI PERCORSI CAUSALI DA NODI CAUSA A NODI EFFETTO
    def find_all_causal_paths(self, relevant_nodes, cause_ids, effect_ids) -> Tuple[bool, List[List[str]]]:
      """
      Trova TUTTI i path causali tra nodi causa e nodi effetto
      Returns: (has_paths, list_of_paths)
      """

      if not cause_ids or not effect_ids:
          return False, []

      subgraph = self.graph.subgraph(relevant_nodes)

      # DEBUG
      # 2. Prepariamo i colori dei nodi
      color_map = []
      for node in subgraph.nodes():
        if node in cause_ids:
            color_map.append('red')       # Nodo Causa
        elif node in effect_ids:
            color_map.append('green')     # Nodo Effetto
        else:
            color_map.append('skyblue')   # Nodi intermedi (il percorso)

        # 3. Visualizzazione
      plt.figure(figsize=(12, 8))
      # Aumentiamo k per distanziare meglio i nodi se sono molti
      pos = nx.spring_layout(subgraph, seed=42, k=1.5)

      nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_color=color_map, # Usiamo la mappa dei colori creata
        node_size=2500,
        font_size=8,
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        edge_color='gray',
        alpha=0.9
      )

      plt.title(f"Analisi Percorso: Rosso (Causa) -> Verde (Effetto)")
      plt.show()


      all_paths = []

      for c_node in cause_ids:
          for e_node in effect_ids:
              if c_node == e_node:
                  #nota se nodo causa è uguale a nodo effetto la relazione è auto-contenuta
                  all_paths.append([c_node])
                  continue
              try:
                  has_path= nx.has_path(subgraph, c_node, e_node)
                  if has_path:
                      # Trova TUTTI i path semplici (senza cicli)
                      paths = list(nx.all_simple_paths(
                          subgraph,
                          c_node,
                          e_node,
                          cutoff=self.s + 1  # Limita lunghezza
                      ))
                      # Filtra path validi
                      all_paths.extend(paths)

              except Exception as e:
                  print(f"Errore: {e}")
                  continue

      # Ordina per qualità (path più corti = più forti causalmente)
      all_paths.sort(key=len)

      return len(all_paths) > 0, all_paths[:self.k]

    # PIPELINE COMPLETA PER VERIFICARE SE CAUSA → EFFETTO
    def is_causal_relation(self, cause: str, effect: str) -> Dict:
        """
        Pipeline completa per verificare se causa → effetto
        """
        # Step 1: Recupera nodi rilevanti per entrambi gli eventi
        cause_ids = self.retrieve_relevant_nodes(cause)
        effect_ids = self.retrieve_relevant_nodes(effect)
        print(f"cause nodes: {len(cause_ids)}, effect nodes {len(effect_ids)} ")

        # Step 2: Espandi i nodi
        expanded_cause = self.expand_directional(cause_ids, "forward")
        expanded_effect = self.expand_directional(effect_ids, "backward")

        # debug
        print(f"expanded cause nodes: {len(expanded_cause)}, expanded effect nodes {len(expanded_effect)} ")

        # Unisci i sottografi
        relevant_nodes = expanded_cause.union(expanded_effect).union(set(cause_ids)).union(set(effect_ids))

        # Step 3: Cerca percorso causale
        has_path, all_found_paths = self.find_all_causal_paths(relevant_nodes, cause_ids, effect_ids)

        # Step 4: Costruisci contesto per LLM
        context_parts = []
        context = ""

        if has_path:
            # Aggiungi relazioni del percorso per tutti i path trovati
            for path_segment in all_found_paths: # Iterate through each individual path
                if len(path_segment)==1:
                  context_parts.append(f"\n- {path_segment[0]}")
                  continue
                for i in range(len(path_segment) - 1):
                    u = path_segment[i]
                    v = path_segment[i+1]
                    if self.graph.has_edge(u, v):
                        edge_data = self.graph[u][v]
                        rel = edge_data.get('relation', 'unknown_relation' ).lower()
                        # Use cause_text and effect_text from edge_data for full text
                        cause_text = edge_data.get('cause_text', u)
                        effect_text = edge_data.get('effect_text', v)

                        # eliminare il punto se presente
                        if cause_text.endswith(('.', ';')):
                            cause_text = cause_text[:-1]

                        # iniziale minuscola
                        effect_text = effect_text[0].lower() + effect_text[1:]
                        rel = rel.replace("_", " ")
                        context_parts.append(f"- {cause_text} --[{rel}]--> {effect_text.lower()}")

            context = "\n".join(context_parts) # Join all parts once after processing all paths


        return {
            "has_path": has_path,
            "context":context
            }
    
    #-----------------------------------------------------------------------------#
    #-----------------APPROCCIO PERCORSI PARZIALI (MIGLIORE)----------------------#

    def find_partial_causal_paths(self, cause, effect) -> str:
        # 1. Recupero nodi rilevanti per entrambi i poli (Causa ed Effetto)
        cause_ids = self.retrieve_relevant_nodes(cause)
        effect_ids = self.retrieve_relevant_nodes(effect)

        # 2. Espansione bidirezionale
        # Espandiamo in AVANTI dalla causa (cosa provoca la causa?)
        cause_extended = self.expand_directional(cause_ids, "forward")
        # Espandiamo all'INDIETRO dall'effetto (da cosa è causato l'effetto?)
        effect_extended = self.expand_directional(effect_ids, "backward")

        # Unione dei nodi per creare un sottografo di contesto
        all_relevant_ids = set(cause_extended) | set(effect_extended)
        subgraph = self.graph.subgraph(all_relevant_ids)

        # 3. Serializzazione Archi (Relazioni)
        relations = []
        nodes_with_edges = set()

        for u, v, data in subgraph.edges(data=True):
            c_text = data.get('cause_text', u)
            e_text = data.get('effect_text', v)
            rel = data.get('relation', 'leads to').replace("_", " ")
            relations.append(f"- {c_text} --[{rel}]--> {e_text.lower()}")

            nodes_with_edges.update([u, v])

        # 4. Recupero Nodi "Orfani" o Specifici (Dettagli aggiuntivi su causa ed effetto)
        cause_context = []
        effect_context = []


        for node_id in cause_ids:
            if node_id not in nodes_with_edges:
                node_text = self.graph.nodes[node_id].get('text', node_id)
                cause_context.append(f"- Cause detail: {node_text}")

        for node_id in effect_ids:
            if node_id not in nodes_with_edges:
                node_text = self.graph.nodes[node_id].get('text', node_id)
                effect_context.append(f"- Target detail: {node_text}")

        # 5. Composizione del Summary
        summary_parts = []

        if relations:
            summary_parts.append("Relevant causal chains in context:\n" + "\n".join(list(set(relations))[:12]))

        if cause_context:
            summary_parts.append("Additional context on Candidate Cause:\n" + "\n".join(list(set(cause_context))[:5]))

        if effect_context:
            summary_parts.append("Additional context on Target Event:\n" + "\n".join(list(set(effect_context))[:5]))

        if not summary_parts:
            summary = "No direct path or relevant context found."
        else:
            summary = "Partial causal chains"+"\n\n".join(summary_parts)
        
        return summary
    
    #-----------------------------------------------------------------------------#
    #---------------------------------SYNTHESIZER---------------------------------#
    # analizzare i percorsi causali estratti e sintetizzarli in un testo narrativo coerente, con sfumature che il Verifier può interpretare per inferire la risposta finale.
    def generate_text_summary_from_causal_chain(self, raw_context, candidate_cause, target_event):
        """
        Sintetizza le catene grezze in un testo narrativo che evidenzia l'intensità
        e la natura dei legami causali senza inventare connessioni.
        """

        system_content = f"""You are a Causal Chain Synthesizer.
        Your task is to take raw causal segments and merge them into a clear, continuous logical narrative.

        STRATEGIES:
        1. PRESERVE THE FLOW: If the input shows A -> B and B -> C, write a cohesive paragraph that maintains this sequence. Use connectors like "which in turn led to" or "subsequently triggering."
        2. STRENGTH OF LINK: Reflect the intensity of the connections provided in the chains. If a link is direct, state it clearly.
        3. CONTEXTUAL GROUPING: If multiple chains share the same starting point or end goal, group them together to show the cumulative effect.
        4. AGNOSTIC TONE: Do not assume external knowledge. Stick strictly to the relationships defined in the provided chains, regardless of the topic (politics, science, history).
        5. NO FRAGMENTATION: Avoid using "Separately" unless the chains are logically and contextually disconnected.
        6. REMOVE NOISE: Delete stock market numbers, unrelated locations, or general trivia.
        7. FOCUS: Keep only details relevant to "{candidate_cause}" and "{target_event}"."""

        user_content = f"""RAW CONTEXT:
        {raw_context}

        CANDIDATE CAUSE: {candidate_cause}
        TARGET EVENT: {target_event}

        Synthesize the relevant causal paths:"""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # Utilizziamo l'apply_chat_template per coerenza con il modello Qwen/Llama
        prompt = self.llm.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.llm.pipeline.tokenizer(prompt, return_tensors="pt").to(self.llm.pipeline.model.device)
        outputs = self.llm.pipeline.model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.1,
            do_sample=False
        )

        summary = self.llm.pipeline.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Estrazione pulita della risposta dell'assistente
        if "assistant\n" in summary:
            summary = summary.split("assistant\n")[-1].strip()

        return summary


    #------------------------------------------------------------------------------#
    #-------------------------GRAFO PULIZIA E VISUALIZZAZIONE-----------------------#

    def visualize(self):
        """Disegna il grafo della causalità."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph, seed=42, k=2) # Layout elastico
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue',
                node_size=2000, font_size=9, font_weight='bold', arrows=True, arrowsize=20)
        plt.title("Grafo degli Eventi Causali")
        plt.show()


    def merge_similare_nodes(self, possible_node_to_fuse_ids):
        """
        Chiede a Qwen di valutare la fusione di nodi simili e aggiorna la struttura del grafo.
        """
        if not possible_node_to_fuse_ids:
            return

        # 1. Ricostruzione dati per il prompt
        nodes_data = []
        for nid in possible_node_to_fuse_ids:
            content = self.node_to_content.get(nid, {})
            # Gestione flessibile se content è stringa o dict
            details = content.get('text', str(content)) if isinstance(content, dict) else str(content)
            nodes_data.append({
                "EVENT_TITLE": nid,
                "EVENT_DETAILS": details
            })

        system_content = """You are a Causal Intelligence Agent.
            Your goal is to inspect the events and detect which are the same concept.
            If they should be fused, produce a SINGLE fused EVENT_TITLE and a summary EVENT_DETAILS.
            IMPORTANT: If the events are different (e.g., different countries or opposite actions), DO NOT fuse them; return them as separate items.
            Return ONLY a valid JSON list of objects."""

        user_content = f"""Analyze these events and fuse only those that represent the same entity or occurrence.
            Input:
            {json.dumps(nodes_data, indent=2)}

            Output format:
            [
            {{
                "EVENT_TITLE": "Final Title",
                "EVENT_DETAILS": "Combined details",
                "FUSED": ["Original Title 1", "Original Title 2"]
            }}
            ]
            """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        # 2. Generazione con Qwen
        prompt = self.llm.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llm.pipeline.tokenizer(prompt, return_tensors="pt").to(self.llm.pipeline.model.device)

        outputs = self.llm.pipeline.model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.llm.pipeline.tokenizer.eos_token_id
        )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        raw_response = self.llm.pipeline.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 3. Parsing del JSON (rimozione di eventuale markdown ```json ... ```)
        try:
            json_str = re.search(r'\[.*\]', raw_response, re.DOTALL).group()
            fused_results = json.loads(json_str)
        except Exception as e:
            print(f"Errore nel parsing JSON di Qwen: {e}")
            return

        # 4. Aggiornamento del Grafo (Logica di Re-wiring)
        for group in fused_results:
            new_title = group["EVENT_TITLE"]
            new_details = group["EVENT_DETAILS"]
            original_nodes = group.get("FUSED", [])

            if not original_nodes: continue

            # Crea il nuovo nodo se non esiste
            if new_title not in self.graph:
                self.graph.add_node(new_title)

            self.node_to_content[new_title] = {"text": new_details}

            for old_node in original_nodes:
                if old_node == new_title or old_node not in self.graph:
                    continue

                # Sposta gli archi uscenti: (old_node -> neighbor) diventa (new_title -> neighbor)
                for neighbor in list(self.graph.successors(old_node)):
                    edge_data = self.graph.get_edge_data(old_node, neighbor)
                    self.graph.add_edge(new_title, neighbor, **edge_data)

                # Sposta gli archi entranti: (predecessor -> old_node) diventa (predecessor -> new_title)
                for predecessor in list(self.graph.predecessors(old_node)):
                    edge_data = self.graph.get_edge_data(predecessor, old_node)
                    self.graph.add_edge(predecessor, new_title, **edge_data)

                # Rimuovi il vecchio nodo
                self.graph.remove_node(old_node)
                if old_node in self.node_to_content:
                    old_content= self.node_to_content[old_node]
                    del self.node_to_content[old_node]
                print(f"✓ Fuso node ID: {old_node} ---> {new_title}")
                print(f"✓ Content: {old_content} ---> {new_details}")



    def merge_nodes_and_sync_db(self, threshold=0.80):
        """
        Fonde i nodi simili nel grafo e ricostruisce il Vector Store FAISS
        per evitare errori di 'Node not in G'.
        """
        print(f"Inizio pulizia grafo: {len(self.graph.nodes)} nodi attuali.")

        nodes = list(self.graph.nodes())

        for i in range(len(nodes)):
            u = nodes[i]
            if u not in self.graph: continue

            possible_node_to_fuse=[]
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                if v not in self.graph: continue

                score = SequenceMatcher(None, u, v).ratio()
                if score >= threshold:
                    possible_node_to_fuse.append(v)

            if possible_node_to_fuse:
                possible_node_to_fuse.append(u)
                # Fare richiesta a qwen
                self.merge_similare_nodes(possible_node_to_fuse)

        # 3. RICOSTRUZIONE TOTALE DEL VECTOR STORE (FAISS)
        remaining_nodes = list(self.graph.nodes())
        # Prepariamo i nuovi dati basandoci SOLO su ciò che è rimasto nel grafo
        new_texts = []
        new_metadatas = []

        for nid in remaining_nodes:
            # Recuperiamo il testo pulito (o l'ID stesso se manca il contenuto)
            content = self.node_to_content.get(nid, str(nid))
            if isinstance(content, dict):
                content = content.get('text', str(content))

            new_texts.append(str(content))
            new_metadatas.append({"node_id": nid})

        # Sovrascriviamo il vecchio vector_store con uno nuovo di zecca
        self.vector_store = FAISS.from_texts(
            texts=new_texts,
            embedding=self.embeddings,
            metadatas=new_metadatas
        )
        # BM25 lavora su una lista di oggetti Documents
        new_docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(new_texts, new_metadatas)
        ]
        self.bm25_retriever = BM25Retriever.from_documents(new_docs)

        print(f"✓ Sincronizzazione completata.")
        print(f"✓ Nodi finali nel grafo: {len(self.graph.nodes)}")
        print(f"✓ Nodi finali nel DB: {len(new_texts)}")
  


# ============================================================================
# UTILIZZO PER SEMEVAL
# ============================================================================


def generate_causal_summary(causal_rag: CausalRAG,  cause, effect, type_search="partial_paths") -> str:
    """
    Genera un sommario causale di causa ed effetto utilizzando CausalRAG.
    """
    context = ""
    if type_search=="full_paths":
        # Verifica causalità
        result = causal_rag.is_causal_relation(
                cause=cause,
                effect=effect
            )
        context = result['context']
        
    if type_search=="partial_paths" or result['has_path']==False:
        
        context = causal_rag.find_partial_causal_paths(
                cause=cause,
                effect=effect
            )
    
    context= causal_rag.generate_text_summary_from_causal_chain(
            raw_context=context,
            candidate_cause=cause,
            target_event=effect
        )

    return context