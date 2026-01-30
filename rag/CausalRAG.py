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

    # REQUEST QWEN TO EXTRACT THE GRAPH
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
          - If you find NO relations, output exactly: [] (nothing else).

          CRITICAL JSON RULE:
           - NEVER use backslashes (\\).
           - If you need to write mathematical terms like E_d(1,1), write them directly as "Ed(1,1)" or "E_d(1,1)" WITHOUT any LaTeX wrapping like \( \).
           - Replace any backslash with a space or just remove it.
            """


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
            # Use official Qwen chat template
            prompt = self.llm.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.llm.pipeline.tokenizer(prompt, return_tensors="pt").to(self.llm.pipeline.model.device)
            with torch.no_grad():
              outputs = self.llm.pipeline.model.generate(
              **inputs,
              max_new_tokens=10000,
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
        """
        Parse the LLM response assuming a JSON list of objects.

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

        # --- PULIZIA PRE-PARSING ---
        # 1. Rimuoviamo i wrapper LaTeX \( \) che Qwen inserisce spesso
        clean_content = response.replace(r'\(', '(').replace(r'\)', ')')
        
        # 2. Rimuoviamo i backslash illegali (quelli non seguiti da virgolette)
        # Questo corregge l'errore char 119/168/240
        clean_content = re.sub(r'\\(?!["])', '', clean_content)

        # 3. Ricerca del blocco JSON
        match = re.search(r'\[\s*\{.*\}\s*\]', clean_content, re.DOTALL)

        if match:
            clean_json = match.group(0)
        else:
            match_single = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if match_single:
                clean_json = "[" + match_single.group(0) + "]"
            else:
                return []

        try:
            # strict=False is vital to ignore unescaped newlines or tabs
            data = json.loads(clean_json, strict=False)
        except json.JSONDecodeError as e:
            # Extreme fallback: remove ALL residual backslashes
            try:
                data = json.loads(clean_json.replace('\\', ''), strict=False)
            except:
                print(f"Fatal persistent error: {e} | Preview: {clean_json[:100]}")
                data = []
        
        # ... resto del filtraggio delle chiavi ...
        if isinstance(data, dict): data = [data]
        for item in data:
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
        return relations


    # GIVEN ALL CHUNKS RELATED TO A TOPIC ID, GENERATE AND POPULATE THE GRAPH
    def index_documents(self, chunks: List[str]):
        """Manual extraction using Qwen """

        print(f"Indexing {len(chunks)} chunks...")


        pbar = tqdm(total=len(chunks), desc="Graph Extraction", unit="chunk")

        for chunk in chunks:
            try:
                # Manual extraction instead of transformer
                relations = self.extract_causal_graph_manual(chunk.page_content)

                for rel in relations:
                    # Transform the atomic name into a standardized ID
                    c_id = rel["cause"].lower().strip().replace(".", "")
                    e_id = rel["effect"].lower().strip().replace(".", "")

                    # Iterate over cause and effect to register them in the graph
                    nodes_to_process = [
                        (c_id, rel["cause_full"]),
                        (e_id, rel["effect_full"])
                    ]

                    for nid, full_text in nodes_to_process:
                        if nid not in self.graph:
                            # NEW NODE:
                            self.graph.add_node(nid)
                            self.node_to_content[nid] = full_text

                        else:
                          # NODE ALREADY PRESENT
                            existing_content = self.node_to_content.get(nid, "")
                            if full_text not in existing_content:
                              # Add the new detail separated by comma
                              self.node_to_content[nid] = existing_content + "," + full_text


                    # Add the Edge (Relation)
                    self.graph.add_edge(
                        c_id,
                        e_id,
                        relation=rel["relation"],
                        # Optional: save original texts here for debugging
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
        # It is necessary to preset the number of documents to retrieve:
        self.bm25_retriever.k = self.k * 2
        self.node_list = list(self.graph.nodes())

        print(f"✓ Graph created: {self.graph.number_of_nodes()} nodes, "
        f"{self.graph.number_of_edges()} edges")

#-------------------------------------------------------------------------------------#
#-------------------------------LOAD/STORE -------------------------------------------

    def save_causal_rag(self, topic_id, folder="causal_data"):
        """Save the entire state of CausalRAG"""
        topic_folder = os.path.join(folder, f"topic_{topic_id}")
        if not os.path.exists(topic_folder):
            os.makedirs(topic_folder)

        # 1. Save the NetworkX Graph
        with open(os.path.join(topic_folder, "graph.pkl"), "wb") as f:
            pickle.dump(self.graph, f)

        # 2. Save the node_to_content Map
        with open(os.path.join(topic_folder, "node_map.pkl"), "wb") as f:
            pickle.dump(self.node_to_content, f)

        # 3. Save the Vector Store (FAISS)
        if self.vector_store:
            self.vector_store.save_local(os.path.join(topic_folder, "faiss_index"))

        print(f"✓ Data saved correctly for topic {topic_id}")

    # RELOAD THE CAUSAL RAG FROM DISK
    def load_causal_rag(self, topic_id, folder="causal_data"):
        """
        Reload the data and rebuild the BM25Retriever based on the saved nodes
        """
        topic_folder = os.path.join(folder, f"topic_{topic_id}")
        if not os.path.exists(topic_folder):
            raise FileNotFoundError(f"Folder not found: {topic_folder}")

        # 1. Load the Graph
        with open(os.path.join(topic_folder, "graph.pkl"), "rb") as f:
            self.graph = pickle.load(f)

        # 2. Load the node_to_content Map
        with open(os.path.join(topic_folder, "node_map.pkl"), "rb") as f:
            self.node_to_content = pickle.load(f)

        # 3. Load the Vector Store (FAISS)
        vs_path = os.path.join(topic_folder, "faiss_index")
        self.vector_store = FAISS.load_local(
            vs_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # --- RECONSTRUCTION BM25RETRIEVER ---
        # 4. Create the list of Documents starting from the loaded node_map
        documents = []
        for nid, content in self.node_to_content.items():
            documents.append(Document(
                page_content=content,
                metadata={"node_id": nid}
            ))

        if documents:
            # Initialize the retriever with the reconstructed documents
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            # Synchronize the k parameter (optional but recommended)
            self.bm25_retriever.k = self.k * 2

        self.node_list = list(self.graph.nodes())
        print(f"✓ Object Reconstructed: {len(self.node_list)} nodes. BM25Retriever ready.")

#-------------------------------------------------------------------------------------#
#----------------IDENTIFY NODES AND RELEVANT CAUSAL CORRELATIONS--------------------#

    # CHOOSE THE MOST RELATED STARTING NODES
    def retrieve_relevant_nodes(self, query: str) -> List[str]:
      """
      Hybrid Search using Reciprocal Rank Fusion (RRF)
      """
      if not self.bm25_retriever or not self.vector_store:
          return []

      # 1. Get the rankings from BM25
      result_docs = self.bm25_retriever.get_relevant_documents(query)
      bm25_results = [doc.metadata['node_id'] for doc in result_docs]

      # 2. Get the rankings from FAISS
      semantic_results = self.vector_store.similarity_search_with_score(query, k=self.k *2)
      faiss_results = [doc.metadata['node_id'] for doc, score in semantic_results]

      # 3. Reciprocal Rank Fusion (RRF)
      # Calculate a combined score: the more a node is high in both lists, the better.
      rrf_scores = {}
      k_constant = 60 # Standard value for RRF to balance ranks

      for rank, node_id in enumerate(bm25_results):
          rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1.0 / (k_constant + rank)

      for rank, node_id in enumerate(faiss_results):
          rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1.0 / (k_constant + rank)

      # 4. Sort the nodes based on the final RRF score
      sorted_nodes = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

      # Debug: see which nodes matched in both
      #combined_hits = [node for node in bm25_results if node in faiss_results]
      return [node for node, score in sorted_nodes[:self.k]]

    # STARTING FROM THE INITIAL NODES, EXPAND THE SUBGRAPH FOLLOWING THE EDGES
    def expand_nodes(self, initial_nodes: List[str]) -> Set[str]:
        """
        Expands the nodes for s steps (Breadth-First Search)
        """
        expanded = set(initial_nodes)
        current_layer = set(initial_nodes)

        for step in range(self.s):
            next_layer = set()
            for node in current_layer:
                if node in self.graph:
                    # Take incoming and outgoing neighbors
                    neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
                    # Consider only neighbors we haven't explored yet
                    new_neighbors = neighbors - expanded
                    next_layer.update(new_neighbors)

            if not next_layer:
                break

            expanded.update(next_layer)
            current_layer = next_layer # Next iteration starts from the newly found nodes

        return expanded

     # STARTING FROM THE INITIAL NODES, EXPAND THE SUBGRAPH FOLLOWING THE EDGES IN A PREDEFINED DIRECTION
    def expand_directional(self, initial_nodes: List[str], direction: str) -> Set[str]:
        """
        direction: 'forward' for causes (successors),
                'backward' for effects (predecessors)
        """
        expanded = set(initial_nodes)
        current_layer = set(initial_nodes)

        for step in range(self.s):
            next_layer = set()
            for node in current_layer:
                if node in self.graph:
                    if direction == 'forward':
                        # Follow the arrow: what causes this node?
                        neighbors = set(self.graph.successors(node))
                    else:
                        # Trace back the arrow: what is this node caused by?
                        neighbors = set(self.graph.predecessors(node))

                    new_neighbors = neighbors - expanded
                    next_layer.update(new_neighbors)

            if not next_layer:
                break

            expanded.update(next_layer)
            current_layer = next_layer

        return expanded
    #------------------------------------------------------------------------------#
    #-------------------------APPROACH: SEARCH FOR CAUSAL PATHS-------------------

    # ENTIRE PROCESS OF NODE SELECTION AND DETECTION OF CAUSAL PATHS FROM CAUSE NODES TO EFFECT NODES
    def find_all_causal_paths(self, relevant_nodes, cause_ids, effect_ids) -> Tuple[bool, List[List[str]]]:
      """
      Find ALL causal paths between cause nodes and effect nodes
      Returns: (has_paths, list_of_paths)
      """

      if not cause_ids or not effect_ids:
          return False, []

      subgraph = self.graph.subgraph(relevant_nodes)

      # DEBUG
      # 2. Prepare node colors
      color_map = []
      for node in subgraph.nodes():
        if node in cause_ids:
            color_map.append('red')       # Cause Node
        elif node in effect_ids:
            color_map.append('green')     # Effect Node
        else:
            color_map.append('skyblue')   # Intermediate nodes (the path)

        # 3. Visualization
      plt.figure(figsize=(12, 8))
      # Increase k to better space nodes if there are many
      pos = nx.spring_layout(subgraph, seed=42, k=1.5)

      nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_color=color_map, # Use the created color map
        node_size=2500,
        font_size=8,
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        edge_color='gray',
        alpha=0.9
      )

      plt.title(f"Path Analysis: Red (Cause) -> Green (Effect)")
      plt.show()


      all_paths = []

      for c_node in cause_ids:
          for e_node in effect_ids:
              if c_node == e_node:
                  #note if cause node equals effect node, the relation is self-contained
                  all_paths.append([c_node])
                  continue
              try:
                  has_path= nx.has_path(subgraph, c_node, e_node)
                  if has_path:
                      # Find ALL simple paths (without cycles)
                      paths = list(nx.all_simple_paths(
                          subgraph,
                          c_node,
                          e_node,
                          cutoff=self.s + 1  # Limit length
                      ))
                      # Filter valid paths
                      all_paths.extend(paths)

              except Exception as e:
                  print(f"Error: {e}")
                  continue

      # Sort by quality (shorter paths = stronger causally)
      all_paths.sort(key=len)

      return len(all_paths) > 0, all_paths[:self.k]

    # COMPLETE PIPELINE TO VERIFY IF CAUSE → EFFECT
    def is_causal_relation(self, cause: str, effect: str) -> Dict:
        """
        Complete pipeline to verify if cause → effect
        """
        # Step 1: Retrieve relevant nodes for both events
        cause_ids = self.retrieve_relevant_nodes(cause)
        effect_ids = self.retrieve_relevant_nodes(effect)
        print(f"cause nodes: {len(cause_ids)}, effect nodes {len(effect_ids)} ")

        # Step 2: Expand the nodes
        expanded_cause = self.expand_directional(cause_ids, "forward")
        expanded_effect = self.expand_directional(effect_ids, "backward")

        # debug
        print(f"expanded cause nodes: {len(expanded_cause)}, expanded effect nodes {len(expanded_effect)} ")

        # Merge the subgraphs
        relevant_nodes = expanded_cause.union(expanded_effect).union(set(cause_ids)).union(set(effect_ids))

        # Step 3: Search for causal path
        has_path, all_found_paths = self.find_all_causal_paths(relevant_nodes, cause_ids, effect_ids)

        # Step 4: Build context for LLM
        context_parts = []
        context = ""

        if has_path:
            # Add path relations for all found paths
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
    #-----------------APPROACH: PARTIAL PATHS (BETTER)----------------------#

    def find_partial_causal_paths(self, cause, effect) -> str:
        # 1. Retrieve relevant nodes for both poles (Cause and Effect)
        cause_ids = self.retrieve_relevant_nodes(cause)
        effect_ids = self.retrieve_relevant_nodes(effect)

        # 2. Bidirectional expansion
        # Expand FORWARD from the cause (what causes the cause?)
        cause_extended = self.expand_directional(cause_ids, "forward")
        # Expand BACKWARD from the effect (what is the effect caused by?)
        effect_extended = self.expand_directional(effect_ids, "backward")

        # Union of nodes to create a context subgraph
        all_relevant_ids = set(cause_extended) | set(effect_extended)
        subgraph = self.graph.subgraph(all_relevant_ids)

        # 3. Edge Serialization (Relations)
        relations = []
        nodes_with_edges = set()

        for u, v, data in subgraph.edges(data=True):
            c_text = data.get('cause_text', u)
            e_text = data.get('effect_text', v)
            rel = data.get('relation', 'leads to').replace("_", " ")
            relations.append(f"- {c_text} --[{rel}]--> {e_text.lower()}")

            nodes_with_edges.update([u, v])

        # 4. Retrieve "Orphan" or Specific Nodes (Additional context on cause and effect)
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

        # 5. Summary Composition
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
    # Analyze the extracted causal paths and synthesize them into a coherent narrative text, with nuances that the Verifier can interpret to infer the final answer.
    def generate_text_summary_from_causal_chain(self, raw_context, candidate_cause, target_event):
        """
        Synthesizes raw chains into a narrative text that highlights the intensity
        and nature of causal links without inventing connections.
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

        # Use apply_chat_template for consistency with the Qwen/Llama model
        prompt = self.llm.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.llm.pipeline.tokenizer(prompt, return_tensors="pt").to(self.llm.pipeline.model.device)
        outputs = self.llm.pipeline.model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.1,
            do_sample=False
        )

        summary = self.llm.pipeline.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean extraction of the assistant's response
        if "assistant\n" in summary:
            summary = summary.split("assistant\n")[-1].strip()

        return summary


    #------------------------------------------------------------------------------#
    #-------------------------GRAPH CLEANUP AND VISUALIZATION-----------------------

    def visualize(self):
        """Draw the causality graph."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph, seed=42, k=2) # Elastic layout
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue',
                node_size=2000, font_size=9, font_weight='bold', arrows=True, arrowsize=20)
        plt.title("Grafo degli Eventi Causali")
        plt.show()


    def merge_similare_nodes(self, possible_node_to_fuse_ids):
        """
        Asks Qwen to evaluate the merging of similar nodes and updates the graph structure.
        """
        if not possible_node_to_fuse_ids:
            return

        # 1. Data reconstruction for the prompt
        nodes_data = []
        for nid in possible_node_to_fuse_ids:
            content = self.node_to_content.get(nid, {})
            # Flexible handling if content is string or dict
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

        # 2. Generation with Qwen
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

        # 3. JSON parsing (remove eventual markdown ```json ... ```)
        try:
            json_str = re.search(r'\[.*\|', raw_response, re.DOTALL).group()
            fused_results = json.loads(json_str)
        except Exception as e:
            print(f"Error in JSON parsing of Qwen: {e}")
            return

        # 4. Graph Update (Re-wiring Logic)
        for group in fused_results:
            new_title = group["EVENT_TITLE"]
            new_details = group["EVENT_DETAILS"]
            original_nodes = group.get("FUSED", [])

            if not original_nodes: continue

            # Create the new node if it doesn't exist
            if new_title not in self.graph:
                self.graph.add_node(new_title)

            self.node_to_content[new_title] = {"text": new_details}

            for old_node in original_nodes:
                if old_node == new_title or old_node not in self.graph:
                    continue

                # Move outgoing edges: (old_node -> neighbor) becomes (new_title -> neighbor)
                for neighbor in list(self.graph.successors(old_node)):
                    edge_data = self.graph.get_edge_data(old_node, neighbor)
                    self.graph.add_edge(new_title, neighbor, **edge_data)

                # Move incoming edges: (predecessor -> old_node) becomes (predecessor -> new_title)
                for predecessor in list(self.graph.predecessors(old_node)):
                    edge_data = self.graph.get_edge_data(predecessor, old_node)
                    self.graph.add_edge(predecessor, new_title, **edge_data)

                # Remove the old node
                self.graph.remove_node(old_node)
                if old_node in self.node_to_content:
                    old_content= self.node_to_content[old_node]
                    del self.node_to_content[old_node]
                print(f"✓ Merged node ID: {old_node} ---> {new_title}")
                print(f"✓ Content: {old_content} ---> {new_details}")



    def merge_nodes_and_sync_db(self, threshold=0.80):
        """
        Merges similar nodes in the graph and rebuilds the FAISS Vector Store
        to avoid 'Node not in G' errors.
        """
        print(f"Starting graph cleanup: {len(self.graph.nodes)} current nodes.")

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
                # Make request to qwen
                self.merge_similare_nodes(possible_node_to_fuse)

        # 3. COMPLETE RECONSTRUCTION OF THE VECTOR STORE (FAISS)
        remaining_nodes = list(self.graph.nodes())
        # Prepare new data based ONLY on what remains in the graph
        new_texts = []
        new_metadatas = []

        for nid in remaining_nodes:
            # Retrieve the clean text (or the ID itself if content is missing)
            content = self.node_to_content.get(nid, str(nid))
            if isinstance(content, dict):
                content = content.get('text', str(content))

            new_texts.append(str(content))
            new_metadatas.append({"node_id": nid})

        # Overwrite the old vector_store with a brand new one
        self.vector_store = FAISS.from_texts(
            texts=new_texts,
            embedding=self.embeddings,
            metadatas=new_metadatas
        )
        # BM25 works on a list of Documents objects
        new_docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(new_texts, new_metadatas)
        ]
        self.bm25_retriever = BM25Retriever.from_documents(new_docs)

        print(f"✓ Synchronization completed.")
        print(f"✓ Final nodes in the graph: {len(self.graph.nodes)}")
        print(f"✓ Final nodes in the DB: {len(new_texts)}")



# ============================================================================
# USAGE FOR SEMEVAL
# ============================================================================

def generate_causal_summary(causal_rag: CausalRAG,  cause, effect, type_search="partial_paths") -> str:
    """
    Generates a causal summary of cause and effect using CausalRAG.
    """
    context = ""
    # Initialize result outside the if condition
    result = {'has_path': False, 'context': ''}

    if type_search=="full_paths":
        # Verify causality
        result = causal_rag.is_causal_relation(
                cause=cause,
                effect=effect
            )
        context = result['context']

    # This condition should use `type_search` directly or ensure `result` is always defined.
    # Modified to ensure context is always populated if `type_search` is 'partial_paths' or if full_paths did not find a path.
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


def build_causal_context(causal_rag, target_event, options):
    raw_context=""
    for option_text in options:

          # None of the others detection
          if "none of the others are correct causes" in option_text.lower().strip():
            continue

          cause= option_text
          effect=target_event
          
          context = causal_rag.find_partial_causal_paths(
                cause=cause,
                effect=effect
          )
    
          raw_context+="\n\n"+ context
    # OOM problems
    if len(raw_context) >=8000 :
        raw_context= raw_context[:8000] + "..."
    context_text= generate_text_summary_from_causal_chain(causal_rag, raw_context, target_event, options)
        
  
    return context_text


def search_causal_query(causal_rag, query, target_event, options)-> str:
     
    relevant_nodes= causal_rag.retrieve_relevant_nodes(query)
    extended_nodes = causal_rag.expand_nodes(relevant_nodes)
    subgraph = causal_rag.graph.subgraph(extended_nodes)
    relations = []
    nodes_with_edges = set()
    for u, v, data in subgraph.edges(data=True):
        c_text = data.get('cause_text', u)
        e_text = data.get('effect_text', v)
        rel = data.get('relation', 'leads to').replace("_", " ")
        relations.append(f"- {c_text} --[{rel}]--> {e_text.lower()}")
        nodes_with_edges.update([u, v])

    # Retrieve "Orphan" or Specific Nodes (Additional context)
    context = []
        
    for node_id in extended_nodes:
        if node_id not in nodes_with_edges:
            node_text = causal_rag.graph.nodes[node_id].get('text', node_id)
            context.append(f"-Detail: {node_text}")
        
    summary_parts = []
    if relations:
        summary_parts.append("Relevant causal chains in context:\n" + "\n".join(list(set(relations))[:12]))

    if context:
        summary_parts.append("Additional context:\n" + "\n".join(list(set(context))[:5]))

    if not summary_parts:
        summary = "No relevant context found."
    else:
        summary = "Causal chains"+"\n\n".join(summary_parts)

    # OOM problems
    if len(summary) >=5000:
        summary= summary[:5000] + "..."
        
    summary= generate_text_summary_from_causal_chain(causal_rag, summary, target_event, options)
        
    return summary
    


def generate_text_summary_from_causal_chain(causal_rag, raw_context, target_event, options):
        """
        Synthesizes raw chains into a narrative text that highlights the intensity
        and nature of causal links without inventing connections.
        """
        
        system_content = f"""You are a Causal Data Refiner. 
            Your ONLY task is to synthesize raw causal segments into a factual text paragraph.
            
            STRICT RULES:
            1. NO INFERENCE: Do not evaluate if a cause is "true" or "plausible". If the input says A --[LEADS_TO]--> B, write that A led to B. 
            2. ZERO NOISE: Delete details (names, numbers, side-events, adjectives) and causal chains that are not related with CANDIDATE CAUSE and the TARGET EVENT. 
            3. NO COMMENTS: Output ONLY the paragraph. Do not explain your choices or add introductory phrases.
            4. DATA LOYALTY: If the input provides isolated facts that don't form a chain, list them as independent sentences within the paragraph. Do not force a connection that isn't in the RAW CONTEXT."""
    

        user_content = f"""RAW CONTEXT:
        {raw_context}

            TARGET EVENT"{target_event}"
            CANDIDATE CAUSES:  
            A) "{options[0]}"
            B) "{options[1]}"
            C) "{options[2]}"
            D) "{options[3]}"

        Synthesize the relevant causal paths:"""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # Utilizziamo l'apply_chat_template per coerenza con il modello Qwen/Llama
        prompt = causal_rag.llm.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = causal_rag.llm.pipeline.tokenizer(prompt, return_tensors="pt").to(causal_rag.llm.pipeline.model.device)
        outputs = causal_rag.llm.pipeline.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=False
        )

        summary = causal_rag.llm.pipeline.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Estrazione pulita della risposta dell'assistente
        if "assistant\n" in summary:
            summary = summary.split("assistant\n")[-1].strip()

        return summary

