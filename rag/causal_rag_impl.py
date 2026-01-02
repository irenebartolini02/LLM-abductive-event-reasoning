"""
Implementazione corretta di CausalRAG secondo il paper
"""

from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
import networkx as nx
from typing import List, Set, Dict
import numpy as np

# ============================================================================
# FASE 1: INDEXING (offline, prima della query)
# ============================================================================

class CausalRAG:
    def __init__(self, model, tokenizer, k=3, s=3):
        """
        Args:
            k: numero di nodi iniziali da recuperare (top-k similarity)
            s: step di espansione nel grafo
        """
        self.k = k
        self.s = s
        
        # Setup LLM
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, 
                       max_new_tokens=512)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Graph transformer - NON limitare solo a relazioni causali
        # Il paper costruisce prima un grafo generale!
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            # Rimuovi allowed_relationships per catturare TUTTE le relazioni
        )
        
        # Setup embeddings per il vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.graph = nx.DiGraph()
        self.vector_store = None
        self.node_to_content = {}  # Mappa node_id -> contenuto testuale
        
    def index_documents(self, documents: List[str]):
        """
        FASE 1: Indexing - costruisce il grafo base e il vector store
        """
        # 1.1 Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        docs = [Document(page_content=doc) for doc in documents]
        chunks = text_splitter.split_documents(docs)
        
        # 1.2 Costruzione grafo generale (non solo causale!)
        print("Costruendo grafo base...")
        graph_documents = self.transformer.convert_to_graph_documents(chunks)
        
        # 1.3 Popola il grafo NetworkX
        node_contents = []
        node_ids = []
        
        for graph_doc in graph_documents:
            # Aggiungi nodi
            for node in graph_doc.nodes:
                self.graph.add_node(node.id)
                # Salva il contenuto per embedding
                self.node_to_content[node.id] = node.id
                node_contents.append(node.id)
                node_ids.append(node.id)
            
            # Aggiungi TUTTE le relazioni (non solo causali)
            for rel in graph_doc.relationships:
                self.graph.add_edge(
                    rel.source.id, 
                    rel.target.id, 
                    relation=rel.type
                )
        
        # 1.4 Crea vector store per i nodi
        print(f"Indicizzando {len(node_contents)} nodi...")
        self.vector_store = FAISS.from_texts(
            texts=node_contents,
            embedding=self.embeddings,
            metadatas=[{"node_id": nid} for nid in node_ids]
        )
        
        print(f"Grafo creato: {self.graph.number_of_nodes()} nodi, "
              f"{self.graph.number_of_edges()} archi")
    
    # ========================================================================
    # FASE 2: QUERY TIME - Discovering & Estimating Causal Paths
    # ========================================================================
    
    def retrieve_initial_nodes(self, query: str) -> List[str]:
        """
        Step 2.1: Trova i k nodi più simili alla query
        """
        results = self.vector_store.similarity_search_with_score(query, k=self.k)
        node_ids = [doc.metadata["node_id"] for doc, _ in results]
        print(f"Nodi iniziali (k={self.k}): {node_ids}")
        return node_ids
    
    def expand_nodes(self, initial_nodes: List[str]) -> Set[str]:
        """
        Step 2.2: Espandi i nodi di s step lungo il grafo
        """
        expanded = set(initial_nodes)
        current_layer = set(initial_nodes)
        
        for step in range(self.s):
            next_layer = set()
            for node in current_layer:
                # Espandi sia predecessori che successori
                if node in self.graph:
                    next_layer.update(self.graph.predecessors(node))
                    next_layer.update(self.graph.successors(node))
            
            expanded.update(next_layer)
            current_layer = next_layer
            
            if not next_layer:
                break
        
        print(f"Nodi dopo espansione (s={self.s}): {len(expanded)} nodi")
        return expanded
    
    def extract_causal_subgraph(self, nodes: Set[str]) -> nx.DiGraph:
        """
        Step 2.3: Estrai il sottografo e identifica relazioni causali
        Qui è dove l'LLM analizza le relazioni per determinare causalità
        """
        subgraph = self.graph.subgraph(nodes).copy()
        causal_graph = nx.DiGraph()
        
        # Analizza ogni arco per determinare se è causale
        for u, v, data in subgraph.edges(data=True):
            relation = data.get('relation', '')
            
            # Prompt LLM per valutare causalità
            # (Semplificato - nel paper usano analisi più sofisticata)
            prompt = f"""Analizza se questa relazione è causale:
Nodo A: {u}
Relazione: {relation}
Nodo B: {v}

La relazione indica che A causa/influenza B? Rispondi solo SI o NO."""
            
            try:
                response = self.llm.invoke(prompt)
                if "SI" in response.upper():
                    causal_graph.add_edge(u, v, relation=relation)
            except:
                # Fallback: mantieni relazioni che sembrano causali
                causal_keywords = ['causa', 'produce', 'genera', 'porta', 
                                 'influenza', 'determina']
                if any(kw in relation.lower() for kw in causal_keywords):
                    causal_graph.add_edge(u, v, relation=relation)
        
        print(f"Grafo causale: {causal_graph.number_of_edges()} relazioni causali")
        return causal_graph
    
    # ========================================================================
    # FASE 3: RETRIEVING CONTEXT CAUSALLY
    # ========================================================================
    
    def generate_causal_summary(self, causal_graph: nx.DiGraph, 
                               initial_nodes: List[str]) -> str:
        """
        Step 3: Genera un sommario causale tracciando i percorsi chiave
        """
        summary_parts = []
        
        # Trova percorsi causali dai nodi iniziali
        for start_node in initial_nodes:
            if start_node not in causal_graph:
                continue
            
            # Percorsi in uscita (cosa causa questo nodo)
            for target in causal_graph.successors(start_node):
                path_info = f"{start_node} → {target}"
                edge_data = causal_graph.get_edge_data(start_node, target)
                if edge_data:
                    path_info += f" ({edge_data.get('relation', 'causa')})"
                summary_parts.append(path_info)
        
        if not summary_parts:
            return "Nessuna relazione causale diretta trovata."
        
        summary = "Relazioni causali chiave:\n" + "\n".join(summary_parts)
        return summary
    



