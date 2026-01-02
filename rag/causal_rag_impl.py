"""
Implementazione CausalRAG per SemEval-2010 Task 8 (Causal Reasoning)
"""

from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import pipeline
import networkx as nx
from typing import List, Set, Dict, Tuple
import re

class CausalRAGSemEval:
    def __init__(self, model, tokenizer, k=3, s=2):
        """
        Args:
            k: numero di nodi iniziali da recuperare
            s: step di espansione nel grafo
        """
        self.k = k
        self.s = s
        
        # Setup LLM
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=256,
            temperature=0.1,  # Bassa temperatura per risposte più deterministiche
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Graph transformer
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["Cause", "Effect", "Event"],
            allowed_relationships=["CAUSES", "RESULTS_IN", "LEADS_TO"]
        )

        
        # Setup embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.graph = nx.DiGraph()
        self.vector_store = None
        self.node_to_content = {}
        
    def extract_causal_graph_manual(self, text: str) -> List[Tuple[str, str, str]]:
        """Estrae nodi e relazioni causali usando l'LLM direttamente"""
        
        prompt = f"""Analizza il seguente testo ed estrai tutti gli eventi causali.
    Testo:
    {text}

    Istruzioni:
    1. Identifica eventi/azioni nel testo
    2. Trova relazioni causali (cosa causa cosa)
    3. Formato risposta:
    CAUSA: [evento che causa]
    EFFETTO: [evento risultante]
    ---
    (ripeti per ogni relazione)

    Risposta:"""

        try:
            response = self.llm.invoke(prompt)
            return self._parse_causal_response(response)
        except Exception as e:
            print(f"Errore estrazione: {e}")
            return []

    def _parse_causal_response(self, response: str) -> List[Tuple[str, str, str]]:
        """Parse della risposta LLM"""
        relations = []
        blocks = response.split('---')
        
        for block in blocks:
            causa_match = re.search(r'CAUSA:\s*(.+)', block, re.IGNORECASE)
            effetto_match = re.search(r'EFFETTO:\s*(.+)', block, re.IGNORECASE)
            
            if causa_match and effetto_match:
                causa = causa_match.group(1).strip()
                effetto = effetto_match.group(1).strip()
                relations.append((causa, effetto, "CAUSES"))
        
        return relations

    def index_documents(self, documents: List[str]):
        """Versione modificata con estrazione manuale"""

        print(f"Indicizzando {len(documents)} documenti...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        
        docs = [Document(page_content=doc) for doc in documents]
        chunks = text_splitter.split_documents(docs)
        
        print(f"Creati {len(chunks)} chunks")
        
        node_contents = []
        node_ids = []
        
        pbar = tqdm(total=len(chunks), desc="Estrazione Grafo", unit="chunk")
        
        for chunk in chunks:
            try:
                # Estrazione manuale invece di transformer
                relations = self.extract_causal_graph_manual(chunk.page_content)
                
                for causa, effetto, rel_type in relations:
                    # Aggiungi nodi
                    for node_text in [causa, effetto]:
                        node_id = node_text[:100]  # Limita lunghezza ID
                        if node_id not in self.graph:
                            self.graph.add_node(node_id)
                            self.node_to_content[node_id] = node_text
                            node_contents.append(node_text)
                            node_ids.append(node_id)
                    
                    # Aggiungi edge
                    self.graph.add_edge(
                        causa[:100], 
                        effetto[:100], 
                        relation=rel_type
                    )
                
                pbar.update(1)
            except Exception as e:
                print(f"Errore chunk: {e}")
                pbar.update(1)
                continue
        
        pbar.close()
        
        # Crea vector store
        if node_contents:
            self.vector_store = FAISS.from_texts(
                texts=node_contents,
                embedding=self.embeddings,
                metadatas=[{"node_id": nid} for nid in node_ids]
            )
            
            print(f"✓ Grafo creato: {self.graph.number_of_nodes()} nodi, "
                f"{self.graph.number_of_edges()} archi")
        else:
            print("⚠ Nessun nodo estratto")   
    
    
    def retrieve_relevant_nodes(self, event_text: str) -> List[str]:
        """
        Trova i k nodi più rilevanti per un evento
        """
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(event_text, k=self.k)
        return [doc.metadata["node_id"] for doc in results]
    
    def expand_nodes(self, initial_nodes: List[str]) -> Set[str]:
        """
        Espandi i nodi di s step
        """
        expanded = set(initial_nodes)
        current_layer = set(initial_nodes)
        
        for step in range(self.s):
            next_layer = set()
            for node in current_layer:
                if node in self.graph:
                    next_layer.update(self.graph.predecessors(node))
                    next_layer.update(self.graph.successors(node))
            
            expanded.update(next_layer)
            current_layer = next_layer
            
            if not next_layer:
                break
        
        return expanded
    
    def find_causal_path(self, cause: str, effect: str, 
                        subgraph_nodes: Set[str]) -> Tuple[bool, List[str]]:
        """
        Cerca un percorso causale tra causa ed effetto nel sottografo
        """
        # Trova nodi rilevanti per causa ed effetto
        cause_nodes = self.vector_store.similarity_search(cause, k=2)
        effect_nodes = self.vector_store.similarity_search(effect, k=2)
        
        cause_ids = [doc.metadata["node_id"] for doc in cause_nodes 
                     if doc.metadata["node_id"] in subgraph_nodes]
        effect_ids = [doc.metadata["node_id"] for doc in effect_nodes
                      if doc.metadata["node_id"] in subgraph_nodes]
        
        if not cause_ids or not effect_ids:
            return False, []
        
        # Cerca percorsi nel sottografo
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        for c_node in cause_ids:
            for e_node in effect_ids:
                try:
                    # Cerca percorso diretto
                    if nx.has_path(subgraph, c_node, e_node):
                        path = nx.shortest_path(subgraph, c_node, e_node)
                        if len(path) <= self.s + 1:  # Rispetta il limite di espansione
                            return True, path
                except:
                    continue
        
        return False, []
    
    def check_causality_with_llm(self, cause: str, effect: str, 
                                 context: str) -> bool:
        """
        Usa LLM per verificare la relazione causale con il contesto del grafo
        """
        prompt = f"""Analizza se esiste una relazione causale basandoti sul contesto fornito.

Contesto dal grafo di conoscenza:
{context}

Domanda: L'evento "{cause}" CAUSA o PORTA A "{effect}"?

Rispondi SOLO con "SI" o "NO", seguito da una brevissima spiegazione (max 20 parole).

Risposta:"""
        
        try:
            response = self.llm.invoke(prompt).strip()
            
            # Parsing della risposta
            first_line = response.split('\n')[0].upper()
            
            # Cerca indicatori positivi
            if any(word in first_line for word in ['SI', 'YES', 'TRUE', 'CAUSA']):
                return True
            # Cerca indicatori negativi
            if any(word in first_line for word in ['NO', 'FALSE', 'NON']):
                return False
            
            # Fallback: analisi più profonda
            positive_keywords = ['causa', 'porta', 'determina', 'produce', 
                               'influenza', 'precede', 'conseguenza']
            return any(kw in response.lower() for kw in positive_keywords)
            
        except Exception as e:
            print(f"Errore LLM: {e}")
            return False
    
    def is_causal_relation(self, cause: str, effect: str) -> Dict:
        """
        Pipeline completa per verificare se causa → effetto
        """
        # Step 1: Recupera nodi rilevanti per entrambi gli eventi
        cause_nodes = self.retrieve_relevant_nodes(cause)
        effect_nodes = self.retrieve_relevant_nodes(effect)
        
        # Step 2: Espandi i nodi
        expanded_cause = self.expand_nodes(cause_nodes)
        expanded_effect = self.expand_nodes(effect_nodes)
        
        # Unisci i sottografi
        relevant_nodes = expanded_cause.union(expanded_effect)
        
        # Step 3: Cerca percorso causale
        has_path, path = self.find_causal_path(cause, effect, relevant_nodes)
        
        # Step 4: Costruisci contesto per LLM
        context_parts = []
        if has_path:
            # Descrivi il percorso
            path_desc = " → ".join(path[:4])  # Limita lunghezza
            context_parts.append(f"Percorso nel grafo: {path_desc}")
            
            # Aggiungi relazioni del percorso
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    rel = edge_data.get('relation', 'collegato a')
                    context_parts.append(f"- {path[i]} {rel} {path[i+1]}")
        else:
            context_parts.append("Nessun percorso diretto trovato nel grafo.")
            # Fornisci nodi rilevanti comunque
            sample_nodes = list(relevant_nodes)[:5]
            context_parts.append(f"Nodi rilevanti: {', '.join(sample_nodes)}")
        
        context = "\n".join(context_parts)
        
        # Step 5: Decisione finale con LLM
        is_causal = self.check_causality_with_llm(cause, effect, context)
        
        return {
            "is_causal": is_causal,
            "has_graph_path": has_path,
            "path": path if has_path else [],
            "context": context
        }


# ============================================================================
# UTILIZZO PER SEMEVAL
# ============================================================================

def evaluate_semeval_entry(causal_rag: CausalRAGSemEval, entry: Dict) -> Dict:
    """
    Valuta un entry SemEval con CausalRAG
    """
    target_event = entry["target_event"]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Target Event: {target_event}")
    print(f"{'='*60}\n")
    
    # Valuta ogni opzione
    for option_key in ['option_A', 'option_B', 'option_C', 'option_D']:
        option_text = entry[option_key]
        option_label = option_key.split('_')[1]
        
        print(f"\n[Opzione {option_label}] {option_text}")
        print("-" * 60)
        
        # Verifica causalità
        result = causal_rag.is_causal_relation(
            cause=option_text,
            effect=target_event
        )
        
        results[option_label] = result["is_causal"]
        
        print(f"✓ Causale: {result['is_causal']}")
        print(f"  Path nel grafo: {result['has_graph_path']}")
        if result['path']:
            print(f"  Percorso: {' → '.join(result['path'][:3])}")
    
    # Predizione finale
    predicted = [k for k, v in results.items() if v]
    golden = entry["golden_answer"].split(',')
    
    print(f"\n{'='*60}")
    print(f"Predicted: {','.join(predicted)}")
    print(f"Golden:    {entry['golden_answer']}")
    print(f"Match: {set(predicted) == set(golden)}")
    print(f"{'='*60}\n")
    
    return {
        "id": entry["id"],
        "predicted": predicted,
        "golden": golden,
        "results": results
    }

