"""
Implementazione CausalRAG per SemEval-2010 Task 8 (Causal Reasoning)
"""

from matplotlib import pyplot as plt
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
            k: number of initial nodes to retrieve (top-k similarity)
            s: expansion steps in the graph
        """
        self.k = k
        self.s = s
        
        # Setup LLM
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=256,
            temperature=0.1,  # Low temperature for more deterministic responses
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
        """
        Extract causal relationships using Qwen with explicit chat roles.
        Returns a list of (cause, relation, effect).
        """

        system_content = (
            "You are an information extraction system specialized in causal reasoning.\n"
            "Your task is to extract causal relations between events from text.\n"
            "Be precise and concise. Do not add explanations."
        )

        user_content = f"""
        Analyze the following text and extract ALL causal relationships.

        Text:
        {text}

        Instructions:
        - Identify events or actions.
        - Determine if one event causes or leads to another.
        - Use ONLY these relations: CAUSES, RESULTS_IN, LEADS_TO.
        - If no causal relation exists, return NOTHING.

        Response format (repeatable):
        CAUSE: <event>
        RELATION: <CAUSES | RESULTS_IN | LEADS_TO>
        EFFECT: <event>
        ---
        """

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

            response = self.llm.invoke(prompt)

            return self._parse_causal_response(response)

        except Exception as e:
            print(f"Error extracting causal graph: {e}")
            return []


    def _parse_causal_response(self, response: str) -> List[Tuple[str, str, str]]:
        """Parse the LLM response"""
        relations = []
        if not response or not isinstance(response, str):
            return relations
        
        blocks = re.split(r'\n-{3,}\n', response)
        
        for block in blocks:
            cause_match = re.search(
                r'CAUSE:\s*(.+)', block, re.IGNORECASE
                )
            relation_match = re.search(
                r'RELATION\s*:\s*(CAUSES|RESULTS_IN|LEADS_TO)', block, re.IGNORECASE
                )
            effect_match = re.search(
                r'EFFECT:\s*(.+)', block, re.IGNORECASE
                )
            
            if not (cause_match and effect_match):
                continue
            
            cause = cause_match.group(1).strip()
            effect = effect_match.group(1).strip()
            
            relation = (
                relation_match.group(1).upper()
                if relation_match
                else "CAUSES"  # fallback sicuro
            )

            relations.append((cause, relation, effect))
        
        return relations

    def index_documents(self, documents: List[str]):
        """Modified version with manual extraction"""

        print(f"Indexing {len(documents)} documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        docs = [Document(page_content=doc) for doc in documents]
        chunks = text_splitter.split_documents(docs)
        
        print(f"Created {len(chunks)} chunks")
        
        node_contents = []
        node_ids = []
        
        pbar = tqdm(total=len(chunks), desc="Graph Extraction", unit="chunk")
        
        for chunk in chunks:
            try:
                # Manual extraction instead of transformer
                relations = self.extract_causal_graph_manual(chunk.page_content)
                
                for cause, effect, rel_type in relations:
                    # Add nodes
                    for node_text in [cause, effect]:
                        node_id = node_text[:100]  # Limit ID length
                        if node_id not in self.graph:
                            self.graph.add_node(node_id)
                            self.node_to_content[node_id] = node_text
                            node_contents.append(node_text)
                            node_ids.append(node_id)
                    
                    # Add edge
                    self.graph.add_edge(
                        cause[:100], 
                        effect[:100], 
                        relation=rel_type
                    )
                
                pbar.update(1)
            except Exception as e:
                print(f"Error chunk: {e}")
                pbar.update(1)
                continue
        
        pbar.close()
        
        # Create vector store
        if node_contents:
            self.vector_store = FAISS.from_texts(
                texts=node_contents,
                embedding=self.embeddings,
                metadatas=[{"node_id": nid} for nid in node_ids]
            )
            
            print(f"✓ Graph created: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges")
        else:
            print("⚠ No nodes extracted")   
    
    
    def retrieve_relevant_nodes(self, event_text: str) -> List[str]:
        """
        Find the k most relevant nodes for an event
        """
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(event_text, k=self.k)
        return [doc.metadata["node_id"] for doc in results]
    
    def expand_nodes(self, initial_nodes: List[str]) -> Set[str]:
        """
        Expand nodes by s steps
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
        Find a causal path between cause and effect in the subgraph
        """
        # Find relevant nodes for cause and effect
        cause_nodes = self.vector_store.similarity_search(cause, k=2)
        effect_nodes = self.vector_store.similarity_search(effect, k=2)
        
        cause_ids = [doc.metadata["node_id"] for doc in cause_nodes 
                     if doc.metadata["node_id"] in subgraph_nodes]
        effect_ids = [doc.metadata["node_id"] for doc in effect_nodes
                      if doc.metadata["node_id"] in subgraph_nodes]
        
        if not cause_ids or not effect_ids:
            return False, []
        
        # Search paths in the subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        for c_node in cause_ids:
            for e_node in effect_ids:
                try:
                    # Search direct path
                    if nx.has_path(subgraph, c_node, e_node):
                        path = nx.shortest_path(subgraph, c_node, e_node)
                        if len(path) <= self.s + 1:  # Respect expansion limit
                            return True, path
                except:
                    continue
        
        return False, []
    

    
    def check_causality_with_llm(self, cause: str, effect: str, 
                                 context: str) -> bool:
        """
        Use LLM to verify the causal relationship with the graph context
        """
        system_content = ("You are an expert in causal reasoning. Your task is to determine if a specific "
            "causal link exists based ONLY on the provided context. Distinguish between "
            "mere temporal succession and actual causal influence.")

        user_content = f"""Context from the knowledge graph:
{context}

Question: Does the event "{cause}" CAUSE or LEAD TO "{effect}"?

Follow this format strictly:
Answer: [YES or NO]
Reason: [Brief explanation, max 15 words]

Answer:
"""
        messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
        try:
            prompt = self.llm.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )

            raw_response = self.llm.invoke(prompt).strip()
            
            # Parsing della risposta
            clean_response = raw_response.split("Answer:")[-1].strip().upper()
            
            if clean_response.startswith("YES"):
                return True
            if clean_response.startswith("NO"):
                return False
        
            # Fallback: check first line for indicators
            words = re.findall(r'\w+', clean_response)
            if words:
                first_word = words[0]
                if first_word == 'YES': return True
                if first_word == 'NO': return False

            # Fallback: deeper analysis with keyword weighting
            positive_weight = sum(1 for kw in ['CAUSE', 'LEADS', 'RESULTED', 'TRIGGERED', 'INDUCED'] if kw in clean_response)
            negative_weight = sum(1 for kw in ['NOT', 'NO EVIDENCE', 'UNLIKELY', 'CORRELATION ONLY'] if kw in clean_response)
                
            return positive_weight > negative_weight
  
        except Exception as e:
            print(f"LLM error: {e}")
            return False
    
    # VA CAMBIATO
    def create_path_summary(self, path: List[str]) -> str:
        """
        Crea un sommario del percorso causale per il contesto
        """
        summary_parts = []
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                rel = edge_data.get('relation', 'is related to').upper()
                summary_parts.append(f"- {self.node_to_content.get(path[i], path[i])} {rel} {self.node_to_content.get(path[i+1], path[i+1])}")
        
        return "\n".join(summary_parts)
    
    
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
        is_causal = False
        context = ""

        if has_path:
            # Fare il riassunto del percorso

            path_desc = " → ".join(path[:4])  # Limita lunghezza
            context_parts.append(f"Percorso nel grafo: {path_desc}")
            
            # Aggiungi relazioni del percorso
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    rel = edge_data.get('relation')
                    context_parts.append(f"- {path[i]} {rel} {path[i+1]}")
       
            context = "\n".join(context_parts)

            summary = self.create_path_summary(path)
            print(f"Sommario percorso:\n{summary}\n")
            # Step 5: Final decision with LLM (only if path exists)
             ## MANCA da fare il riassunto del contesto al posto di passare le relazioni 
            is_causal = self.check_causality_with_llm(cause, effect, context)

        
        return {
            "is_causal": is_causal,
            "has_graph_path": has_path,
            "path": path if has_path else [],
            "context": context,
            "context parts": context_parts  
        }
    
    def visualize(self):
        """Disegna il grafo della causalità."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph, seed=42, k=2) # Layout elastico
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue',
                node_size=2000, font_size=9, font_weight='bold', arrows=True, arrowsize=20)
        plt.title("Grafo degli Eventi Causali")
        plt.show()


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

