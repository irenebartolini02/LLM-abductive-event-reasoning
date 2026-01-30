import torch
import re


class RagChain:
    def __init__(self, embedding_model, reranker, k_per_option=10, k_final=5, chunk_size=800, chunk_overlap=150):
        self.embedder = embedding_model
        self.k_final = k_final
        self.reranker = reranker
        self.k_per_option = k_per_option
        self.vector_index = {} # Dizionario per topic_id -> {'embeddings': ..., 'chunks': ...}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def doc_splitter(self, text):
        """
        Divide il testo in frasi e le raggruppa in chunk rispettando i limiti.
        """
        # 1. Pulizia preventiva
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # 2. Split in frasi
        sentences = re.split(r'(?<=[.!?]) +', text.strip())

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if current_chunk: chunks.append(current_chunk.strip())
                chunks.append(sentence[:self.chunk_size].strip()) # Troncamento estremo
                current_chunk = ""
                continue

            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk.strip())
                # Semplice logica di overlap: ricominciamo dalla frase attuale
                # (Puoi potenziarla recuperando le ultime frasi del chunk precedente)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _get_keywords(self, text, exclude_set=None):
        """Estrae parole significative escludendo le stop-words e un set opzionale di entità."""
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'was', 'were', 'is', 'are', 'to', 'and', 'or', 'but', 'how'}

        # Pulizia base
        words = re.findall(r'\w+', text.lower())

        # Set finale di parole chiave
        keywords = {w for w in words if w not in stop_words and len(w) > 2}

        # Point 1: Rimuoviamo le entità "rumorose" (il soggetto della domanda)
        if exclude_set:
            keywords = keywords - exclude_set

        return keywords

    def index_documents(self, docs_by_topic):
        """Trasforma i documenti in chunk e crea gli indici vettoriali per ogni topic."""
        for topic_id, doc_list in docs_by_topic.items():
            all_topic_chunks = []
            for d in doc_list:
                content = d.get("content", "")
                if content:
                    # Dividiamo il documento in piccoli pezzi
                    doc_chunks = self.doc_splitter(content)
                    all_topic_chunks.extend(doc_chunks)

            if all_topic_chunks:
                # Trasformiamo tutti i chunk del topic in vettori in un colpo solo
                embeddings = self.embedder.encode(all_topic_chunks, convert_to_tensor=True)
                self.vector_index[topic_id] = {
                    "chunks": all_topic_chunks,
                    "embeddings": embeddings
                }
        print(f"Indicizzazione completata per {len(self.vector_index)} topic.")

    def retrieve(self, topic_id, question, options):
        """Recupera i chunk più simili per ognuna delle 4 opzioni."""
        if topic_id not in self.vector_index:
            return []

        data = self.vector_index[topic_id]
        chunk_embeddings = data["embeddings"]
        all_chunks = data["chunks"]

        selected_indices = set()

        for opt in options:
            # Query composta: Domanda + Opzione specifica
            query_text = f"Question: {question} Option: {opt}"
            query_vec = self.embedder.encode(query_text, convert_to_tensor=True).reshape(1, -1)

            # Calcolo similarità con tutti i chunk del topic
            similarities = compute_cosine_similarity(query_vec, chunk_embeddings)[0]

            # Recupero i top K per questa specifica opzione
            top_k = torch.topk(similarities, k=min(self.k_per_option, len(all_chunks)))
            for idx in top_k.indices:
                selected_indices.add(idx.item())

        return [all_chunks[i] for i in selected_indices]


    def retrieve_hybrid(self, topic_id: int, question: str, options: List[str]) -> List[str]:
          if topic_id not in self.vector_index: return []

          data = self.vector_index[topic_id]
          chunks = data["chunks"]
          embeddings = data["embeddings"]

          # Identifichiamo il "soggetto" della domanda per non dargli boost
          subject_keywords = self._get_keywords(question)
          #print(f"\nSubject Keywords: {subject_keywords}\n")
          final_indices = {}

          for opt in options:
              if "none of the others" in opt.lower(): continue

              # 1. Semantica (Bi-Encoder)
              # ---------------------------------------------------------
              # ### NEW: MULTI-QUERY STRATEGY
              # ---------------------------------------------------------
              # Generiamo 3 prospettive diverse per massimizzare il recupero
              queries = [
                  f"Evidence that {opt} is the cause of {question}",  # 1. Causale (Originale)
                  f"{opt}",                                           # 2. Focus sull'Opzione (Chi è/Cos'è?)
                  f"{question}"                                       # 3. Focus sull'Evento (Cos'è successo?)
              ]

              ''' print(f"\nQueries of retrieve Hybrid========\n")
              for q in queries:
                print(q)
 '''
              # Codifichiamo tutte le query in batch [3, 768]
              q_embs = self.embedder.encode(queries, convert_to_tensor=True)
              # Calcoliamo la similarità di TUTTE le query contro TUTTI i chunk [3, num_chunks]
              # compute_cosine_similarity supporta matrici, quindi funziona nativamente
              all_sims = compute_cosine_similarity(q_embs, embeddings)

              # "Max-Pooling" delle similarità: per ogni chunk, teniamo il punteggio più alto
              # ottenuto tra le 3 query. Se un chunk parla molto bene dell'evento ma non della causa,
              # verrà comunque pescato grazie alla query #3.
              best_sims, _ = torch.max(all_sims, dim=0) # [num_chunks]

              # 2. Keyword Boosting Intelligente
              # Escludiamo le parole della domanda per focalizzarci solo su ciò che aggiunge l'opzione
              opt_keywords = self._get_keywords(opt, exclude_set=subject_keywords)

              boosts = []
              for c in chunks:
                  c_lower = c.lower()
                  # Boost alto (2.0) solo per le parole specifiche e nuove dell'opzione
                  score = sum(2.0 for kw in opt_keywords if kw in c_lower)
                  boosts.append(score * 0.1)

              boost_tensor = torch.tensor(boosts, device=best_sims.device)
              combined = best_sims + boost_tensor

              # Prendiamo i top 15 per ogni opzione come candidati per il reranker
              vals, idxs = torch.topk(combined, k=min(self.k_per_option, len(chunks)))
              for v, i in zip(vals, idxs):
                  idx_item = i.item()
                  # Logica di unione: se un chunk è già stato trovato, aggiorniamo il suo score se è migliore
                  if idx_item not in final_indices or v > final_indices[idx_item]:
                      final_indices[idx_item] = v

          # Deduplicazione e preparazione lista per Reranker
          sorted_idx = sorted(final_indices, key=final_indices.get, reverse=True)
          unique_candidates = []
          seen_texts = set()
          for i in sorted_idx:
              txt = chunks[i].strip()
              if txt not in seen_texts:
                  unique_candidates.append(txt)
                  seen_texts.add(txt)
                  if len(unique_candidates) >= self.k_per_option: break

          return unique_candidates

    def retrieve_and_rerank(self, topic_id, question, options):
          # 1. Fase di Retrieval Hybrid (prende 20 candidati basandosi su semantica + keyword)
          candidates = self.retrieve_hybrid(topic_id, question, options)

          ''' print(f"\n Candidates, to be reranked ================")
          for c in candidates:
            print(c)
            print(c)
 '''
          if not candidates: return []

          # 2. Fase di Reranking (Cross-Encoder)
          # Costruiamo una query che chieda al modello di trovare la causa
          query = f"Target Event: {question}. Which text provides evidence for these options: {', '.join(options)}?"

          #print(f"\nQuery of reranker======================\n {query}\n")

          pairs = [[query, cand] for cand in candidates]
          scores = self.reranker.predict(pairs)

          # Ordiniamo i chunk per lo score del Reranker
          # Il Cross-Encoder capirà che "Abe segretario nel 2000" non è utile per "Video social nel 2022"
          ranked_candidates = [c for _, c in sorted(zip(scores, candidates), reverse=True)]

          return ranked_candidates[:self.k_final]

    def search(self, topic_id: int, query: str, k: int = 3) -> List[str]:
        """
        Retrieves documents for a specific agentic query (e.g., "Date of Facebook rebrand").
        Uses standard vector similarity + Reranking.
        """
        if topic_id not in self.vector_index:
            return []

        data = self.vector_index[topic_id]
        chunk_embeddings = data["embeddings"]
        all_chunks = data["chunks"]

        # 1. DENSE RETRIEVAL (Vector Search)
        # ---------------------------------------------------------
        # We fetch slightly more candidates (k*3) to give the Reranker good options
        initial_k = k * 3
        
        query_vec = self.embedder.encode(query, convert_to_tensor=True).reshape(1, -1)
        similarities = compute_cosine_similarity(query_vec, chunk_embeddings)[0]
        
        # Get top candidates
        top_k_vals = torch.topk(similarities, k=min(initial_k, len(all_chunks)))
        candidates = [all_chunks[i] for i in top_k_vals.indices]

        if not candidates:
            return []

        # 2. RERANKING (Cross-Encoder)
        # ---------------------------------------------------------
        # The Cross-Encoder checks if the text ACTUALLY answers the specific question.
        pairs = [[query, cand] for cand in candidates]
        scores = self.reranker.predict(pairs)

        # Sort by reranker score
        ranked_candidates = [c for _, c in sorted(zip(scores, candidates), reverse=True)]

        # Return the top K (default 3 is usually enough for a specific fact)
        return ranked_candidates[:k]
    

def build_context(rag, topic_id, target_event, options):
    top_chunks = rag.retrieve_and_rerank(topic_id, target_event, options)
    # Restituisce i chunk uniti come stringa
    return "\n\n".join([f"[Initial Chunk {i+1}]: {c}" for i, c in enumerate(top_chunks)])

def search_query(rag,query,topic_id):
    top_chunks = rag.search(query=query, topic_id=topic_id,k=1)
    return "\n\n".join([f"[Retrieved chunk for: {query}]: {c}" for i, c in enumerate(top_chunks)])


def compute_cosine_similarity(query_embedding, chunk_embeddings):
    """
    Calcola la similarità coseno tra una query e una matrice di chunk.
    query_embedding: [1, d]
    chunk_embeddings: [n_chunks, d]
    """
    # Normalizzazione per ottenere il prodotto scalare come similarità coseno
    query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    chunk_norms = torch.nn.functional.normalize(chunk_embeddings, p=2, dim=1)

    # Risultato: tensore di similarità [1, n_chunks]
    return torch.mm(query_norm, chunk_norms.transpose(0, 1))