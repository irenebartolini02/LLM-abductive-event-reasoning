import torch
import re


class RagChain:
    def __init__(self, embedding_model, reranker, k_per_option=10, k_final=5, chunk_size=800, chunk_overlap=150):
        self.embedder = embedding_model
        self.k_final = k_final
        self.reranker = reranker
        self.k_per_option = k_per_option
        self.vector_index = {} # Dictionary for topic_id -> {'embeddings': ..., 'chunks': ...}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def doc_splitter(self, text):
        """
        Splits the text into sentences and groups them into chunks respecting limits.
        """
        # 1. Preventive cleanup
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # 2. Split into sentences
        sentences = re.split(r'(?<=[.!?]) +', text.strip())

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if current_chunk: chunks.append(current_chunk.strip())
                chunks.append(sentence[:self.chunk_size].strip()) # Extreme truncation
                current_chunk = ""
                continue

            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk.strip())
                # Simple overlap logic: restart from the current sentence
                # (You can enhance it by recovering the last sentences from the previous chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _get_keywords(self, text, exclude_set=None):
        """Extracts significant words excluding stop-words and an optional set of entities."""
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'was', 'were', 'is', 'are', 'to', 'and', 'or', 'but', 'how'}

        # Basic cleanup
        words = re.findall(r'\w+', text.lower())

        # Final set of keywords
        keywords = {w for w in words if w not in stop_words and len(w) > 2}

        # Point 1: Remove "noisy" entities (the subject of the question)
        if exclude_set:
            keywords = keywords - exclude_set

        return keywords

    def index_documents(self, docs_by_topic):
        """Transforms documents into chunks and creates vector indices for each topic."""
        for topic_id, doc_list in docs_by_topic.items():
            all_topic_chunks = []
            for d in doc_list:
                content = d.get("content", "")
                if content:
                    # Split the document into small pieces
                    doc_chunks = self.doc_splitter(content)
                    all_topic_chunks.extend(doc_chunks)

            if all_topic_chunks:
                # Transform all chunks of the topic into vectors in one go
                embeddings = self.embedder.encode(all_topic_chunks, convert_to_tensor=True)
                self.vector_index[topic_id] = {
                    "chunks": all_topic_chunks,
                    "embeddings": embeddings
                }
        print(f"Indexing completed for {len(self.vector_index)} topics.")

    def retrieve(self, topic_id, question, options):
        """Retrieves the most similar chunks for each of the 4 options."""
        if topic_id not in self.vector_index:
            return []

        data = self.vector_index[topic_id]
        chunk_embeddings = data["embeddings"]
        all_chunks = data["chunks"]

        selected_indices = set()

        for opt in options:
            # Composite query: Question + Specific option
            query_text = f"Question: {question} Option: {opt}"
            query_vec = self.embedder.encode(query_text, convert_to_tensor=True).reshape(1, -1)

            # Calculate similarity with all chunks of the topic
            similarities = compute_cosine_similarity(query_vec, chunk_embeddings)[0]

            # Retrieve top K for this specific option
            top_k = torch.topk(similarities, k=min(self.k_per_option, len(all_chunks)))
            for idx in top_k.indices:
                selected_indices.add(idx.item())

        return [all_chunks[i] for i in selected_indices]


    def retrieve_hybrid(self, topic_id: int, question: str, options: List[str]) -> List[str]:

          data = self.vector_index[topic_id]
          chunks = data["chunks"]
          embeddings = data["embeddings"]

          # Identify the "subject" of the question to avoid boosting it
          subject_keywords = self._get_keywords(question)
          #print(f"\nSubject Keywords: {subject_keywords}\n")
          final_indices = {}

          for opt in options:
              if "none of the others" in opt.lower(): continue

              # 1. Semantic (Bi-Encoder)
              # ---------------------------------------------------------
              # ### NEW: MULTI-QUERY STRATEGY
              # ---------------------------------------------------------
              # Generate 3 different perspectives to maximize retrieval
              queries = [
                  f"Evidence that {opt} is the cause of {question}",  # 1. Causal (Original)
                  f"{opt}",                                           # 2. Focus on the Option (Who is/What is?)
                  f"{question}"                                       # 3. Focus on the Event (What happened?)
              ]

              ''' print(f"\nQueries of retrieve Hybrid========\n")
              for q in queries:
                print(q)
 '''
              # Encode all queries in batch [3, 768]
              q_embs = self.embedder.encode(queries, convert_to_tensor=True)
              # Calculate similarity of ALL queries against ALL chunks [3, num_chunks]
              # compute_cosine_similarity supports matrices, so it works natively
              all_sims = compute_cosine_similarity(q_embs, embeddings)

              # "Max-Pooling" of similarities: for each chunk, keep the highest score
              # obtained among the 3 queries. If a chunk speaks well about the event but not the cause,
              # it will still be retrieved thanks to query #3.
              best_sims, _ = torch.max(all_sims, dim=0) # [num_chunks]

              # 2. Intelligent Keyword Boosting
              # Exclude the words of the question to focus only on what the option adds
              opt_keywords = self._get_keywords(opt, exclude_set=subject_keywords)

              boosts = []
              for c in chunks:
                  c_lower = c.lower()
                  # High boost (2.0) only for specific and new words of the option
                  score = sum(2.0 for kw in opt_keywords if kw in c_lower)
                  boosts.append(score * 0.1)

              boost_tensor = torch.tensor(boosts, device=best_sims.device)
              combined = best_sims + boost_tensor

              # Take top 15 for each option as candidates for the reranker
              vals, idxs = torch.topk(combined, k=min(self.k_per_option, len(chunks)))
              for v, i in zip(vals, idxs):
                  idx_item = i.item()
                  # Union logic: if a chunk was already found, update its score if it's better
                  if idx_item not in final_indices or v > final_indices[idx_item]:
                      final_indices[idx_item] = v

          # Deduplication and preparation for Reranker
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
          # 1. Hybrid Retrieval Phase (retrieves 20 candidates based on semantic + keyword)
          candidates = self.retrieve_hybrid(topic_id, question, options)

          ''' print(f"\n Candidates, to be reranked ================")
          for c in candidates:
            print(c)
            print(c)
 '''
          if not candidates: return []

          # 2. Reranking Phase (Cross-Encoder)
          # Build a query that asks the model to find the cause
          query = f"Target Event: {question}. Which text provides evidence for these options: {', '.join(options)}?"

          #print(f"\nQuery of reranker======================\n {query}\n")

          pairs = [[query, cand] for cand in candidates]
          scores = self.reranker.predict(pairs)

          # Sort chunks by Reranker score
          # The Cross-Encoder will understand that "Abe secretary in 2000" is not useful for "Social video in 2022"
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
    Calculate cosine similarity between a query and a matrix of chunks.
    query_embedding: [1, d]
    chunk_embeddings: [n_chunks, d]
    """
    # Normalization to obtain the scalar product as cosine similarity
    query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    chunk_norms = torch.nn.functional.normalize(chunk_embeddings, p=2, dim=1)

    # Result: similarity tensor [1, n_chunks]
    return torch.mm(query_norm, chunk_norms.transpose(0, 1))