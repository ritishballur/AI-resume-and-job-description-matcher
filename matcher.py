from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import chunk_text, simple_clean

class Matcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def match_resume_to_jd(self, resume_text, jd_text, top_k=5):
        resume_chunks = chunk_text(simple_clean(resume_text))
        jd_chunks = chunk_text(simple_clean(jd_text))

        resume_emb = self.embed_texts(resume_chunks)
        jd_emb = self.embed_texts(jd_chunks)

        sim = cosine_similarity(resume_emb, jd_emb)

        best_per_chunk = sim.max(axis=1)
        idx_sorted = np.argsort(best_per_chunk)[::-1]
        top_idx = idx_sorted[:top_k]

        results = []
        for i in top_idx:
            results.append({
                'resume_chunk': resume_chunks[i],
                'score': float(best_per_chunk[i])
            })

        overall_score = float(np.mean(best_per_chunk[top_idx])) if len(top_idx) > 0 else 0.0

        return {
            'overall_score': overall_score,
            'top_matches': results
        }