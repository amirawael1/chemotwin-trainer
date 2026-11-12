"""
utils/evidence_verifier.py
─────────────────────────────
Semantic similarity engine backing two things at once:
  1. Dense retrieval for the Guideline Agent (replacing the previous
     hardcoded gate->protocol lookup with real embedding-similarity
     search over data/guideline_corpus.py).
  2. The evidence-alignment / hallucination check (Safety Responsibility
     #3): given a piece of generated text (an intervention action, or
     an LLM narrative), how well is it actually supported by the
     retrieved guideline text, expressed as a cosine similarity score.

Dual backend:
  - PREFERRED: sentence-transformers, real semantic sentence
    embeddings. This is what your thesis methodology should describe
    and what your real environment should run.
  - FALLBACK: scikit-learn TF-IDF + cosine similarity, used
    automatically if sentence-transformers isn't installed. This is
    lexical, not semantic, similarity — weaker, but keeps the pipeline
    from hard-failing, consistent with this project's existing
    "abstain/fallback toward safety rather than crash" philosophy
    (see toxicity_models.py's deterministic fallback).

Swap the embedding model by changing EMBEDDING_MODEL_NAME below — e.g.
to a biomedical-domain model such as
'pritamdeka/S-PubMedBert-MS-MARCO' for closer alignment with your
proposal's BioLinkBERT/Clinical-Longformer framing, if your hardware
can support it. 'all-MiniLM-L6-v2' is the practical default: small,
fast, and adequate for short guideline-chunk-length text on a laptop.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    # Deliberately broad, not just ImportError: a present-but-broken
    # install (e.g. torch failing to load a Windows DLL like shm.dll,
    # a corrupted wheel, a missing Visual C++ Redistributable) raises
    # OSError/RuntimeError/etc at import time, not ImportError. Any
    # failure here should fall back to TF-IDF rather than crash the
    # whole pipeline at import time — this module's entire point is to
    # degrade gracefully, so the except clause has to be broad enough
    # to actually catch what real broken installs throw.
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity


@lru_cache(maxsize=1)
def _get_sentence_model():
    """Loaded once per process. Only called if sentence-transformers is available."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def backend_in_use() -> str:
    return "sentence-transformers" if _SENTENCE_TRANSFORMERS_AVAILABLE else "tfidf-fallback"


def _embed_semantic(texts: List[str]) -> np.ndarray:
    model = _get_sentence_model()
    return np.array(model.encode(texts, convert_to_numpy=True, show_progress_bar=False))


def _embed_tfidf(query_texts: List[str], corpus_texts: List[str]) -> Any:
    """
    TF-IDF fallback. Unlike the semantic path, this fits a fresh
    vectorizer over (query + corpus) each call — corpora here are small
    (tens of chunks), so this is cheap and avoids stale-vocabulary bugs
    from caching a fitted vectorizer across different query sets.
    Returns the fitted (query_vectors, corpus_vectors) pair.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = query_texts + corpus_texts
    matrix = vectorizer.fit_transform(all_texts)
    n_query = len(query_texts)
    return matrix[:n_query], matrix[n_query:]


def retrieve_top_k_chunks(
    query_text: str,
    corpus: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Dense (or TF-IDF fallback) retrieval: ranks `corpus` entries
    (each a dict with a "text" field, as in data/guideline_corpus.py)
    by similarity to `query_text`. Returns the top_k chunks, each with
    an added "similarity" float field.
    """
    if not corpus:
        return []

    corpus_texts = [c["text"] for c in corpus]

    if _SENTENCE_TRANSFORMERS_AVAILABLE:
        query_vec = _embed_semantic([query_text])
        corpus_vecs = _embed_semantic(corpus_texts)
        sims = sk_cosine_similarity(query_vec, corpus_vecs)[0]
    else:
        query_vec, corpus_vecs = _embed_tfidf([query_text], corpus_texts)
        sims = sk_cosine_similarity(query_vec, corpus_vecs)[0]

    ranked_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in ranked_indices:
        chunk = dict(corpus[idx])
        chunk["similarity"] = round(float(sims[idx]), 4)
        results.append(chunk)
    return results


# Default similarity thresholds, PER BACKEND. These are starting points
# for development, NOT clinically validated cutoffs — calibrate against
# a labeled set of known-grounded vs. known-fabricated claims before
# relying on this for anything beyond development/testing, and document
# whatever you end up using in your evaluation chapter.
#
# Why two sets: TF-IDF cosine similarity between two independently-
# phrased sentences on the same topic typically sits much lower
# (~0.1-0.3) than dense semantic embeddings of the same pair (~0.4-0.7),
# because TF-IDF only rewards literal shared vocabulary, not paraphrase/
# synonymy. Using the semantic thresholds against TF-IDF scores flags
# correctly-grounded text as a hallucination — confirmed empirically
# during development (see test_evidence_check.py in the accompanying
# test suite).
THRESHOLDS_BY_BACKEND = {
    "sentence-transformers": {"grounded": 0.55, "moderate": 0.35},
    "tfidf-fallback": {"grounded": 0.28, "moderate": 0.14},
}


def verify_evidence_alignment(
    claim_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    grounded_threshold: Optional[float] = None,
    moderate_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    The hallucination check itself: scores `claim_text` (e.g. an
    intervention action, or a sentence from an LLM narrative) against
    the guideline chunks already retrieved for it, and classifies how
    well-grounded it is.

    If grounded_threshold/moderate_threshold aren't passed explicitly,
    defaults are chosen based on which embedding backend is active
    (see THRESHOLDS_BY_BACKEND) — don't hardcode a single pair of
    thresholds elsewhere and assume they work for both backends.

    Returns: {is_grounded, max_similarity, best_chunk, hallucination_risk}
    hallucination_risk in {"LOW", "MODERATE", "HIGH"}.
    """
    backend = backend_in_use()
    defaults = THRESHOLDS_BY_BACKEND[backend]
    if grounded_threshold is None:
        grounded_threshold = defaults["grounded"]
    if moderate_threshold is None:
        moderate_threshold = defaults["moderate"]

    if not retrieved_chunks:
        return {
            "is_grounded": False,
            "max_similarity": 0.0,
            "best_chunk": None,
            "hallucination_risk": "HIGH",
            "note": "No retrieved guideline chunks available to check against.",
        }

    corpus_texts = [c["text"] for c in retrieved_chunks]

    if _SENTENCE_TRANSFORMERS_AVAILABLE:
        claim_vec = _embed_semantic([claim_text])
        chunk_vecs = _embed_semantic(corpus_texts)
    else:
        claim_vec, chunk_vecs = _embed_tfidf([claim_text], corpus_texts)

    sims = sk_cosine_similarity(claim_vec, chunk_vecs)[0]
    best_idx = int(np.argmax(sims))
    max_sim = float(sims[best_idx])

    if max_sim >= grounded_threshold:
        risk = "LOW"
    elif max_sim >= moderate_threshold:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    return {
        "is_grounded": max_sim >= moderate_threshold,
        "max_similarity": round(max_sim, 4),
        "best_chunk": retrieved_chunks[best_idx],
        "hallucination_risk": risk,
    }