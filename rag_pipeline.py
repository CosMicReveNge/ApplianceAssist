"""
rag_pipeline.py
---------------
Core RAG pipeline components:
    1. Embedding generation  (Sentence Transformers – all-MiniLM-L6-v2)
    2. FAISS vector store    (in-memory index, cosine similarity)
    3. Similarity retrieval  with relevance threshold
    4. Gemini LLM integration (google-genai SDK v1+)
    5. Full RAG query function

Design notes:
    - The embedding model is a module-level singleton — loaded once, reused.
    - FAISS index is rebuilt only when new documents are uploaded.
    - Gemini is called ONLY after relevant chunks pass the threshold check.
    - Uses the new `google-genai` package (replaces deprecated google-generativeai).
"""

import time
import logging
import os
from typing import List, Tuple, Optional
import streamlit as st

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

from pdf_processor import DocumentChunk

# Load .env for local development (no-op on Render)
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RESULTS       = 3      # chunks to retrieve per query
RELEVANCE_THRESHOLD = 0.30   # minimum cosine similarity to include a chunk
GEMINI_MODEL_NAME   = "gemini-3.5-flash-lite"   # fast, free-tier model

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are ApplianceAssist AI, a helpful assistant that answers
questions using uploaded home appliance manuals.

STRICT RULES:

1. Answer questions ONLY using information from the
   retrieved manual context.

2. Do not invent, assume, or add instructions that are
   not present in the retrieved context.

3. PROCEDURE COMPLETENESS:
   - When answering a question about a procedure, include
     ALL relevant steps available in the retrieved context.
   - Preserve the original sequence of steps.
   - Include all numbered steps, substeps, warnings,
     cautions, prerequisites, and important notes.
   - If a procedure continues across multiple pages,
     combine the relevant information in the correct order.
   - Never intentionally stop halfway through a step.
   - Do not omit the final reassembly, verification, or
     safety instructions if they are present in the context.
   - If the retrieved context ends midway through a
     procedure, explicitly mention that the available
     information is incomplete.

4. SOURCE CITATIONS:
   - Cite the source filename and page number for each
     relevant instruction.
   - Do not invent page numbers.
   - Keep citations close to the instructions they support.

5. SAFETY:
   - Preserve all warnings and caution statements.
   - For electrical, gas, or refrigerant-related repairs,
     recommend contacting a qualified technician.

6. MISSING INFORMATION:
   - If the manual does not contain the answer, clearly
     state that the uploaded manual does not provide
     sufficient information.
   - Never answer from general knowledge.

7. RESPONSE FORMAT:
   - Use clear headings for different stages of a procedure.
   - Use numbered lists for sequential instructions.
   - Explain prerequisites before starting a procedure.
   - Include a short safety note when relevant.
   - Keep answers easy to understand.

Your primary objective is accuracy, completeness,
correct sequencing, and faithful use of the manual."""


# ---------------------------------------------------------------------------
# Embedding model — Streamlit resource cache
# ---------------------------------------------------------------------------

@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it across Streamlit reruns.
    """

    logger.info(
        "Loading embedding model '%s'…",
        EMBEDDING_MODEL_NAME
    )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Embedding model loaded successfully.")

    return model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate L2-normalised embeddings for a list of texts.

    L2 normalisation means that FAISS IndexFlatIP (inner product) gives
    true cosine similarity scores in the range [−1, 1].

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        numpy array shape (len(texts), 384), dtype float32.

    Raises:
        RuntimeError: If the model cannot produce embeddings.
    """
    model = get_embedding_model()
    try:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalise at source
        )
        return embeddings.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------
class FAISSIndex:
    """
    Wraps faiss.IndexFlatIP with the associated DocumentChunk list.

    After L2 normalisation, IndexFlatIP gives cosine similarity.
    Results are filtered by a relevance threshold before being returned.
    """

    def __init__(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings."
            )

        self.chunks = chunks
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info("FAISS index built: %d vectors, dim %d.", len(chunks), dim)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_RESULTS,
        threshold: float = RELEVANCE_THRESHOLD,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Find the top-k chunks whose cosine similarity ≥ threshold.

        Args:
            query_embedding: 1-D float32 array (dim,), L2-normalised.
            top_k:           Maximum results to return.
            threshold:       Minimum cosine similarity (0–1).

        Returns:
            List of (DocumentChunk, score) sorted by descending similarity.
        """
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = self.index.search(query_vec, top_k)

        results: List[Tuple[DocumentChunk, float]] = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:              # FAISS pads with -1 when < top_k vectors
                continue
            if float(sim) < threshold:
                continue
            results.append((self.chunks[idx], float(sim)))

        return results


def build_faiss_index(chunks: List[DocumentChunk]) -> FAISSIndex:
    """
    Embed all chunks and build a FAISSIndex.

    Raises:
        ValueError:   If chunks list is empty.
        RuntimeError: If embedding or index construction fails.
    """
    if not chunks:
        raise ValueError(
            "No text chunks to index. "
            "Please upload at least one valid PDF with extractable text."
        )

    texts = [c.text for c in chunks]
    try:
        embeddings = embed_texts(texts)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to generate embeddings for uploaded documents: {exc}"
        ) from exc

    try:
        return FAISSIndex(chunks, embeddings)
    except Exception as exc:
        raise RuntimeError(f"Failed to build FAISS index: {exc}") from exc


# ---------------------------------------------------------------------------
# Gemini API — new google-genai SDK
# ---------------------------------------------------------------------------
def get_gemini_api_key() -> str:
    """
    Read and validate the Gemini API key from the environment.

    Raises:
        EnvironmentError: If the key is missing or is still the placeholder.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file (local) or Render's Environment Variables."
        )
    if api_key.lower().startswith("your_"):
        raise EnvironmentError(
            "GEMINI_API_KEY looks like a placeholder. "
            "Replace it with your actual Gemini API key."
        )
    return api_key



def call_gemini(prompt: str) -> str:
    """
    Send a RAG prompt to Gemini and return the response text.

    Automatically retries temporary 503 server errors.
    """

    api_key = get_gemini_api_key()

    try:
        client = genai.Client(
    api_key=api_key,
    http_options=genai_types.HttpOptions(
        timeout=300000
    )
)

        max_retries = 1

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                        max_output_tokens=1200,
                    ),
                )

                # Extract text from the response
                if response.text:
                    return response.text.strip()

                # Handle blocked / filtered responses
                if response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(
                        candidate, "finish_reason", None
                    )

                    if finish_reason:
                        return (
                            f"Response blocked by the safety filter "
                            f"(reason: {finish_reason}). "
                            f"Please rephrase your question."
                        )

                return "No response generated. Please try again."

            except Exception as exc:
                err = str(exc).lower()

                # Retry temporary server errors (503)
                if "503" in err or "unavailable" in err:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt

                        logging.warning(
                            f"Gemini temporarily unavailable. "
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)
                        continue

                    raise RuntimeError(
                        "Gemini is temporarily experiencing high demand. "
                        "Please try again in a few minutes."
                    ) from exc

                # Handle quota / rate limits
                if "quota" in err or "rate" in err or "429" in err:
                    raise RuntimeError(
                        "Gemini API quota or rate limit exceeded. "
                        "Please wait and try again."
                    ) from exc

                # Handle invalid API keys
                if (
                    "api_key" in err
                    or "api key" in err
                    or "401" in err
                    or "403" in err
                ):
                    raise EnvironmentError(
                        "Gemini API key is invalid or unauthorised. "
                        "Please check your GEMINI_API_KEY."
                    ) from exc

                # Handle network errors
                if (
                    "network" in err
                    or "connection" in err
                    or "timeout" in err
                ):
                    raise RuntimeError(
                        "Network error while contacting Gemini API. "
                        "Please check your internet connection."
                    ) from exc

                raise RuntimeError(
                    f"Gemini API error: {exc}"
                ) from exc

    except EnvironmentError:
        raise


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(
    question: str,
    retrieved_chunks: List[Tuple[DocumentChunk, float]],
) -> str:
    """
    Construct the full RAG prompt to send to Gemini.

    The retrieved context is placed before the question so the model
    treats it as authoritative grounding evidence.

    Args:
        question:         User's question string.
        retrieved_chunks: List of (DocumentChunk, similarity_score) pairs.

    Returns:
        Formatted prompt string ready for Gemini.
    """
    context_parts = []
    for chunk, score in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk.source}, Page {chunk.page}]\n{chunk.text}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    return (
        f"CONTEXT (extracted from uploaded appliance manuals):\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer the question using only the context above. "
        f"If the context does not contain the answer, say so explicitly."
    )


# ---------------------------------------------------------------------------
# Source citations
# ---------------------------------------------------------------------------
def extract_unique_sources(
    retrieved_chunks: List[Tuple[DocumentChunk, float]]
) -> List[dict]:
    """
    Build a de-duplicated list of source citations from retrieved chunks.

    Returns:
        List of dicts with keys: 'source', 'page', 'similarity'.
        Sorted by descending similarity score.
    """
    seen: set = set()
    sources: List[dict] = []

    for chunk, score in retrieved_chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": chunk.source,
                "page": chunk.page,
                "similarity": round(score, 3),
            })

    return sorted(sources, key=lambda x: x["similarity"], reverse=True)


# ---------------------------------------------------------------------------
# Main RAG query function
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    faiss_index: FAISSIndex,
    top_k: int = TOP_K_RESULTS,
    threshold: float = RELEVANCE_THRESHOLD,
) -> Tuple[str, List[dict], bool]:
    """
    Full RAG pipeline: embed question → retrieve → call Gemini → cite.

    Args:
        question:    User's natural-language question.
        faiss_index: Pre-built FAISSIndex.
        top_k:       Number of chunks to retrieve.
        threshold:   Minimum cosine similarity for inclusion.

    Returns:
        answer:  Generated answer string.
        sources: List of source citation dicts (source, page, similarity).
        found:   True if relevant context was found; False if threshold unmet.

    Raises:
        ValueError:       If the question is empty.
        EnvironmentError: If the API key is missing.
        RuntimeError:     If embedding or API call fails.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    # Step 1 — Embed the question
    try:
        q_embedding = embed_texts([question])[0]
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to embed the question: {exc}") from exc

    # Step 2 — Retrieve relevant chunks from FAISS
    retrieved = faiss_index.search(q_embedding, top_k=top_k, threshold=threshold)

    if not retrieved:
        # No chunk passed the relevance threshold — don't call Gemini
        return (
            "The uploaded manuals do not contain sufficient information to "
            "answer this question. Please ensure you have uploaded the correct "
            "appliance manual, or try rephrasing your question.",
            [],
            False,
        )

    # Step 3 — Build the grounded prompt
    prompt = build_prompt(question, retrieved)

    # Step 4 — Call Gemini
    answer = call_gemini(prompt)

    # Step 5 — Extract citations from the retrieved chunks
    sources = extract_unique_sources(retrieved)

    return answer, sources, True
