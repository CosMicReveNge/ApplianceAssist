
"""
rag_pipeline.py
---------------
Core RAG pipeline components:
    1. Embedding generation using Gemini Embedding API
    2. FAISS vector store (in-memory index, cosine similarity)
    3. Similarity retrieval with relevance threshold
    4. Gemini LLM integration (google-genai SDK v1+)
    5. Full RAG query function

Design notes:
    - Embeddings are generated remotely using Gemini.
    - No local Sentence Transformer or PyTorch model is required.
    - FAISS index is rebuilt when new documents are uploaded.
    - Gemini is called for answers only after relevant chunks
      pass the threshold check.
"""

import time
import logging
import os
from typing import List, Tuple

import numpy as np
import faiss

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

EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 16

TOP_K_RESULTS = 3
RELEVANCE_THRESHOLD = 0.30

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"


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
# Gemini API key and client
# ---------------------------------------------------------------------------

def get_gemini_api_key() -> str:
    """
    Read and validate the Gemini API key from the environment.
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


def get_gemini_client():
    """
    Create a Gemini API client.
    """

    api_key = get_gemini_api_key()

    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(
            timeout=60000
        )
    )


# ---------------------------------------------------------------------------
# Gemini Embeddings
# ---------------------------------------------------------------------------

def embed_texts(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> np.ndarray:
    """
    Generate L2-normalized embeddings using Gemini Embedding API.

    RETRIEVAL_DOCUMENT is used for manual chunks.
    RETRIEVAL_QUERY is used for user questions.

    Returns:
        Normalized NumPy array of shape
        (len(texts), EMBEDDING_DIMENSION).
    """

    if not texts:
        raise ValueError(
            "Cannot generate embeddings for empty text."
        )

    # Validate input texts.
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError(
            "All embedding inputs must be non-empty strings."
        )

    client = get_gemini_client()

    all_embeddings = []

    try:

        # Process texts in batches.
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):

            batch = texts[start:start + EMBEDDING_BATCH_SIZE]

            logger.info(
                "Generating embeddings for batch %d to %d of %d",
                start + 1,
                start + len(batch),
                len(texts)
            )

            response = client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=batch,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSION
                )
            )

            if not response.embeddings:
                raise RuntimeError(
                    "Gemini returned an empty embedding response."
                )

            if len(response.embeddings) != len(batch):
                raise RuntimeError(
                    "Number of returned embeddings does not match "
                    "the number of input texts."
                )

            for embedding in response.embeddings:

                if embedding.values is None:
                    raise RuntimeError(
                        "Gemini returned an embedding without values."
                    )

                all_embeddings.append(embedding.values)

        # Convert embeddings into a float32 NumPy array.
        embeddings = np.asarray(
            all_embeddings,
            dtype=np.float32
        )

        expected_shape = (
            len(texts),
            EMBEDDING_DIMENSION
        )

        if embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected embedding shape: {embeddings.shape}. "
                f"Expected: {expected_shape}"
            )

        # Check for invalid numerical values.
        if not np.all(np.isfinite(embeddings)):
            raise RuntimeError(
                "Gemini returned embeddings containing invalid values."
            )

        # L2 normalization for cosine similarity using FAISS.
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        if np.any(norms == 0):
            raise RuntimeError(
                "Gemini returned a zero-length embedding."
            )

        embeddings = embeddings / norms

        logger.info(
            "Successfully generated %d embeddings.",
            len(embeddings)
        )

        return embeddings.astype(np.float32)

    except Exception as exc:

        logger.exception(
            "Gemini embedding generation failed."
        )

        raise RuntimeError(
            f"Embedding generation failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

class FAISSIndex:
    """
    Wraps faiss.IndexFlatIP with the associated DocumentChunk list.

    After L2 normalization, IndexFlatIP gives cosine similarity.
    Results are filtered using a relevance threshold.
    """

    def __init__(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray
    ):

        if not chunks:
            raise ValueError(
                "Cannot build an index with zero chunks."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional array."
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs "
                f"{embeddings.shape[0]} embeddings."
            )

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"received {embeddings.shape[1]}."
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "Embeddings contain invalid numerical values."
            )

        self.chunks = chunks

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)

        self.index.add(
            embeddings.astype(np.float32)
        )

        logger.info(
            "FAISS index built: %d vectors, dimension %d.",
            len(chunks),
            dim
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_RESULTS,
        threshold: float = RELEVANCE_THRESHOLD
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve the most relevant document chunks.

        Returns:
            List of (DocumentChunk, similarity_score) pairs.
        """

        if self.index.ntotal == 0:
            return []

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "Similarity threshold must be between -1 and 1."
            )

        query_vec = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query_vec.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension does not match FAISS index."
            )

        if not np.all(np.isfinite(query_vec)):
            raise ValueError(
                "Query embedding contains invalid values."
            )

        # Do not request more results than indexed vectors.
        top_k = min(top_k, self.index.ntotal)

        similarities, indices = self.index.search(
            query_vec,
            top_k
        )

        results = []

        for sim, idx in zip(
            similarities[0],
            indices[0]
        ):

            if idx == -1:
                continue

            if float(sim) < threshold:
                continue

            results.append(
                (self.chunks[idx], float(sim))
            )

        return results


def build_faiss_index(
    chunks: List[DocumentChunk]
) -> FAISSIndex:
    """
    Generate Gemini embeddings and build a FAISS index.
    """

    if not chunks:
        raise ValueError(
            "No text chunks to index. "
            "Please upload a valid PDF with extractable text."
        )

    texts = [
        chunk.text for chunk in chunks
    ]

    # Generate document embeddings.
    embeddings = embed_texts(
        texts,
        task_type="RETRIEVAL_DOCUMENT"
    )

    # Build FAISS index.
    try:

        return FAISSIndex(
            chunks,
            embeddings
        )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to build FAISS index: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Gemini LLM integration
# ---------------------------------------------------------------------------

def call_gemini(prompt: str) -> str:
    """
    Send a RAG prompt to Gemini and return the response text.

    Retries temporary 503 errors once.
    """

    client = get_gemini_client()

    max_retries = 2

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1200
                )
            )

            if response.text:
                return response.text.strip()

            if response.candidates:

                candidate = response.candidates[0]

                finish_reason = getattr(
                    candidate,
                    "finish_reason",
                    None
                )

                if finish_reason:
                    return (
                        "Response could not be generated "
                        f"(finish reason: {finish_reason}). "
                        "Please rephrase your question."
                    )

            return "No response generated. Please try again."

        except Exception as exc:

            err = str(exc).lower()

            # Retry temporary server errors.
            if "503" in err or "unavailable" in err:

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    logger.warning(
                        "Gemini temporarily unavailable. "
                        "Retrying in %d seconds...",
                        wait_time
                    )

                    time.sleep(wait_time)

                    continue

                raise RuntimeError(
                    "Gemini is temporarily experiencing high demand. "
                    "Please try again in a few minutes."
                ) from exc

            # Handle quota / rate limits.
            if (
                "quota" in err
                or "rate" in err
                or "429" in err
            ):

                raise RuntimeError(
                    "Gemini API quota or rate limit exceeded. "
                    "Please wait and try again."
                ) from exc

            # Handle invalid API keys.
            if (
                "api_key" in err
                or "api key" in err
                or "401" in err
                or "403" in err
            ):

                raise EnvironmentError(
                    "Gemini API key is invalid or unauthorized. "
                    "Please check your GEMINI_API_KEY."
                ) from exc

            # Handle network errors.
            if (
                "network" in err
                or "connection" in err
                or "timeout" in err
            ):

                raise RuntimeError(
                    "Network error while contacting Gemini API. "
                    "Please try again."
                ) from exc

            logger.exception(
                "Unexpected Gemini API error."
            )

            raise RuntimeError(
                f"Gemini API error: {exc}"
            ) from exc

    raise RuntimeError(
        "Gemini request failed after retries."
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    retrieved_chunks: List[Tuple[DocumentChunk, float]]
) -> str:
    """
    Construct the RAG prompt using retrieved manual context.
    """

    context_parts = []

    for chunk, score in retrieved_chunks:

        context_parts.append(
            f"[Source: {chunk.source}, Page {chunk.page}]\n"
            f"{chunk.text}"
        )

    context_text = "\n\n---\n\n".join(
        context_parts
    )

    return (
        "CONTEXT (extracted from uploaded appliance manuals):\n\n"
        f"{context_text}\n\n"
        "---\n\n"
        f"QUESTION: {question}\n\n"
        "Answer the question using only the context above. "
        "If the context does not contain the answer, "
        "say so explicitly."
    )


# ---------------------------------------------------------------------------
# Source citations
# ---------------------------------------------------------------------------

def extract_unique_sources(
    retrieved_chunks: List[Tuple[DocumentChunk, float]]
) -> List[dict]:
    """
    Build a de-duplicated list of source citations.
    """

    seen = set()
    sources = []

    for chunk, score in retrieved_chunks:

        key = (
            chunk.source,
            chunk.page
        )

        if key not in seen:

            seen.add(key)

            sources.append({
                "source": chunk.source,
                "page": chunk.page,
                "similarity": round(score, 3)
            })

    return sorted(
        sources,
        key=lambda x: x["similarity"],
        reverse=True
    )


# ---------------------------------------------------------------------------
# Main RAG query function
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    faiss_index: FAISSIndex,
    top_k: int = TOP_K_RESULTS,
    threshold: float = RELEVANCE_THRESHOLD
) -> Tuple[str, List[dict], bool]:
    """
    Full RAG pipeline:

    1. Embed user question using Gemini.
    2. Retrieve relevant chunks from FAISS.
    3. Check relevance threshold.
    4. Generate grounded answer using Gemini.
    5. Extract source citations.

    Returns:
        answer, sources, found
    """

    if not isinstance(question, str):
        raise ValueError(
            "Question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if faiss_index is None:
        raise ValueError(
            "FAISS index is not available. Upload a manual first."
        )

    # Step 1: Generate query embedding.
    q_embedding = embed_texts(
        [question],
        task_type="RETRIEVAL_QUERY"
    )[0]

    # Step 2: Retrieve relevant chunks.
    retrieved = faiss_index.search(
        q_embedding,
        top_k=top_k,
        threshold=threshold
    )

    # Step 3: Check whether relevant context exists.
    if not retrieved:

        return (
            "The uploaded manuals do not contain sufficient "
            "information to answer this question. Please ensure "
            "you have uploaded the correct appliance manual, "
            "or try rephrasing your question.",
            [],
            False
        )

    # Step 4: Build grounded prompt.
    prompt = build_prompt(
        question,
        retrieved
    )

    # Step 5: Generate answer.
    answer = call_gemini(prompt)

    # Step 6: Extract citations.
    sources = extract_unique_sources(
        retrieved
    )

    return answer, sources, True
