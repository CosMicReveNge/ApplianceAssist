"""
app.py
------
Streamlit frontend for ApplianceAssist AI.

Architecture:
    - Session state stores the FAISS index and chat history.
    - PDFs are processed only when new files are uploaded (not on every rerun).
    - The embedding model is loaded once at startup via get_embedding_model().
    - Gemini is called only after relevant context is retrieved.

Session state keys:
    faiss_index        : FAISSIndex | None
    processed_files    : set of filenames already indexed
    chat_history       : list of {"role": str, "content": str, "sources": list}
    total_chunks       : int – total chunks in current index
    processing_errors  : list[str]
    processing_warnings: list[str]
"""

import logging
import sys
import streamlit as st

# Configure root logger early so module loggers inherit this config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Import project modules (after logging setup)
from pdf_processor import process_uploaded_pdf, DocumentChunk
from rag_pipeline import (
    build_faiss_index,
    answer_question,
    FAISSIndex,
)
# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ApplianceAssist AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**ApplianceAssist AI** – A RAG-powered assistant for home "
            "appliance manuals. Upload your PDF manual and ask questions!"
        ),
    },
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, modern look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── General layout ── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* ── App title ── */
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #5f6368;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    /* ── Chat bubbles ── */
    .chat-user {
        background: #e8f0fe;
        border-left: 4px solid #1a73e8;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 8px;
        margin-bottom: 0.5rem;
    }
    .chat-assistant {
        background: #f8f9fa;
        border-left: 4px solid #34a853;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 8px;
        margin-bottom: 0.5rem;
    }
    /* ── Source citation card ── */
    .source-card {
        background: #fff8e1;
        border: 1px solid #f9a825;
        border-radius: 6px;
        padding: 0.4rem 0.75rem;
        font-size: 0.85rem;
        color: #5f4000;
        display: inline-block;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    /* ── Status badges ── */
    .badge-success {
        background: #e6f4ea;
        color: #137333;
        border: 1px solid #34a853;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    /* ── Divider ── */
    hr.thin { border: none; border-top: 1px solid #e0e0e0; margin: 1rem 0; }
    /* ── Hide Streamlit default hamburger menu ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialise all session state keys to their defaults if not present."""
    defaults = {
        "faiss_index": None,        # FAISSIndex | None
        "processed_files": set(),   # set of filenames already in the index
        "all_chunks": [],           # flat list of all DocumentChunks indexed
        "chat_history": [],         # list of message dicts
        "total_chunks": 0,
        "processing_errors": [],
        "processing_warnings": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# PDF processing helpers
# ---------------------------------------------------------------------------
def process_new_files(uploaded_files) -> None:
    """
    Process newly uploaded PDF files that are not yet in the index.

    Only processes files whose names are not already in
    st.session_state.processed_files, avoiding duplicate work on reruns.

    Updates st.session_state in place:
        - all_chunks
        - processed_files
        - faiss_index
        - total_chunks
        - processing_errors
        - processing_warnings
    """
    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state.processed_files
    ]

    if not new_files:
        return  # Nothing new to process

    new_chunks: list = []
    errors: list = []
    warnings: list = []

    progress = st.progress(0, text="Processing documents…")

    for i, uploaded_file in enumerate(new_files):
        progress.progress(
            (i + 1) / len(new_files),
            text=f"Processing: {uploaded_file.name}",
        )
        file_bytes = uploaded_file.read()
        chunks, w, e = process_uploaded_pdf(file_bytes, uploaded_file.name)

        errors.extend(e)
        warnings.extend(w)

        if chunks:
            new_chunks.extend(chunks)
            st.session_state.processed_files.add(uploaded_file.name)

    progress.empty()

    if new_chunks:
        # Combine with existing chunks and rebuild the index
        all_chunks = st.session_state.all_chunks + new_chunks
        with st.spinner("Building search index…"):
            try:
                st.session_state.faiss_index = build_faiss_index(all_chunks)
                st.session_state.all_chunks = all_chunks
                st.session_state.total_chunks = len(all_chunks)
            except (ValueError, RuntimeError) as exc:
                errors.append(str(exc))

    st.session_state.processing_errors = errors
    st.session_state.processing_warnings = warnings


# ---------------------------------------------------------------------------
# Chat rendering helpers
# ---------------------------------------------------------------------------
def render_sources(sources: list) -> None:
    """Render source citation cards below an assistant message."""
    if not sources:
        return
    st.markdown("**📎 Sources:**")
    cols = st.columns(min(len(sources), 3))
    for col, src in zip(cols, sources):
        with col:
            st.markdown(
                f'<div class="source-card">📄 {src["source"]}<br/>'
                f'Page {src["page"]} &nbsp;|&nbsp; '
                f'Relevance: {src["similarity"]:.0%}</div>',
                unsafe_allow_html=True,
            )


def render_chat_history() -> None:
    """Render the full chat history from session state."""
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                if sources:
                    render_sources(sources)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Render the sidebar: uploads, document status, and app info."""
    with st.sidebar:
        st.markdown("## 📂 Upload Manuals")
        st.markdown(
            "Upload one or more PDF appliance manuals. "
            "The assistant will answer questions based on their content."
        )

        uploaded_files = st.file_uploader(
            label="Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files. Scanned-image-only PDFs will not work.",
            key="file_uploader",
        )

        if uploaded_files:
            process_new_files(uploaded_files)

        # ── Error / warning display ──
        if st.session_state.processing_errors:
            for err in st.session_state.processing_errors:
                st.error(f"❌ {err}")

        if st.session_state.processing_warnings:
            with st.expander("⚠️ Warnings"):
                for warn in st.session_state.processing_warnings:
                    st.warning(warn)

        # ── Document status ──
        st.markdown("<hr class='thin'>", unsafe_allow_html=True)
        st.markdown("### 📋 Indexed Documents")

        if st.session_state.processed_files:
            st.markdown(
                f'<span class="badge-success">✅ '
                f'{len(st.session_state.processed_files)} document(s) | '
                f'{st.session_state.total_chunks} text chunks</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            for fname in sorted(st.session_state.processed_files):
                st.markdown(f"📄 `{fname}`")
        else:
            st.info("No documents indexed yet. Upload a PDF to get started.")

        # ── Clear everything ──
        st.markdown("<hr class='thin'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All & Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        # ── App info ──
        st.markdown("<hr class='thin'>", unsafe_allow_html=True)
        st.markdown(
            """
            <small>
            <b>ApplianceAssist AI</b><br/>
            Powered by Google Gemini + FAISS<br/>
            Embedding: all-MiniLM-L6-v2<br/>
            </small>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
def render_main() -> None:
    """Render the main chat interface."""

    # ── Header ──
    st.markdown(
        '<h1 class="app-title">🏠 ApplianceAssist AI</h1>'
        '<p class="app-subtitle">'
        'Ask questions about your home appliance manuals. '
        'Upload a PDF in the sidebar to get started.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Welcome message (shown only when chat is empty) ──
    if not st.session_state.chat_history:
        st.info(
            "👋 **Welcome to ApplianceAssist AI!**\n\n"
            "I can help you with:\n"
            "- 🔧 Troubleshooting error codes and problems\n"
            "- 🧹 Cleaning and maintenance schedules\n"
            "- ⚙️ Understanding specific features and modes\n"
            "- ⚠️ Safety precautions\n"
            "- 📐 Installation instructions\n\n"
            "**To get started:** Upload your appliance manual PDF in the sidebar."
        )

        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "Why is my washing machine displaying error E3?",
            "How often should I clean my refrigerator filter?",
            "What does the eco mode do and how do I activate it?",
            "How do I troubleshoot a microwave that is not heating?",
            "What safety precautions should I take before cleaning the appliance?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    handle_question(q)
                    st.rerun()

    # ── Chat history ──
    render_chat_history()

    # ── Clear chat button (only when there's a history) ──
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Input area ──
    with st.form(key="question_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="Your question",
                placeholder="e.g. What does error code E5 mean?",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button(
                "Ask ➤", use_container_width=True, type="primary"
            )

    if submitted:
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter a question before submitting.")
        else:
            handle_question(user_input.strip())
            st.rerun()


# ---------------------------------------------------------------------------
# Question handling
# ---------------------------------------------------------------------------
def handle_question(question: str) -> None:
    """
    Handle a user question through the full RAG pipeline.

    1. Validate prerequisites (documents loaded, API key present).
    2. Call answer_question() from rag_pipeline.
    3. Append question and answer to chat history.
    4. Display user-friendly error messages on failure.
    """
    # Guard: no documents indexed yet
    if st.session_state.faiss_index is None:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "⚠️ No documents have been indexed yet. "
                "Please upload one or more PDF appliance manuals using the "
                "sidebar before asking questions."
            ),
            "sources": [],
        })
        return

    # Append user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    # Run RAG pipeline
    with st.spinner("🔍 Searching manuals and generating answer…"):
        try:
            answer, sources, found = answer_question(
                question=question,
                faiss_index=st.session_state.faiss_index,
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except EnvironmentError as exc:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": (
                    f"🔑 **API Key Error:** {exc}\n\n"
                    f"Please ensure your `GEMINI_API_KEY` is correctly set."
                ),
                "sources": [],
            })

        except RuntimeError as exc:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ **Error:** {exc}",
                "sources": [],
            })

        except Exception as exc:
            logger.exception("Unexpected error handling question: %s", question)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": (
                    "❌ An unexpected error occurred. "
                    "Please refresh the page and try again. "
                    f"Details: {exc}"
                ),
                "sources": [],
            })




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()