"""
app.py
------
Streamlit frontend for ApplianceAssist AI.

Architecture:
    - Session state stores the FAISS index and chat history.
    - PDFs are processed only when new files are uploaded.
    - Gemini generates document and query embeddings remotely.
    - FAISS performs similarity search.
    - Gemini generates answers using retrieved manual context.
"""

import logging
import sys
import html
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Import project modules
from pdf_processor import process_uploaded_pdf
from rag_pipeline import build_faiss_index, answer_question


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
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

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

    .badge-success {
        background: #e6f4ea;
        color: #137333;
        border: 1px solid #34a853;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    hr.thin {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 1rem 0;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise session state variables."""

    defaults = {
        "faiss_index": None,
        "processed_files": set(),
        "all_chunks": [],
        "chat_history": [],
        "total_chunks": 0,
        "processing_errors": [],
        "processing_warnings": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------

def process_new_files(uploaded_files) -> None:
    """
    Process newly uploaded PDFs and rebuild the FAISS index.

    Files are marked as processed only after successful indexing.
    """

    new_files = [
        file
        for file in uploaded_files
        if file.name not in st.session_state.processed_files
    ]

    if not new_files:
        return

    new_chunks = []
    successfully_processed_files = []
    errors = []
    warnings = []

    progress = st.progress(0, text="Processing documents...")

    try:
        for i, uploaded_file in enumerate(new_files):

            progress.progress(
                (i + 1) / len(new_files),
                text=f"Processing: {uploaded_file.name}",
            )

            try:
                file_bytes = uploaded_file.getvalue()

                chunks, file_warnings, file_errors = process_uploaded_pdf(
                    file_bytes,
                    uploaded_file.name,
                )

                warnings.extend(file_warnings)
                errors.extend(file_errors)

                if chunks:
                    new_chunks.extend(chunks)
                    successfully_processed_files.append(uploaded_file.name)
                elif not file_errors:
                    warnings.append(
                        f"{uploaded_file.name}: No usable text chunks were found."
                    )

            except Exception as exc:
                logger.exception(
                    "Error processing uploaded file: %s",
                    uploaded_file.name,
                )
                errors.append(
                    f"{uploaded_file.name}: Failed to process PDF. {exc}"
                )

        if new_chunks:

            # Combine new chunks with previously indexed chunks.
            all_chunks = st.session_state.all_chunks + new_chunks

            with st.spinner("Generating embeddings and building search index..."):
                try:
                    new_index = build_faiss_index(all_chunks)

                    # Update session state only after successful index creation.
                    st.session_state.faiss_index = new_index
                    st.session_state.all_chunks = all_chunks
                    st.session_state.total_chunks = len(all_chunks)

                    st.session_state.processed_files.update(
                        successfully_processed_files
                    )

                    logger.info(
                        "Successfully indexed %d chunks.",
                        len(all_chunks),
                    )

                except Exception as exc:
                    logger.exception("Failed to build FAISS index.")
                    errors.append(f"Could not build search index: {exc}")

        st.session_state.processing_errors = errors
        st.session_state.processing_warnings = warnings

    finally:
        progress.empty()


# ---------------------------------------------------------------------------
# Source citation rendering
# ---------------------------------------------------------------------------

def render_sources(sources: list) -> None:
    """Display source citations below an assistant response."""

    if not sources:
        return

    st.markdown("**📎 Sources:**")

    cols = st.columns(min(len(sources), 3))

    for col, src in zip(cols, sources):

        source_name = html.escape(str(src.get("source", "Unknown")))
        page = src.get("page", "N/A")
        similarity = src.get("similarity", 0)

        with col:
            st.markdown(
                f"""
                <div class="source-card">
                    📄 {source_name}<br/>
                    Page {page} &nbsp;|&nbsp;
                    Relevance: {similarity:.0%}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Chat history rendering
# ---------------------------------------------------------------------------

def render_chat_history() -> None:
    """Display previous messages."""

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
    """Render PDF uploads, document status, and application information."""

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
            help="Select one or more PDF files. Scanned-image-only PDFs are not supported.",
            key="file_uploader",
        )

        if uploaded_files:
            process_new_files(uploaded_files)

        # Display processing errors
        if st.session_state.processing_errors:

            for error in st.session_state.processing_errors:
                st.error(f"❌ {error}")

        # Display processing warnings
        if st.session_state.processing_warnings:

            with st.expander("⚠️ Warnings"):

                for warning in st.session_state.processing_warnings:
                    st.warning(warning)

        # Document status
        st.markdown(
            "<hr class='thin'>",
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 Indexed Documents")

        if st.session_state.processed_files:

            st.markdown(
                f"""
                <span class="badge-success">
                    ✅ {len(st.session_state.processed_files)} document(s)
                    | {st.session_state.total_chunks} text chunks
                </span>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")

            for filename in sorted(st.session_state.processed_files):
                st.markdown(f"📄 `{filename}`")

        else:

            st.info(
                "No documents indexed yet. "
                "Upload a PDF to get started."
            )

        # Clear everything
        st.markdown(
            "<hr class='thin'>",
            unsafe_allow_html=True,
        )

        if st.button(
            "🗑️ Clear All & Reset",
            use_container_width=True,
        ):

            for key in list(st.session_state.keys()):
                del st.session_state[key]

            st.rerun()

        # Application information
        st.markdown(
            "<hr class='thin'>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <small>
            <b>ApplianceAssist AI</b><br/>
            Powered by Google Gemini + FAISS<br/>
            Embeddings: Gemini Embedding API<br/>
            </small>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Question handling
# ---------------------------------------------------------------------------

def handle_question(question: str) -> None:
    """Retrieve relevant manual content and generate an answer."""

    # Check whether a document has been indexed.
    if st.session_state.faiss_index is None:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "⚠️ No documents have been indexed yet. "
                "Please upload one or more PDF appliance manuals "
                "using the sidebar before asking questions."
            ),
            "sources": [],
        })

        return

    # Store the user's question.
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    # Run the RAG pipeline.
    with st.spinner("🔍 Searching manuals and generating answer..."):

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
                    "Please ensure your `GEMINI_API_KEY` is correctly configured."
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

            logger.exception(
                "Unexpected error handling question: %s",
                question,
            )

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
# Main content
# ---------------------------------------------------------------------------

def render_main() -> None:
    """Render the main chat interface."""

    # Header
    st.markdown(
        '<h1 class="app-title">🏠 ApplianceAssist AI</h1>'
        '<p class="app-subtitle">'
        'Ask questions about your home appliance manuals. '
        'Upload a PDF in the sidebar to get started.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr class='thin'>",
        unsafe_allow_html=True,
    )

    # Welcome screen
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

        for i, question in enumerate(sample_questions):

            with cols[i % 2]:

                if st.button(
                    question,
                    key=f"sample_{i}",
                    use_container_width=True,
                ):

                    handle_question(question)
                    st.rerun()

    # Previous messages
    render_chat_history()

    # Clear chat
    if st.session_state.chat_history:

        if st.button(
            "🗑️ Clear Chat",
            key="clear_chat",
        ):

            st.session_state.chat_history = []
            st.rerun()

    st.markdown(
        "<hr class='thin'>",
        unsafe_allow_html=True,
    )

    # Question input
    with st.form(
        key="question_form",
        clear_on_submit=True,
    ):

        col_input, col_btn = st.columns([5, 1])

        with col_input:

            user_input = st.text_input(
                label="Your question",
                placeholder="e.g. What does error code E5 mean?",
                label_visibility="collapsed",
            )

        with col_btn:

            submitted = st.form_submit_button(
                "Ask ➤",
                use_container_width=True,
                type="primary",
            )

    if submitted:

        if not user_input or not user_input.strip():

            st.warning("⚠️ Please enter a question before submitting.")

        else:

            handle_question(user_input.strip())
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Streamlit application."""

    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()