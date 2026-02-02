import streamlit as st
from backend.pdf_loader import load_pdfs
from backend.text_splitter import split_text
from backend.vector_store import create_faiss_index, search_index
from backend.qa_engine import answer_question

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="PDF Chat Assistant",
    layout="wide"
)

st.title("📄 PDF Chat Assistant")
st.caption("Ask questions strictly based on uploaded PDFs")

# ------------------ SESSION STATE ------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.text_chunks = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📤 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📊 Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                documents = load_pdfs(uploaded_files)

                all_chunks = []
                for doc in documents:
                    all_chunks.extend(split_text(doc))

                index, embeddings, text_chunks = create_faiss_index(all_chunks)

                st.session_state.index = index
                st.session_state.text_chunks = text_chunks
                st.session_state.chat_history = []

                st.success("PDFs processed successfully!")

    st.divider()

    # 🧹 Clear Chat Button
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared successfully!")

# ------------------ CHAT DISPLAY ------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ CHAT INPUT ------------------
user_question = st.chat_input("Ask a question from the PDFs...")

if user_question:
    if st.session_state.index is None:
        st.warning("Please upload and process PDFs first.")
    else:
        # User message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Assistant response
        with st.spinner("Thinking..."):
            retrieved_chunks = search_index(
                st.session_state.index,
                user_question,
                st.session_state.text_chunks
            )

            answer = answer_question(user_question, retrieved_chunks)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(answer)
