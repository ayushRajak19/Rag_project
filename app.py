import os
import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question

# set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("🦙 Llama-3.3-70B - Document RAG")

uploaded_file = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_file:
    for uploaded_files in uploaded_file:
        save_path = os.path.join(working_dir, uploaded_files.name)

        # ✅ FIXED LINE
        with open(save_path, "wb") as f:
            f.write(uploaded_files.getbuffer())

        process_document_to_chroma_db(uploaded_files.name)

    st.info("Document(s) Processed Successfully")

# text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer = answer_question(user_question)

    st.markdown("### Llama-3.3-70B Response")
    st.markdown(answer)
