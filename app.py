import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from file_processing import *

if 'model' not in st.session_state:
# Load environment variable
    load_dotenv()
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    st.session_state.client = genai.Client()
    st.session_state.doc_read = False
    st.session_state.uploaded_file = None

st.title("Document QA with Gemini")

if st.session_state.uploaded_file is None:
    st.session_state.uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])
query = st.text_input("Ask a question")

if st.button("Reset"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# After uploading the file
if st.session_state.uploaded_file is not None:
    if not st.session_state.doc_read:
        with st.spinner("Reading and processing..."):
            if st.session_state.uploaded_file.type == "text/plain":
                st.session_state.file_text = st.session_state.uploaded_file.read().decode("utf-8")
            elif st.session_state.uploaded_file.type == "application/pdf":
                st.session_state.file_text = extract_pdf_text(st.session_state.uploaded_file)

            st.session_state.chunk_embeddings = embed_chunks(
                gen_chunks(st.session_state.file_text), st.session_state.model
            )

        st.success("File processed!")

        # with st.spinner("Summarizing..."):
        #     summary = summarize_text(st.session_state.file_text, st.session_state.client)
        #     st.session_state.summary = summary
        st.session_state.doc_read = True
        # # Show summary
        
        # st.markdown(f"**Summary:** {st.session_state.summary}")

    # Handle query
    if query:
        with st.spinner("Thinking..."):
            answer = get_answer(query, st.session_state.chunk_embeddings, st.session_state.model, st.session_state.client)
            # best_chunks = return_top_k_chunks(query,st.session_state.chunk_embeddings,st.session_state.model)
            # answer = f'Query: {query}\n'
            # for sim, chunk_info in best_chunks:
            #     answer += f'Similarity: {sim}\nChunk: {chunk_info['chunk']}'
            #     answer += '\n'
            st.markdown(f"**Answer:** {answer}")

        

