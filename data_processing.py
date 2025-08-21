import fitz
import numpy as np

def extract_pdf_text(file):
    # Read PDF using PyMuPDF
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text() + "\n"

    return text

def gen_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    current_index = 0

    while current_index < len(words):
        # Get the chunk
        end_index = current_index + chunk_size
        chunk = " ".join(words[current_index:end_index])

        # Store
        chunks.append(chunk)

        # Move index forward with overlap
        current_index += chunk_size - overlap

    return chunks

def embed_chunks(chunks, model):
    chunk_embeddings = [{
        'chunk': chunk,
        'embedding': model.encode(chunk, normalize_embeddings=True),
        'meta': {
            'chapter': 1
        }
    } for chunk in chunks]

    return chunk_embeddings

def cosine_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def return_top_k_chunks(query,embeddings,model,k=3):
    query_embedding = model.encode(query)
    ranked_chunks = [(cosine_sim(query_embedding,embedding['embedding']),embedding) for embedding in embeddings]
    ranked_chunks.sort(reverse=True)

    return ranked_chunks[:min(k,len(embeddings))]

def get_answer(query, embeddings, model, client):
    best_chunks = return_top_k_chunks(query, embeddings, model)
    context = "\n".join(chunk['chunk'] for _, chunk in best_chunks)
    prompt = f"""
    You're an AI assistant helping users understand documents.

    Use the information below to answer the question clearly and conversationally.

    Context:
    {context}

    Question:
    {query}

    Answer in simple, friendly language. Be concise and accurate. Only use information from the context. If the answer isn't in the context, say so politely.
    """
    response = client.models.generate_content(
    model='gemini-2.5-flash', contents = prompt
    )

    return response.text

def summarize_text(text, client):
    prompt = f"""
    Summarize the following text
    
    {text}

    and prompt the user to ask any doubts if they have."""

    response = client.models.generate_content(
    model='gemini-2.5-flash', contents = prompt
    )

    return response.text
