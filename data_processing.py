import fitz
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import Counter

def extract_pdf_text(file):
    # Read PDF using PyMuPDF
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text() + "\n"

    return text

def gen_chunks(text, chunk_size=300, overlap=1):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) <= chunk_size:
            current_chunk.append(sentence)
            current_length += len(words)
        else:
            chunks.append(" ".join(current_chunk))
            # Start new chunk with `overlap` previous sentences
            current_chunk = current_chunk[-overlap:] + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)

    # Append last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

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

def calculate_keyword_score(chunk,keywords):
    words = re.findall(r"\w+",chunk.lower())
    freq = Counter(words)
    return sum([freq[kw] for kw in keywords])

def get_keywords(query):
    # Get a list of all English stopwords
    stop_words = set(stopwords.words('english'))
    # Tokenize the query into words
    tokens = word_tokenize(query)
    # Convert all words to lowercase and remove stopwords and punctuation
    keywords = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return keywords

def get_keyword_chunks(query, chunks, k=3):
    keywords = get_keywords(query)
    ranked_chunks = [(calculate_keyword_score(chunk,keywords),chunk) for chunk in chunks]
    ranked_chunks.sort(reverse=True)
    return ranked_chunks[:min(k,len(chunks))]

def cosine_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_semantic_chunks(chunk_embeddings,query_encoding,k=3):
    ranked_chunks = [(cosine_sim(query_encoding,embedding['embedding']),embedding) for embedding in chunk_embeddings]
    ranked_chunks.sort(reverse=True)

    return ranked_chunks[:min(k,len(chunk_embeddings))]

def return_top_2k_chunks(query,query_encoding,chunks,chunk_embeddings,k=3):
    semantic_chunks = [(sim,embedding['chunk']) for sim,embedding in get_semantic_chunks(chunk_embeddings,query_encoding,k)]
    keyword_chunks = get_keyword_chunks(query,chunks,k)

    keyword_scores = np.array([x for x,_ in keyword_chunks])
    keyword_scores = keyword_scores/np.max(keyword_scores) if np.max(keyword_scores)!=0 else keyword_scores
    keyword_chunks = [(score,chunk) for score,(_,chunk) in zip(keyword_scores,keyword_chunks)]

    best = sorted(keyword_chunks+semantic_chunks,reverse=True)
    return best[:min(2*k,len(best))]

def get_answer(query, chunks, embeddings, model, client):
    best_chunks = return_top_2k_chunks(query, model.encode(query, normalize_embeddings=True),chunks, embeddings)
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
