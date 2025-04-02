import json
from django.conf import settings
import faiss
import numpy as np
import pymupdf 
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os


GENAI_API_KEY = "AIzaSyB1gi_E8yTKMW5aoeYJr1LGl0dGs9y7EoI" # Gemini api keys
genai.configure(api_key=GENAI_API_KEY)

index = faiss.IndexFlatL2(768)  # Adjust dimensions based on embedding model
embedding_data = {}  # Dictionary to store document mappings

def chat_ui(request):
    """Render the chat UI."""
    return render(request, "chats/chat.html")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = pymupdf.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    print(text)
    return text

def get_gemini_embedding(text):
    """Generate embeddings using Gemini AI."""
    try:
        response = genai.embed_content(
            model="models/embedding-001", 
            content=text, 
            task_type="retrieval_query"  # Use retrieval_query for queries, retrieval_document for documents
        )
        embedding = np.array(response["embedding"], dtype=np.float32)
        # Normalizing the embedding
        normalized_embedding = embedding / np.linalg.norm(embedding)
        return normalized_embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
    

def load_embeddings():
    global embedding_data
    try:
        # Due to resource issues i am extracting data from only one file of the folder, as gemini as limited free api calls
        pdf_path = os.path.join(settings.BASE_DIR, "static/pdfs/America's_Choice_2500_Gold_SOB (1) (1).pdf")
        pdf_text = extract_text_from_pdf(pdf_path)  
        text_chunks = [pdf_text[i : i + 500] for i in range(0, len(pdf_text), 500)]  # Split into chunks

        chunk_embeddings = []
        for i, chunk in enumerate(text_chunks):
            chunk_embedding = get_gemini_embedding(chunk)
            if chunk_embedding is None:
                continue
            embedding_data[i] = {"text": chunk, "vector": chunk_embedding.tolist()}
            chunk_embeddings.append(chunk_embedding)

        vectors = np.array(chunk_embeddings, dtype=np.float32)
        index.add(vectors)
        print(f"Loaded {len(text_chunks)} chunks into FAISS index.")

    except Exception as e:
        print(f"Error loading embeddings: {e}")


@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("query", "")
        
        if not query:
            return JsonResponse({"response": "Please enter a valid query."})
        
        # Convert query to embedding using Gemini
        query_vector = get_gemini_embedding(query)
        if query_vector is None:
            return JsonResponse({"response": "Failed to generate query embedding."})
        
        # Normalize vectors for better comparison
        query_vector = query_vector / np.linalg.norm(query_vector)
     
        # Always use the best match regardless of distance
        distances, indices = index.search(np.array([query_vector]), 1)
        print(f"Distance value: {distances[0][0]}")
        
        # Always use the best match 
        best_match_index = indices[0][0]
        if distances[0][0] < 0.8:  # Threshold for relevance
            best_match = embedding_data.get(best_match_index, {}).get("text", "I Don't know")
        else:
            best_match = "I Don't know"

        return JsonResponse({"response": best_match})
