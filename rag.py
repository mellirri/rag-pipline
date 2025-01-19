import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

# Step 1: Preprocess Lecture Notes
def process_lecture_notes(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)
    return chunks

# Step 2: Build Vector Index
def build_vector_index(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("vector_store")
    return vector_store

# Step 3: Load Vector Store
def load_vector_store(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# Step 4: Query Ollama API
def query_ollama(question, context):
    url = "http://localhost:11434/api/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "mistral",
        "prompt": f"Context: {context}\\n\\nQuestion: {question}\\nAnswer:"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("text", "Нет ответа")
    else:
        raise Exception(f"Error from Ollama API: {response.status_code}, {response.text}")

# Step 5: Create RAG API
def create_app():
    app = FastAPI()
    vector_store = load_vector_store()

    @app.get("/query")
    async def query(question: str):
        try:
            docs = vector_store.similarity_search(question, k=5)
            context = "\\n".join([doc.page_content for doc in docs])
            answer = query_ollama(question, context)
            return JSONResponse(content={"answer": answer, "context": context})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

if __name__ == "__main__":
    lecture_file = "lectures.txt"
    lecture_chunks = process_lecture_notes(lecture_file)
    build_vector_index(lecture_chunks)
    app = create_app()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
