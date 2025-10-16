# app.py
import os
import uuid
import tempfile
import json
import re
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pymongo import MongoClient
import google.generativeai as genai

from doc_processor import process_file_to_qdrant

# ---------------- CONFIG ----------------
GEMINI_API_KEY = "<your-gemini-api-key>"
QDRANT_API_KEY = "<your-qdrant-api-eky>"
QDRANT_URL = "<your-qdrant-connection-url>"

COLLECTION_NAME = "<your-qdrant-collection-name>"
TOP_K = 5

MONGO_URI = "<your-mongodb-connection-uri>"
MONGO_DB = "<your-mongodb-database-name>"
MONGO_COLLECTION = "<your-mongodb-collection-name>"   # for chats
MONGO_DOCS_COLLECTION = "<your-mongodb-collection-name>"  # new collection for storing file metadata

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)

# Qdrant setup
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# MongoDB setup
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
chat_collection = db[MONGO_COLLECTION]
docs_collection = db[MONGO_DOCS_COLLECTION]

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Ensure Qdrant collection
try:
    qdrant_client.get_collection(COLLECTION_NAME)
except Exception:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

# Ensure index on metadata.file_name for filtering
try:
    qdrant_client.recreate_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.file_name",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print("Index on metadata.file_name created successfully.")
except Exception as e:
    print("Error creating Qdrant index:", e)

# ---------------- HELPERS ----------------
def search_conversation(session_id: str):
    records = list(chat_collection.find(
        {"session_id": session_id},
        {"eng_query": 1, "eng_ans": 1, "_id": 0}
    ).sort("_id", -1).limit(10))

    if records:
        return "\n".join([f"Q: {rec['eng_query']}\nA: {rec['eng_ans']}" for rec in reversed(records)])
    return ""


def get_date_day():
    today = datetime.today()
    return today.strftime("%d-%m-%Y"), today.strftime("%A")


def search_qdrant(query_text: str):
    query_vector = embedding_model.encode(query_text).tolist()
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K,
    )
    return [hit.payload["text"] for hit in search_result]


def ask_gemini(raw_query, eng_query, context_chunks):
    context = "\n\n".join(context_chunks)
    model_ans = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=f"""
You are an advanced AI model that performs three tasks based on the provided input:

1. Understand the provided context chunks, which contain relevant knowledge and prior information.
2. Generate a clear and concise answer in proper English for the given 'eng_query' using the context chunks.
3. Format the final answer so that it mirrors the language and style of the 'raw_query'.  
   - If the raw_query is in User's Language (eg. Hindi) written in English transliteration, the final answer must also be in User's Language written in English transliteration.
   - If the raw_query is in User's Language (eg. Hindi) texted in User's Language scripts only, the final answer must also be in User's Language scripts only.
   - If the raw_query is in English, the final answer must be in English.

Only output a JSON object with exactly these three fields:
1. eng_answer: The answer generated in clear and concise English, derived from the context_chunks.
2. final_answer: The answer formatted in the same style and language as the 'raw_query'.

Do not output anything else. Example format:
{{
  "eng_answer": "Shanti's age is 25 years.",
  "final_answer": "Shanti ki umar 25 saal hai (or) शांति की उम्र 25 साल है (depending the raw query format, language, and texting)"
}}

Inputs:
Raw Query: {raw_query}
English Query: {eng_query}
Context Chunks:
{context_chunks}
""")

    response = model_ans.generate_content(
        contents=[{"role": "user", "parts": [f"{raw_query}\n\n{eng_query}\n\n{context_chunks}"]}]
    )

    clean_response = re.sub(r"```[a-z]*", "", response.text.strip()).replace("```", "").strip()
    result = json.loads(clean_response)

    eng_ans = result["eng_answer"]
    ans = result["final_answer"]

    return eng_ans, ans


def make_query(raw_query, conversational_context, current_date, current_day):
    model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=f"""
You are an advanced AI model that takes a user query, the previous conversation context, the current date, and the current day.  
Your task is to:
1. Detect the language of the user query.
2. Rewrite the query completely in proper, context-aware English, resolving any indirect references to the previous conversational context.
3. If the query refers to relative time expressions such as "yesterday", "tomorrow", "day after tomorrow", "next week", etc., use the provided current date and day to calculate the exact upcoming date and day mentioned in the query.
   - Provide the calculated date in both 'DD-MM-YYYY' and '15 Aug 2025' formats.
   - Incorporate these dates clearly into the rewritten query to enable accurate data retrieval from the knowledge base.

The input query may contain multiple languages written in English transliteration.

Previous conversational context:
{conversational_context}

Current Date (DD-MM-YYYY): {current_date}
Current Day: {current_day}

User Query:
{raw_query}

As output, give only the properly rewritten, clear, and context-aware English query, with exact dates and days replacing relative time expressions, suitable for direct use in data retrieval.
""")

    response = model.generate_content(
        contents=[{"role": "user", "parts": [f"{raw_query}\n\n{current_date}\n\n{current_day}\n\n{conversational_context}"]}]
    )

    return response.text.strip()


def save_record(session_id, raw_query, eng_query, eng_ans, ans):
    new_record = {
        "session_id": session_id,
        "raw_query": raw_query,
        "eng_query": eng_query,
        "eng_ans": eng_ans,
        "ans": ans,
    }
    chat_collection.insert_one(new_record)


# ---------------- ENDPOINTS ----------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Process file into Qdrant
    result = process_file_to_qdrant(
        tmp_path, 
        collection_name=COLLECTION_NAME, 
        doc_id=doc_id, 
        table_data=False, 
        file_name=file.filename
    )

    # Save metadata into docs collection
    docs_collection.insert_one({
        "_id": doc_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "qdrant_result": result,
        "uploaded_at": datetime.utcnow()
    })

    os.unlink(tmp_path)

    return {"success": True, "doc_id": doc_id, "result": result}


@app.post("/chat")
async def chat_endpoint(data: dict = Body(...)):
    raw_query = data.get("message")
    session_id = data.get("session_id", "default")

    if not raw_query:
        return {"success": False, "error": "Message is required"}

    conversation_context = search_conversation(session_id)
    current_date, current_day = get_date_day()

    eng_query = make_query(raw_query, conversation_context, current_date, current_day)
    context_chunks = search_qdrant(eng_query)

    eng_ans, ans = ask_gemini(raw_query, eng_query, context_chunks)

    save_record(session_id, raw_query, eng_query, eng_ans, ans)

    return {
        "success": True,
        "session_id": session_id,
        "eng_query": eng_query,
        "eng_ans": eng_ans,
        "answer": ans,
        "context_used": context_chunks,
    }


from typing import List

@app.get("/files", response_model=List[str])
def list_files():
    """Fetch all unique file names processed so far"""
    files = docs_collection.find({}, {"filename": 1, "_id": 0})
    file_names = [doc["filename"] for doc in files if "filename" in doc]
    return file_names


@app.delete("/delete/{filename}")
def delete_file(filename: str):
    """Delete a file by filename from MongoDB + Qdrant chunks"""

    # 1. Delete from MongoDB
    result = docs_collection.delete_one({"filename": filename})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in MongoDB")


    return {"success": True, "message": f"File '{filename}' deleted from MongoDB and Qdrant"}
