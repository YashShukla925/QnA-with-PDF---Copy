# backend/main.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import fitz
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- Config ----------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DB_DIR = "faiss_index"
UPLOAD_DIR = "uploaded_docs"
METADATA_FILE = "metadata.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Multilingual Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

# ---------- Utility functions ----------
def extract_text(file_path):
    """Extract all page texts from PDF using PyPDF2"""
    reader = PdfReader(file_path)
    texts = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        texts.append((i, text))
    return texts

def extract_page_text_fitz(file_name: str, page: int):
    """Extract text from a specific page using PyMuPDF"""
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        return None

    doc = fitz.open(file_path)
    if page < 1 or page > len(doc):
        doc.close()
        return None

    page_obj = doc[page - 1]
    text = page_obj.get_text()
    doc.close()
    return text if text.strip() != "" else None

def chunk_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def process_documents(file_paths):
    all_chunks = []
    all_metadata = []

    for file in file_paths:
        file_name = os.path.basename(file)
        page_texts = extract_text(file)

        for page_num, text in page_texts:
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "file_name": file_name,
                    "page_number": page_num
                })

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings, metadatas=all_metadata)
    vector_store.save_local(DB_DIR)

    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f)

def get_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            return json.load(f)
    return []

def answer_query(question: str, lang: str = "en"):
    """Answer a question from the PDF corpus in a specific language"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question, k=5)

    prompt_template = f"""
    Use the context to answer the question in {lang}.
    If the answer is not in the context, say "Answer is not available in the context".

    Context:
    {{context}}

    Question:
    {{question}}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(gemini_model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    sources = [{"file": d.metadata["file_name"], "page": d.metadata["page_number"]} for d in docs]

    return {"answer": response["output_text"], "sources": sources}

def translate_text_gemini(text: str, target_lang: str = "hi"):
    """Translate any text to the target language using Gemini"""
    prompt = f"Translate the following text to {target_lang}:\n\n{text}"
    resp = gemini_model.invoke(prompt)
    return resp.content

# ---------- FastAPI App ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Upload PDFs
@app.post("/upload/")
async def upload_docs(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(path)

    process_documents(file_paths)
    return {"status": "success", "files": [file.filename for file in files]}

# Query PDFs
@app.get("/query/")
def query_docs(question: str, lang: str = "en"):
    return answer_query(question, lang)

# Metadata endpoint
@app.get("/metadata/")
def get_docs_metadata():
    return get_metadata()

# Get raw page text
@app.get("/page_text/")
async def page_text(file: str = Query(...), page: int = Query(...)):
    text = extract_page_text_fitz(file, page)
    if text is None:
        return {"error": "Invalid file or page"}
    return {"file": file, "page": page, "text": text}

# Translate page
@app.get("/translate_page/")
async def translate_page(file: str, page: int, target_lang: str = "hi"):
    raw_text = extract_page_text_fitz(file, page)
    if not raw_text:
        return {"error": "No text found on this page"}

    translated = translate_text_gemini(raw_text, target_lang)
    return {"translated": translated, "file": file, "page": page}

# Serve PDF
@app.get("/pdf/{file_name}")
async def get_pdf(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(file_path, media_type="application/pdf")
