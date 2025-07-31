from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DB_DIR = "faiss_index"
METADATA_FILE = "metadata.json"

def extract_text(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def process_documents(file_paths):
    all_chunks = []
    metadata = []
    for file in file_paths:
        text = extract_text(file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.append({"file": os.path.basename(file), "chunks": len(chunks)})

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
    vector_store.save_local(DB_DIR)

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def get_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            return json.load(f)
    return []

def answer_query(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question, k=5)

    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]
