#load document
import os 

from langchain_community.document_loaders import PyMuPDFLoader 
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
pdf_paths = [
    os.path.join(DATA_DIR, "bihar01.pdf"),
    os.path.join(DATA_DIR, "bihar02.pdf"),
    os.path.join(DATA_DIR, "bihar03.pdf"),
    os.path.join(DATA_DIR, "bihar04.pdf"),
    os.path.join(DATA_DIR, "bihar05.pdf"),
    os.path.join(DATA_DIR, "bihar06.pdf"),
    os.path.join(DATA_DIR, "bihar07.pdf"),
    os.path.join(DATA_DIR, "bihar08.pdf"),
    os.path.join(DATA_DIR, "bihar09.pdf"),
    os.path.join(DATA_DIR, "bihar10.pdf"),
    os.path.join(DATA_DIR, "bihar11.pdf"),
]

docs = []
for path in pdf_paths:
    if os.path.exists(path):
        print(f"Loading {path}...")
        loader = PyMuPDFLoader(path)
        docs.extend(loader.load())
    else:
        print(f"Warning: {path} not found.")

print(f"Total documents loaded: {len(docs)}")

#split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

#connect to llm
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="conversational",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)
print("Connected to LLM.")


#embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embeddings model ready.")


#load chroma
from langchain_chroma import Chroma
vectorstore = Chroma(persist_directory="./.chroma_db", embedding_function=embeddings)
print("Loaded Chroma.")

