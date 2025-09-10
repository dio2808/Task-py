import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load PDFs
docs = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("data", file))
        docs.extend(loader.load())

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embeddings with Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Build FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Save locally
vectorstore.save_local("faiss_index")

print(" Vector store created ")
