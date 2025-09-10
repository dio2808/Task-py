from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

# 1. Load FAISS index with Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 2. Initialize local LLM (example: llama3)
llm = ChatOllama(model="llama3", temperature=0.2)

# 3. Setup RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
)

# 4. Ask a question
query = input("Ask your question: ")
result = qa.run(query)

print("\n Answer:", result)
