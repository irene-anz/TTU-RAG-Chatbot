from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="vectordb", embedding_function=embedding)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})

docs = retriever.invoke("where is ttu located")
for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content)