from langchain_huggingface import HuggingFaceEmbeddings
print("Pre-downloading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Done!")