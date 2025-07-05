from Data_Vectorization import load_vector_store

# Load the vector store
db = load_vector_store()

# Query with any dummy keyword
retriever = db.as_retriever()
docs = retriever.get_relevant_documents("test")

# Inspect metadata
for i, doc in enumerate(docs):
    print(f"--- Document {i+1} ---")
    print("Content:", doc.page_content[:100], "...")  # Preview text
    print("Filename:", doc.metadata.get("filename", "Not found"))
    print("Source:", doc.metadata.get("source", "Not found"))
    print("Page:", doc.metadata.get("page", "Not found"))
    print("-----------------------")
