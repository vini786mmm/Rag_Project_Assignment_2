import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the directory for the vector store
CHROMA_PATH = "VectorDB"

def create_and_save_vector_store(chunked_docs):
    """
    Creates a Chroma vector store from document chunks, embeds them, and saves to disk.

    Args:
        chunked_docs (list): A list of document chunks with metadata.
    """
    if not chunked_docs:
        st.warning("No document chunks to process. Vector store not created.")
        return

    with st.spinner("Creating embeddings and saving vector store... This may take a moment."):
        try:
            # ✅ Initialize the embedding model on CPU to avoid meta tensor issues
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )

            # ✅ Create the vector store and persist to disk
            db = Chroma.from_documents(
                documents=chunked_docs,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
                collection_name="documents"
            )

            db.persist()
            st.success("✅ Vector store created and saved successfully!")

        except Exception as e:
            st.error(f"❌ An error occurred during vector store creation:\n{str(e)}")

def load_vector_store():
    """
    Loads an existing Chroma vector store from disk.

    Returns:
        Chroma: The loaded Chroma vector store instance.
    """
    try:
        # ✅ Ensure consistent device for embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # ✅ Load Chroma vector store
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="documents"
        )
        return db

    except Exception as e:
        st.error(f"❌ Failed to load vector store: {str(e)}")
        return None
