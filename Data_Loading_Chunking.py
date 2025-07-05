import os
import json
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def save_chunks_to_json(chunked_documents, output_path="chunked_data.json"):
    """
    Saves chunked documents (text and metadata) to a JSON file.

    Args:
        chunked_documents (list): List of LangChain Document objects.
        output_path (str): Path to save the JSON file.
    """
    json_ready_data = [
        {
            "text": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in chunked_documents
    ]

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_ready_data, f, indent=2, ensure_ascii=False)
        st.success(f"✅ Chunked data saved to {output_path}")
    except Exception as e:
        st.error(f"❌ Failed to save chunks to JSON: {str(e)}")

def load_and_chunk_documents(data_dir="documents"):
    """
    Loads PDF documents from a specified directory, splits them into chunks,
    and returns the chunks with metadata (including source, page, and filename).

    Args:
        data_dir (str): The path to the directory containing PDF files.

    Returns:
        list: A list of document chunks with metadata.
    """
    if not os.path.exists(data_dir):
        st.error(f"The '{data_dir}' directory is missing. Please create it and add your PDF files.")
        st.stop()

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    if not pdf_files:
        st.error(f"No PDF files found in the '{data_dir}' directory.")
        st.stop()

    with st.spinner("Loading research papers..."):
        docs = []
        for pdf_file in pdf_files:
            file_path = os.path.join(data_dir, pdf_file)
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["filename"] = pdf_file
                doc.metadata["source"] = file_path
            docs.extend(loaded_docs)

    with st.spinner("Chunking documents..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_documents = text_splitter.split_documents(docs)

    if not chunked_documents:
        st.warning("No documents were chunked. Check the content of your PDF files.")
        st.stop()

    missing_meta = [i for i, doc in enumerate(chunked_documents) if "filename" not in doc.metadata]
    if missing_meta:
        st.warning(f"{len(missing_meta)} chunks are missing 'filename' metadata. Check input PDFs.")
    else:
        st.success("All chunks include 'filename' metadata.")

    # Optional: Preview metadata of first few chunks
    for i, chunk in enumerate(chunked_documents[:3]):
        st.text(f"Chunk {i+1} → source: {chunk.metadata.get('source')}, "
                f"page: {chunk.metadata.get('page')}, "
                f"filename: {chunk.metadata.get('filename')}")

    # ✅ Save to JSON
    save_chunks_to_json(chunked_documents, output_path="chunked_data.json")

    return chunked_documents
