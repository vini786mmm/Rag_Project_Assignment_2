import os
import streamlit as st
from dotenv import load_dotenv

from Data_Loading_Chunking import load_and_chunk_documents
from Data_Vectorization import create_and_save_vector_store, CHROMA_PATH
from QA_chain import create_qa_chain

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function to run the Streamlit RAG application.
    """
    st.set_page_config(page_title="RAG Q&A with Groq & Chroma", layout="wide")
    st.title("Ask Questions About Your Research Papers")
    st.markdown("""
        This app uses a Retrieval-Augmented Generation (RAG) system with:
        - 💬 [Groq](https://groq.com) for LLM
        - 🤗 HuggingFace Embeddings
        - 🧠 Chroma Vector Store
    """)

    # Validate Groq API Key
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ `GROQ_API_KEY` environment variable not found. Please add it to your `.env` file.")
        st.stop()

    # Sidebar for setup
    with st.sidebar:
        st.header("📄 Data Setup")
        st.info("Place your PDF files inside the `documents/` folder, then click below.")
        if st.button("Initialize / Update Vector Store"):
            chunked_docs = load_and_chunk_documents()
            create_and_save_vector_store(chunked_docs)

    # Main Q&A interface
    if os.path.exists(CHROMA_PATH):
        try:
            retrieval_chain = create_qa_chain()

            st.header("💡 Ask a Question")
            user_prompt = st.text_input("Type your question here", "", key="user_prompt")

            if user_prompt:
                with st.spinner("🔍 Thinking..."):
                    response = retrieval_chain.invoke({"input": user_prompt})

                    # Handle flexible key formats
                    answer = response.get("answer") or response.get("result", "No answer returned.")
                    source_docs = response.get("source_documents", [])

                    st.subheader("📌 Answer:")
                    st.write(answer)

                    # Show source documents
                    with st.expander("🗂️ View Sources"):
                        if source_docs:
                            for i, doc in enumerate(source_docs, 1):
                                source = doc.metadata.get("filename", "Not found")
                                print(f"Source {i}: {source}")
                                page = doc.metadata.get("page", "N/A")
                                st.info(f"📄 Source {i}: {source} (Page {page})")
                                st.text(doc.page_content)
                        else:
                            st.warning("No source documents returned.")
        except Exception as e:
            st.error(f"🚨 An error occurred: {str(e)}")
            st.info("Make sure your PDFs are in the `documents/` folder and try re-initializing.")
    else:
        st.warning("⚠️ Vector store not found.")
        st.info("Please use the **sidebar** to initialize the vector store.")

if __name__ == "__main__":
    main()
