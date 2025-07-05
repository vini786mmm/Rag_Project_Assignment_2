import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from Data_Vectorization import load_vector_store


def create_qa_chain():
    """
    Creates and returns a retrieval-augmented generation (RAG) chain.

    The chain includes:
    - A retriever from a Chroma vector store
    - A language model (Groq's LLaMA3)
    - A document chain that feeds context to the model using a prompt

    Returns:
        retrieval_chain (Runnable): A LangChain retrieval chain.
    """
    # Load vector store
    db = load_vector_store()
    if db is None:
        raise RuntimeError("Vector store could not be loaded.")

    retriever = db.as_retriever()

    # Retrieve Groq API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")

    # Initialize the LLM (LLaMA3 8B)
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"  # Make sure case matches Groq's naming conventions
    )

    # Prompt Template for RAG
    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant for question-answering tasks.
    Use only the provided context to answer the user's question.
    Be concise and accurate.
    If the answer is not in the context, say "I don't know."

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Combine documents into a single prompt using "stuff" strategy
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Create retrieval chain: retriever + document QA chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
