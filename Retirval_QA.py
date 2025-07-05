import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from QA_chain import create_qa_chain

# Load environment variables and disable Chroma telemetry
load_dotenv()
os.environ["LANGCHAIN_CHROMA_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

def answer_all_questions():
    """
    Loads a QA chain and uses it to answer predefined questions,
    printing results with source document metadata.
    """
    # Create the QA chain
    qa_chain = create_qa_chain()

    # Define a list of questions
    questions = [
        "What are the main components of a RAG model, and how do they interact?",
        "What are the two sub-layers in each encoder layer of the Transformer model?",
        "Explain how positional encoding is implemented in Transformers and why it is necessary.",
        "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
        "What is few-shot learning, and how does GPT-3 implement it during inference?"
    ]

    results = []

    for question in questions:
        try:
            # Invoke the QA chain
            response = qa_chain.invoke({"input": question})
            answer = response.get("answer") or response.get("result", "").strip()

            # Get source documents and extract filenames
            source_docs = response.get("source_documents", [])
            source_info = [doc.metadata.get("filename", doc.metadata.get("source", "Unknown source")) for doc in source_docs]

            results.append({
                "question": question,
                "answer": answer,
                "sources": source_info
            })

        except Exception as e:
            results.append({
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": []
            })

    return results


# Run the QA system
if __name__ == "__main__":
    answers = answer_all_questions()

    for item in answers:
        print(f"\nQ: {item['question']}")
        print(f"A: {item['answer']}")
        if item['sources']:
            print("Sources:")
            for src in item['sources']:
                print(f" - {src}")
        else:
            print("Sources: None")
        print("-" * 80)
