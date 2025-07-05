Modular RAG System with Groq, Chroma, and Hugging Face
This project implements a modular Retrieval-Augmented Generation (RAG) system for Question Answering (QA) on a collection of research papers. It uses the Groq API for high-speed LLM inference, Hugging Face for state-of-the-art sentence embeddings, and Chroma DB for an efficient, persistent vector store. The entire application is wrapped in a user-friendly Streamlit web interface.

Features
Modular Design: The code is logically separated into modules for data loading/chunking, vectorization, and the QA chain.

Document Preprocessing: Loads and chunks PDF documents from a local directory.

Vectorization: Uses Hugging Face's all-MiniLM-L6-v2 model to create vector embeddings of document chunks.

Retrieval System: Employs a Chroma vector store to quickly find the most relevant document chunks for a given query.

Answer Generation: Integrates the Llama3 model via the Groq API to generate accurate, context-aware answers.

Source Attribution: Displays the source documents and specific text chunks used to formulate the answer.

User-Friendly UI: A simple Streamlit interface allows users to easily process data and ask questions.

Project Structure
.
├── data/
│   └── (Add your PDF research papers here)
├── vectorstore/
│   └── (Chroma DB files will be saved here automatically)
├── .env
├── Data_Loading_Chunking.py
├── Data_Vectorization.py
├── QA_Chain.py
├── app.py
├── requirements.txt
└── README.md

Setup and Installation
Follow these steps to set up and run the project locally.

1. Clone the Repository

git clone <repository-url>
cd <repository-directory>

Or, simply create the files as provided.

2. Create a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

Install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

4. Set Up Environment Variables

You need an API key from Groq to use their language models.

Create a file named .env in the root of your project directory.

Add your API key to the .env file like this:

GROQ_API_KEY="your-groq-api-key-here"

5. Add Your Documents

Create a directory named data in the root of your project.

Place the PDF research papers you want to query inside the data directory.

How to Run the Application
Start the Streamlit App

Once the setup is complete, run the following command in your terminal:

streamlit run app.py

Initialize the Vector Store

When you first run the app, use the sidebar on the left.

Click the "Initialize/Update Vector Store" button. The app will process all the PDFs in your data folder, create embeddings, and save a persistent Chroma vector store in the vectorstore directory. This only needs to be done once, or whenever you add, remove, or change the documents in the data folder.

Ask Questions

After the vector store is initialized, the main part of the page will show a text input box.

Type your question into the box and press Enter.

The system will retrieve relevant information, generate an answer, and display it. You can also view the source chunks used for the answer in the "View Sources" expander.