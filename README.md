# RAG Implementation with LangChain and Locally Installed LLM

This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://www.langchain.com/) in Python. It uses a locally installed large language model (LLM) to answer questions based on the content of PDF documents. The entire process is executed locally without any reliance on cloud APIs.

---

## Project Overview

The system allows you to:

1. Load one or more PDF files from a `docs/` folder.
2. Split the text into manageable chunks for processing.
3. Convert those chunks into numerical vector embeddings using a pretrained model.
4. Store the vectors using a vector database (FAISS).
5. Use a similarity search to retrieve the most relevant chunks for any user query.
6. Feed the retrieved chunks and user query into a locally running LLM via Ollama.
7. Return a generated response, along with source references.

---

## How the Pipeline Works (Step-by-Step)

### 1. PDF Loading

- The `PyPDFLoader` class from `langchain_community.document_loaders` is used to extract text from PDF files.
- All PDF files in the `./docs` directory are processed and loaded page-by-page into memory as documents.

### 2. Text Splitting

- Text from PDFs can be long and unstructured.
- The `RecursiveCharacterTextSplitter` is used to break text into smaller overlapping chunks.
- This is necessary because LLMs have a limited context window.
- Parameters used:
  - `chunk_size=1024` characters
  - `chunk_overlap=128` characters

### 3. Embedding Generation

- Each text chunk is converted into a dense vector representation (called an **embedding**).
- This is done using the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face.
- Embeddings capture the **semantic meaning** of the text, so similar content ends up with similar vectors.

### 4. Vector Store (FAISS)

- The embeddings are stored in **FAISS**, a high-speed vector database.
- FAISS allows fast **approximate nearest neighbor search** based on vector similarity.
- It uses **cosine similarity** (or a similar metric) to compare the query embedding to stored document chunks.

### 5. Retriever

- The FAISS vector store is wrapped in a retriever.
- When a user inputs a question, the question is embedded and compared to all stored document chunks.
- The top `k=5` most similar chunks are retrieved.
- These form the **context** for the LLM to generate its response.

### 6. LLM (via Ollama)

- The model used is `llama3`, running locally via [Ollama](https://ollama.com).
- Ollama is a tool to run LLMs on your own machine using optimized packages.
- The `OllamaLLM` wrapper in LangChain allows sending prompts and receiving responses from the local model.
- The model is used in a **"stuff" chain** — which means all retrieved chunks are stuffed into the prompt before passing to the model.

### 7. QA Chain and Response

- The `RetrievalQA` chain ties together the retriever and the LLM.
- The LLM is prompted with: the user's question and the retrieved document chunks.
- The LLM then generates an answer that is grounded in the source documents.
- Source chunks used in the response are also returned for transparency.

---

## Folder Structure

