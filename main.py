import os
import glob
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def load_and_split_documents(pdf_paths, chunk_size=1024, chunk_overlap=128):
    """
    Load PDF documents and split into text chunks.
    """
    all_docs = []
    for path in pdf_paths:
        print(f"Loading PDF: {path}")
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            print(f"Loaded {len(pages)} pages from {path}")
            all_docs.extend(pages)
        except Exception as e:
            print(f"Warning: failed to load {path!r}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(all_docs)
    print(f"Split into {len(docs)} chunks.")
    return docs



def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb


def create_retriever(vectordb, k=5):
    """
    Return a retriever for the vector store.
    """
    return vectordb.as_retriever(search_kwargs={"k": k})

def create_qa_chain(retriever, model_name="llama3", streaming=False):
    """
    Build a RetrievalQA chain with Ollama.
    """
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
    llm = OllamaLLM(
        model=model_name,
        callbacks=callbacks
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def main():
    root = os.path.dirname(__file__)
    docs_dir = os.path.join(root, "docs")  # Use 'docs' folder
    pdf_paths = glob.glob(os.path.join(docs_dir, "*.pdf"))

    print(f"Looking for PDFs in: {docs_dir}")
    print(f"PDFs found: {pdf_paths}")

    if not pdf_paths:
        print("No PDF files found in ./docs/. Please add PDFs into the 'docs' folder and retry.")
        return

    docs = load_and_split_documents(pdf_paths)
    vectordb = build_vector_store(docs)
    retriever = create_retriever(vectordb)
    qa_chain = create_qa_chain(retriever, model_name="llama3", streaming=True)

    print("\nRAG pipeline is ready! Ask a question (type 'exit' to quit)")
    while True:
        question = input("\n> ")
        if question.lower() in ("exit", "quit"):
            break
        start_time = time.time()  # Start timing
        res = qa_chain.invoke(question)
        end_time = time.time()    # End timing
        duration = end_time - start_time

        print(f"\nGenerated in {duration:.2f} seconds")

        for doc in res.get("source_documents", []):
            print(f"- Source (page {doc.metadata.get('page', 'N/A')}): {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    main()
