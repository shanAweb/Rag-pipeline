import os 
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Function to load documents
def load_documents(docs_path='docs'):
    """Load all the text files from the docs directory"""
    print(f"Loading documents from {docs_path}")

    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()


    if len(documents) == 0:
        raise FileNotFoundError(f"No. txt files found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\n Document {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents

# Functio to split documents and create chunks
def split_documents(documents, chunk_size = 800, chunk_overlap=0):
    """Split the document into smaller chunks"""
    print("Documents splitting is starting")

    text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"===== Chunks {i+1} ======")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(f"{chunk.page_content}")
            print("_" * 50)

        if len(chunks) > 5:
            print(f"\n..... {len(chunks) - 5} more chunks") 

    return chunks

# Function to create vector embeddiengs and store in ChromaDB
def create_vector_store(chunks, persist_directory="db/chromedb"):
    "Create and persist ChromaDB vector store"

    print("=== Creating embeddiengs and storing in ChromaDB ===")

    embedding_model = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001')

    BATCH_SIZE = 10

    initial_batch = chunks[:BATCH_SIZE]
    print(f"--- Initializing ChromaDB with first {len(initial_batch)} chunks ---")

    print("---Creating ChromaDB store----")
    vectorstore=Chroma.from_documents(
        documents=initial_batch, #first batch of 10
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space" : "cosine"}
    )

    remaining_chunks = chunks[BATCH_SIZE:]

    if remaining_chunks:
        print(f"--- Processing remaining {len(remaining_chunks)} chunks in batches ---")
        for i in range(0, len(remaining_chunks), BATCH_SIZE):
            batch = remaining_chunks[i : i+BATCH_SIZE]
            try:
                vectorstore.add_documents(batch)
                print(f"Successfully stored chunks {i + BATCH_SIZE} to {i + BATCH_SIZE + len(batch)}")
            # Wait 10 seconds between every 10 small chunks
                time.sleep(10) 
            except Exception as e:
                if "429" in str(e):
                    print("Quota hit! Sleeping for 60 seconds to reset...")
                    time.sleep(60) # Long sleep if we hit a wall
                    vectorstore.add_documents(batch) # Retry once
                else:
                    raise e
    print("---Finished creating vector store---")
    print (f"Vector store created and saved to {persist_directory}")

    return vectorstore

def main():
    print("main function")

    #1. Loading the files
    documents = load_documents(docs_path="docs")
    #2. Chunking the files
    chunks = split_documents(documents)
    #3. Embedding and Storing in Vector DB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()