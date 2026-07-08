import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vectorstore(resume_text: str):
    """
    Builds a temporary in-memory FAISS vector database for on-the-fly single resume scoring.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(resume_text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def add_resume_to_vectorstore(resume_text: str, filename: str):
    """
    Chunks a candidate's resume, tags each chunk with the filename as metadata, and saves/updates
    the local persistent FAISS database index.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(resume_text)
    metadatas = [{"filename": filename} for _ in chunks]
    
    db_path = "./vectorstore"
    if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "index.faiss")):
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_texts(chunks, metadatas=metadatas)
    else:
        vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
        
    vectorstore.save_local(db_path)
    return vectorstore


def retrieve_candidates_from_db(job_description: str, k: int = 15) -> dict:
    """
    Performs a semantic search across the entire persistent FAISS index,
    aggregates relevant segments grouped by filename, and formats their matching contexts.
    """
    db_path = "./vectorstore"
    if not (os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "index.faiss"))):
        return {}
        
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    relevant_chunks = vectorstore.similarity_search(job_description, k=k)
    
    # Group chunk texts by filename
    candidate_chunks = {}
    for chunk in relevant_chunks:
        filename = chunk.metadata.get("filename", "Unknown")
        if filename not in candidate_chunks:
            candidate_chunks[filename] = []
        candidate_chunks[filename].append(chunk.page_content)
        
    # Build formatted context block for each candidate
    candidate_contexts = {}
    for filename, chunks in candidate_chunks.items():
        candidate_contexts[filename] = "\n".join([f"[Chunk {i+1}]: {c}" for i, c in enumerate(chunks)])
        
    return candidate_contexts


def delete_resume_from_vectorstore(filename: str):
    """
    Loads the persistent FAISS index, filters out all chunks matching the 
    specified filename metadata, and saves the updated index.
    """
    db_path = "./vectorstore"
    if not (os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "index.faiss"))):
        return

    # Load database
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    # Extract all current texts and metadata
    docs = vectorstore.docstore._dict.values()
    
    # Filter out chunks belonging to the deleted file
    remaining_texts = []
    remaining_metadatas = []
    
    for doc in docs:
        if doc.metadata.get("filename", "").lower() != filename.lower():
            remaining_texts.append(doc.page_content)
            remaining_metadatas.append(doc.metadata)
            
    # Save database back
    if remaining_texts:
        new_store = FAISS.from_texts(remaining_texts, embeddings, metadatas=remaining_metadatas)
        new_store.save_local(db_path)
    else:
        # If no resumes remain, completely remove the folder
        import shutil
        shutil.rmtree(db_path, ignore_errors=True)

