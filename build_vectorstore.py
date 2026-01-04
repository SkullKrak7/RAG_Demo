from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore():
    """Build vector store once and persist it"""
    documents = []
    
    # Load PDFs
    for pdf_file, doc_type in [
        ('doc1_standards.pdf', 'iso_standard'),
        ('doc2_manual.pdf', 'manual'),
        ('doc3_procedures.pdf', 'procedure')
    ]:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(f'data/{pdf_file}')
        docs = loader.load()
        for doc in docs:
            doc.metadata['doctype'] = doc_type
        documents.extend(docs)
    
    print(f"Loaded {len(documents)} pages")
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store with faster embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"  # Faster, smaller
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vectorstore"  # Save to disk
    )
    
    print("Vector store saved to ./vectorstore")

if __name__ == "__main__":
    build_vectorstore()
