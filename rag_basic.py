"""RAG #1: Simple Document Q&A (ChromaDB)"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pathlib import Path


def load_and_split_pdf(pdf_path: str):
    print(f'Loading {pdf_path}...')
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Split into {len(chunks)} chunks')
    return chunks


def create_vectorstore(chunks):
    print('Creating embeddings...')
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    db_path = Path('./chroma_db').resolve()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_path)
    )
    print('Vector store created')
    return vectorstore


def setup_qa_chain(vectorstore):
    llm = OllamaLLM(
        model='llama3.2:1b',
        temperature=0,
        base_url='http://localhost:11434'
    )
    
    prompt_template = '''Use the following context to answer the question.
If you don't know, say I don't know.

Context: {context}

Question: {question}

Answer:'''

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        chain_type_kwargs={'prompt': PROMPT},
        return_source_documents=True
    )
    
    return qa_chain


def main():
    pdf_path = 'sample_manual.pdf'
    
    if not Path(pdf_path).exists():
        print(f'ERROR: {pdf_path} not found!')
        return
    
    chunks = load_and_split_pdf(pdf_path)
    vectorstore = create_vectorstore(chunks)
    qa_chain = setup_qa_chain(vectorstore)
    
    print('\n=== RAG #1 Demo Ready ===')
    print('Ask questions (type quit to exit)\n')
    
    while True:
        question = input('Question: ')
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        print('\nThinking...')
        result = qa_chain.invoke({'query': question})
        
        print(f'\nAnswer: {result["result"]}')
        print(f'\nSources: {len(result["source_documents"])} chunks')
        print('-' * 60)


if __name__ == '__main__':
    main()
