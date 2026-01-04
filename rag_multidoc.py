from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pathlib import Path


def load_documents_with_metadata():
    documents = []
    
    print('Loading standards document...')
    loader1 = PyPDFLoader('data/doc1_standards.pdf')
    docs1 = loader1.load()
    for doc in docs1:
        doc.metadata['doctype'] = 'standard'
        doc.metadata['priority'] = 'high'
        doc.metadata['category'] = 'technical_spec'
    documents.extend(docs1)
    
    print('Loading manual...')
    loader2 = PyPDFLoader('data/doc2_manual.pdf')
    docs2 = loader2.load()
    for doc in docs2:
        doc.metadata['doctype'] = 'manual'
        doc.metadata['priority'] = 'medium'
        doc.metadata['category'] = 'procedures'
    documents.extend(docs2)
    
    print('Loading procedures...')
    loader3 = PyPDFLoader('data/doc3_procedures.pdf')
    docs3 = loader3.load()
    for doc in docs3:
        doc.metadata['doctype'] = 'procedure'
        doc.metadata['priority'] = 'medium'
        doc.metadata['category'] = 'guidelines'
    documents.extend(docs3)
    
    print(f'Loaded {len(documents)} pages total')
    return documents


def create_vectorstore(documents):
    print('Splitting documents...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Created {len(chunks)} chunks')
    
    print('Creating embeddings...')
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory='./.chromadb_multi'
    )
    print('Vector store created with metadata')
    return vectorstore


def setup_qa_chain(vectorstore, filter_metadata=None):
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
    
    search_kwargs = {'k': 3}
    if filter_metadata:
        search_kwargs['filter'] = filter_metadata
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
        chain_type_kwargs={'prompt': PROMPT},
        return_source_documents=True
    )
    return qa_chain


def main():
    documents = load_documents_with_metadata()
    vectorstore = create_vectorstore(documents)
    
    print('\n=== RAG 2: Multi-Document Demo ===')
    print('Filters:')
    print('1. All documents (no filter)')
    print('2. Standards only (doctype=standard)')
    print('3. Manuals only (doctype=manual)')
    print('4. Procedures only (doctype=procedure)')
    print('5. High priority only (priority=high)')
    print('(quit to exit)\n')
    
    while True:
        filter_choice = input('Select filter (1-5): ')
        
        if filter_choice.lower() in ['quit', 'exit', 'q']:
            break
        
        filter_metadata = None
        if filter_choice == '2':
            filter_metadata = {'doctype': 'standard'}
            print('Filtering: Standards only')
        elif filter_choice == '3':
            filter_metadata = {'doctype': 'manual'}
            print('Filtering: Manuals only')
        elif filter_choice == '4':
            filter_metadata = {'doctype': 'procedure'}
            print('Filtering: Procedures only')
        elif filter_choice == '5':
            filter_metadata = {'priority': 'high'}
            print('Filtering: High priority only')
        else:
            print('Searching: All documents')
        
        question = input('\nQuestion: ')
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        qa_chain = setup_qa_chain(vectorstore, filter_metadata)
        
        print('Thinking...')
        result = qa_chain.invoke({'query': question})
        
        print(f'\nAnswer: {result["result"]}')
        print(f'Sources: used {len(result["source_documents"])} chunks')
        
        doctypes = [doc.metadata.get('doctype', 'unknown') 
                   for doc in result['source_documents']]
        print(f'Document types: {", ".join(set(doctypes))}')
        print('-' * 60 + '\n')


if __name__ == '__main__':
    main()
