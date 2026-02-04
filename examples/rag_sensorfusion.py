from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
import pandas as pd
from pathlib import Path
from datetime import datetime


THRESHOLDS = {
    'rpm_max': 650,
    'force_max_kn': 14.0,
    'temp_max_c': 500
}


def load_sensor_data(csv_path='data/sensor_log.csv'):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def analyze_defect_event(df, event_index):
    event = df.iloc[event_index]
    
    violations = []
    if event['rpm'] > THRESHOLDS['rpm_max']:
        violations.append(f"RPM {event['rpm']} exceeds max {THRESHOLDS['rpm_max']}")
    if event['force_kn'] > THRESHOLDS['force_max_kn']:
        violations.append(f"Force {event['force_kn']}kN exceeds max {THRESHOLDS['force_max_kn']}kN")
    if event['temperature_c'] > THRESHOLDS['temp_max_c']:
        violations.append(f"Temperature {event['temperature_c']}C exceeds max {THRESHOLDS['temp_max_c']}C")
    
    start_idx = max(0, event_index - 2)
    end_idx = min(len(df), event_index + 3)
    context_data = df.iloc[start_idx:end_idx]
    
    return {
        'defect_type': event['defect_type'],
        'timestamp': event['timestamp'],
        'violations': violations,
        'context': context_data
    }


def load_documents_with_metadata():
    documents = []
    
    print('Loading ISO standards...')
    loader1 = PyPDFLoader('data/doc1_standards.pdf')
    docs1 = loader1.load()
    for doc in docs1:
        doc.metadata['doctype'] = 'iso_standard'
        doc.metadata['topic'] = 'fsw_parameters'
    documents.extend(docs1)
    
    print('Loading manuals...')
    loader2 = PyPDFLoader('data/doc2_manual.pdf')
    docs2 = loader2.load()
    for doc in docs2:
        doc.metadata['doctype'] = 'manual'
        doc.metadata['topic'] = 'procedures'
    documents.extend(docs2)
    
    print('Loading procedures...')
    loader3 = PyPDFLoader('data/doc3_procedures.pdf')
    docs3 = loader3.load()
    for doc in docs3:
        doc.metadata['doctype'] = 'procedure'
        doc.metadata['topic'] = 'troubleshooting'
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
        persist_directory='./.chromadb_sensor'
    )
    print('Vector store created')
    return vectorstore


def explain_defect(defect_analysis, vectorstore):
    sensor_context = f"""
Defect Type: {defect_analysis['defect_type']}
Time: {defect_analysis['timestamp']}
Threshold Violations:
{chr(10).join(f'  - {v}' for v in defect_analysis['violations'])}

Recent Sensor Readings:
{defect_analysis['context'].to_string(index=False)}
"""
    
    question = f"""Based on the following sensor data, explain why a {defect_analysis['defect_type']} defect occurred:

{sensor_context}

What is the root cause according to FSW standards?"""
    
    llm = OllamaLLM(
        model='llama3.2:1b',
        temperature=0,
        base_url='http://localhost:11434'
    )
    
    prompt_template = '''You are an FSW defect analysis expert. Use the document context to explain the defect based on the sensor data in the question.

Context: {context}

Question: {question}

Provide a technical root cause explanation:'''
    
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
    
    result = qa_chain.invoke({'query': question})
    return result


def main():
    print('=== RAG 3: Sensor-Fusion Defect Analysis ===\n')
    sensor_df = load_sensor_data()
    print(f'Loaded {len(sensor_df)} sensor readings\n')
    
    documents = load_documents_with_metadata()
    vectorstore = create_vectorstore(documents)
    
    defects = sensor_df[sensor_df['defect_type'] != 'none']
    print(f'\nFound {len(defects)} defect events:')
    for idx, defect in defects.iterrows():
        print(f'  {idx}. {defect["timestamp"]} - {defect["defect_type"]}')
    
    print('\n' + '='*60)
    print('Select a defect to analyze (enter row number) or quit:')
    print('='*60 + '\n')
    
    while True:
        choice = input('Defect # (or quit): ')
        
        if choice.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            event_idx = int(choice)
            if event_idx not in defects.index:
                print('Invalid defect number. Try again.\n')
                continue
        except ValueError:
            print('Enter a number or quit.\n')
            continue
        
        print('\n' + '='*60)
        print('ANALYZING DEFECT EVENT...')
        print('='*60)
        
        defect_analysis = analyze_defect_event(sensor_df, event_idx)
        
        print(f'\nDefect: {defect_analysis["defect_type"]}')
        print(f'Time: {defect_analysis["timestamp"]}')
        print(f'\nThreshold Violations:')
        for v in defect_analysis['violations']:
            print(f'  {v}')
        
        print('\nRetrieving relevant ISO standards and procedures...')
        result = explain_defect(defect_analysis, vectorstore)
        
        print(f'\nROOT CAUSE ANALYSIS:\n')
        print(result['result'])
        print(f'\nSources: {len(result["source_documents"])} document chunks')
        
        doctypes = [doc.metadata.get('doctype', 'unknown') 
                   for doc in result['source_documents']]
        print(f'Document types: {", ".join(set(doctypes))}')
        print('\n' + '='*60 + '\n')


if __name__ == '__main__':
    main()
