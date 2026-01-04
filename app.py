import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
import pandas as pd
import os


st.set_page_config(
    page_title="FSW RAG Demo",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Sensor thresholds
THRESHOLDS = {
    'rpm_max': 650,
    'force_max_kn': 14.0,
    'temp_max_c': 500
}


@st.cache_resource
def load_vectorstore():
    """Load pre-built vector store (instant!)"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    
    vectorstore = Chroma(
        persist_directory="./vectorstore",
        embedding_function=embeddings
    )
    
    return vectorstore


def get_llm():
    """Get LLM via HuggingFace Inference Provider"""
    hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
    
    if not hf_token:
        st.error("Set HF_TOKEN in Streamlit secrets")
        st.stop()
        
    llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=hf_token,
            max_new_tokens=512,
            temperature=0.1,
            provider="novita"
            )
    
    chat_model = ChatHuggingFace(llm=llm)
    
    return chat_model


def setup_qa_chain(vectorstore, filter_metadata=None):
    """Setup QA chain"""
    llm = get_llm()
    
    # CHANGED: Better prompt that forces use of context
    prompt_template = """You are an expert in friction stir welding. Use the following context from ISO 25239 standards and operational procedures to answer the question. Be specific about root causes and corrective actions.

Context: {context}

Question: {question}

Detailed Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    search_kwargs = {'k': 5}
    if filter_metadata:
        search_kwargs['filter'] = filter_metadata
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain


# Sidebar
st.sidebar.title("🔧 FSW Defect Analysis")
st.sidebar.markdown("**RAG Demo by Karthik Kagolanu**")


demo_mode = st.sidebar.radio(
    "Select Demo:",
    ["RAG 1: Simple QA", "RAG 2: Multi-Doc Filtering", "RAG 3: Sensor Fusion"],
    index=0
)


# Main content
st.title("Friction Stir Welding RAG Demo")
st.markdown("*Retrieval-Augmented Generation for FSW defect analysis*")
st.markdown("---")


# Load vectorstore once
if 'vectorstore' not in st.session_state:
    with st.spinner("Loading vector store..."):
        st.session_state.vectorstore = load_vectorstore()


# RAG 1
if demo_mode == "RAG 1: Simple QA":
    st.header("RAG 1: Basic Document QA")
    
    question = st.text_input("Ask about FSW processes:", key="rag1")
    
    if st.button("Get Answer", key="btn1"):
        if question:
            with st.spinner("Generating answer..."):
                qa_chain = setup_qa_chain(st.session_state.vectorstore)
                result = qa_chain.invoke({"query": question})
                
                st.success("Answer:")
                st.write(result['result'])
                
                with st.expander("View Sources"):
                    st.write(f"{len(result['source_documents'])} chunks used")


# RAG 2
elif demo_mode == "RAG 2: Multi-Doc Filtering":
    st.header("RAG 2: Document Filtering")
    
    doc_filter = st.selectbox(
        "Filter documents:",
        ["All", "ISO Standards", "Manuals", "Procedures"]
    )
    
    question = st.text_input("Ask your question:", key="rag2")
    
    if st.button("Get Answer", key="btn2"):
        if question:
            filter_metadata = None
            if "Standards" in doc_filter:
                filter_metadata = {'doctype': 'iso_standard'}
            elif "Manuals" in doc_filter:
                filter_metadata = {'doctype': 'manual'}
            elif "Procedures" in doc_filter:
                filter_metadata = {'doctype': 'procedure'}
            
            with st.spinner("Searching..."):
                qa_chain = setup_qa_chain(st.session_state.vectorstore, filter_metadata)
                result = qa_chain.invoke({"query": question})
                
                st.success("Answer:")
                st.write(result['result'])
                
                with st.expander("View Sources"):
                    doctypes = [doc.metadata.get('doctype') for doc in result['source_documents']]
                    st.write(f"Types: {', '.join(set(doctypes))}")


# RAG 3
# RAG 3
elif demo_mode == "RAG 3: Sensor Fusion":
    st.header("RAG 3: Sensor-Fusion Analysis")
    
    # Load sensor data
    sensor_df = pd.read_csv('data/sensor_log.csv')
    defects = sensor_df[sensor_df['defect_type'] != 'none']
    
    st.subheader("Defect Events")
    st.dataframe(defects[['timestamp', 'defect_type', 'rpm', 'force_kn', 'temperature_c']])
    
    selected_idx = st.selectbox(
        "Select defect:",
        defects.index,
        format_func=lambda x: f"{defects.loc[x, 'timestamp']} - {defects.loc[x, 'defect_type']}"
    )
    
    if st.button("Analyze", key="btn3"):
        event = sensor_df.iloc[selected_idx]
        
        violations = []
        if event['rpm'] > THRESHOLDS['rpm_max']:
            violations.append(f"RPM {event['rpm']} exceeds max {THRESHOLDS['rpm_max']}")
        if event['force_kn'] > THRESHOLDS['force_max_kn']:
            violations.append(f"Force {event['force_kn']}kN exceeds max {THRESHOLDS['force_max_kn']}kN")
        if event['temperature_c'] > THRESHOLDS['temp_max_c']:
            violations.append(f"Temperature {event['temperature_c']}C exceeds max {THRESHOLDS['temp_max_c']}C")
        
        st.warning("Violations:")
        for v in violations:
            st.write(f"- {v}")
        
        start_idx = max(0, selected_idx - 2)
        end_idx = min(len(sensor_df), selected_idx + 3)
        context_data = sensor_df.iloc[start_idx:end_idx]
        
        sensor_context = f"""
Defect Type: {event['defect_type']}
Time: {event['timestamp']}
Threshold Violations:
{chr(10).join(f'  - {v}' for v in violations)}

Recent Sensor Readings:
{context_data.to_string(index=False)}
"""
        
        question = f"""Based on the following sensor data, explain why a {event['defect_type']} defect occurred:

{sensor_context}

What is the root cause according to FSW standards?"""
        
        with st.spinner("Analyzing..."):
            llm = get_llm()
            
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
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
                chain_type_kwargs={'prompt': PROMPT},
                return_source_documents=True
            )
            
            result = qa_chain.invoke({'query': question})
            
            with st.expander("Debug: Retrieved Context"):
                st.write("Question sent to LLM:")
                st.code(question)
                st.write(f"Retrieved {len(result['source_documents'])} chunks:")
                for i, doc in enumerate(result['source_documents'], 1):
                    st.write(f"Chunk {i}: (from {doc.metadata.get('doctype', 'unknown')})")
                    st.text(doc.page_content[:400])
                    st.markdown("---")
            
            st.success("Root Cause:")
            st.write(result['result'])
            
            doctypes = [doc.metadata.get('doctype', 'unknown') 
                       for doc in result['source_documents']]
            st.info(f"Sources: {', '.join(set(doctypes))}")

st.sidebar.markdown("---")
st.sidebar.markdown("[GitHub Repo](https://github.com/SkullKrak7/RAG_Demo)")
