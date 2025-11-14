"""
Streamlit Web Interface for Document Assistant
Provides a user-friendly interface for conversational document interaction
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our RAG system
from document_assistant import ConversationalRAG, StructuredDataHandler

# Page configuration
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.conversation_history = []
    st.session_state.data_loaded = False

# Header
st.markdown('<h1 class="main-header">📚 Document Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Conversational Interface for Multi-format Documents</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 System Controls")
    
    # Load documents button
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Loading documents and initializing RAG system..."):
            try:
                st.session_state.rag_system = ConversationalRAG("Gen AI/Source")
                st.session_state.data_loaded = True
                st.success("✅ System initialized successfully!")
                
                # Show loaded documents info
                st.info(f"📄 Loaded {len(st.session_state.rag_system.all_chunks)} document chunks")
                
                # Show available dataframes
                if st.session_state.rag_system.structured_handler.dataframes:
                    st.info(f"📊 Loaded {len(st.session_state.rag_system.structured_handler.dataframes)} CSV files")
                    
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
    
    # System status
    st.divider()
    st.header("📊 System Status")
    if st.session_state.data_loaded:
        st.success("✅ System Ready")
        
        # Display loaded files
        if st.session_state.rag_system:
            st.subheader("📁 Loaded Files:")
            source_path = Path("Gen AI/Source")
            if source_path.exists():
                files = list(source_path.glob("*"))
                for file in files:
                    file_icon = "📄" if file.suffix in ['.pdf', '.docx'] else "📊"
                    st.text(f"{file_icon} {file.name}")
    else:
        st.warning("⏳ System not initialized")
    
    # Quick actions
    st.divider()
    st.header("⚡ Quick Actions")
    
    if st.button("📋 Show Test Questions"):
        st.session_state.show_test_questions = True
    
    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.success("History cleared!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # Show test questions if requested
    if 'show_test_questions' in st.session_state and st.session_state.show_test_questions:
        with st.expander("📝 Test Questions", expanded=True):
            test_questions = [
                "What is the order quantity handled by Ranjit?",
                "What is the percentage of orders that haven't dispatched?",
                "List of products under the recoil kits orders?",
                "How much GST was charged for my insurance and when does my third party insurance expire?",
                "What are the roles we are currently hiring for?",
                "For the product Glock 17 what is the planned WO release date?",
                "Ok try Glock - 17",
                "What is our criteria to hire a data scientist?",
                "What are the benifits of log book and how can i set it up?",
                "How do i create a zone?"
            ]
            
            for i, q in enumerate(test_questions, 1):
                if st.button(f"{i}. {q}", key=f"test_q_{i}"):
                    st.session_state.current_question = q
        
        st.session_state.show_test_questions = False
    
    # Query input
    query = st.text_input(
        "Ask a question about your documents:",
        value=st.session_state.get('current_question', ''),
        placeholder="e.g., What is the order quantity handled by Ranjit?",
        key="query_input"
    )
    
    # Clear current question after using it
    if 'current_question' in st.session_state:
        del st.session_state.current_question
    
    # Submit button
    if st.button("🔍 Search", type="primary") and query:
        if not st.session_state.data_loaded:
            st.error("⚠️ Please initialize the system first!")
        else:
            with st.spinner("Searching documents..."):
                try:
                    # Get response from RAG system
                    response = st.session_state.rag_system.generate_response(query)
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'query': query,
                        'response': response
                    })
                    
                    # Display response
                    st.success("✅ Response generated!")
                    
                    # Show answer
                    st.markdown("### 📝 Answer:")
                    st.markdown(f'<div class="response-box">{response["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Show sources
                    if response['sources']:
                        st.markdown("### 📚 Sources:")
                        unique_sources = list(set(response['sources']))
                        for source in unique_sources:
                            st.markdown(f'<div class="source-box">📄 {source}</div>', unsafe_allow_html=True)
                    
                    # Show structured data if available
                    if response['data']:
                        st.markdown("### 📊 Structured Data:")
                        st.json(response['data'])
                        
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
    
    # Conversation history
    if st.session_state.conversation_history:
        st.divider()
        st.header("📜 Conversation History")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            with st.expander(f"Q{i}: {conv['query'][:100]}...", expanded=(i==1)):
                st.markdown(f"**Question:** {conv['query']}")
                st.markdown(f"**Answer:** {conv['response']['answer']}")
                if conv['response']['sources']:
                    st.markdown(f"**Sources:** {', '.join(set(conv['response']['sources']))}")

with col2:
    st.header("📈 Analytics & Insights")
    
    if st.session_state.data_loaded and st.session_state.rag_system:
        # Show data statistics
        structured_handler = st.session_state.rag_system.structured_handler
        
        if structured_handler.dataframes:
            st.subheader("📊 Data Overview")
            
            for df_name, df in structured_handler.dataframes.items():
                with st.expander(f"📁 {df_name}"):
                    st.write(f"**Rows:** {len(df)}")
                    st.write(f"**Columns:** {len(df.columns)}")
                    
                    # Show sample data
                    if st.checkbox(f"Show sample data for {df_name}", key=f"sample_{df_name}"):
                        st.dataframe(df.head(5))
                    
                    # Create visualizations based on data type
                    if 'status' in df.columns:
                        fig = px.pie(
                            df['status'].value_counts().reset_index(),
                            values='count',
                            names='status',
                            title="Order Status Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if 'mktg_specialistsmanagers' in df.columns and 'qty' in df.columns:
                        manager_qty = df.groupby('mktg_specialistsmanagers')['qty'].sum().reset_index()
                        manager_qty = manager_qty.sort_values('qty', ascending=False).head(10)
                        
                        fig = px.bar(
                            manager_qty,
                            x='mktg_specialistsmanagers',
                            y='qty',
                            title="Top 10 Managers by Order Quantity"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Document statistics
        st.subheader("📄 Document Statistics")
        total_chunks = len(st.session_state.rag_system.all_chunks)
        
        # Count chunks by type
        chunk_types = {}
        for chunk in st.session_state.rag_system.all_chunks:
            chunk_type = chunk.source_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        if chunk_types:
            fig = go.Figure(data=[
                go.Bar(x=list(chunk_types.keys()), y=list(chunk_types.values()))
            ])
            fig.update_layout(title="Document Chunks by Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Show metrics
        col1_metric, col2_metric = st.columns(2)
        with col1_metric:
            st.metric("Total Chunks", total_chunks)
        with col2_metric:
            st.metric("Unique Sources", len(set(chunk.metadata.get('filename', 'Unknown') for chunk in st.session_state.rag_system.all_chunks)))
    
    else:
        st.info("📌 Initialize the system to see analytics")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Document Assistant v1.0 | RAG-based Multi-format Document Interaction System</p>
    <p>Supports PDF, DOCX, and CSV files | Powered by ChromaDB and Transformers</p>
</div>
""", unsafe_allow_html=True)
