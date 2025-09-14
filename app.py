import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Data Analysis Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    /* Main container */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Chat container */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e1e5e9;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: right;
    }
    
    /* Bot message */
    .bot-message {
        background: #ffffff;
        color: #333;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e1e5e9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Typing indicator */
    .typing-indicator {
        background: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e1e5e9;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #667eea;
        padding: 0.5rem 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Data Analysis Chatbot</h1>
    <p>Ask questions about your data and get intelligent insights!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Google API Key",
        type="password",
        value="",
        help="Enter your Google Generative AI API key"
    )
    
    # File upload
    st.markdown("### 📁 Data File")
    csv_file = st.text_input(
        "CSV File Path",
        value="cleaned_SIH_report_from_excel.csv",
        help="Enter the path to your CSV file"
    )
    
    # Initialize agent button
    if st.button("🚀 Initialize Agent", use_container_width=True):
        with st.spinner("Initializing agent..."):
            try:
                # Set environment variable
                os.environ["GOOGLE_API_KEY"] = api_key
                
                # Load dataframe
                st.session_state.df = pd.read_csv(csv_file)
                
                # Create LLM and agent
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
                st.session_state.agent = create_pandas_dataframe_agent(
                    llm=llm, 
                    df=st.session_state.df, 
                    verbose=True, 
                    allow_dangerous_code=True
                )
                
                st.success("✅ Agent initialized successfully!")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hello! I'm your data analysis assistant. I've loaded your CSV file and I'm ready to answer questions about your data. What would you like to know?",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
            except Exception as e:
                st.error(f"❌ Error initializing agent: {str(e)}")
    
    # Dataset info
    if st.session_state.df is not None:
        st.markdown("### 📊 Dataset Info")
        st.markdown(f"**Rows:** {len(st.session_state.df)}")
        st.markdown(f"**Columns:** {len(st.session_state.df.columns)}")
        
        with st.expander("View Column Names"):
            for col in st.session_state.df.columns:
                st.write(f"• {col}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "How many districts are there in bihar?",
        "What are the column names in this dataset?",
        "Show me basic statistics of the data",
        "What is the shape of this dataset?",
        "Show me the first 5 rows"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
            if st.session_state.agent is not None:
                st.session_state.messages.append({
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.rerun()

# Main chat interface
col1, col2 = st.columns([1, 3])

with col2:
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Welcome to the Data Analysis Chatbot!</h3>
                <p>Initialize the agent using the sidebar to start chatting about your data.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <small style="opacity: 0.7;">{message["timestamp"]}</small><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <small style="opacity: 0.7;">🤖 {message["timestamp"]}</small><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if st.session_state.agent is not None:
        with st.container():
            user_input = st.text_input(
                "Ask a question about your data...",
                key="user_input",
                placeholder="e.g., How many districts are there in bihar?",
                label_visibility="collapsed"
            )
            
            col_send, col_clear = st.columns([1, 4])
            with col_send:
                send_button = st.button("Send 📤", use_container_width=True)
            
            if send_button and user_input:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # Show typing indicator
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Get response from agent
                        response = st.session_state.agent.invoke(user_input)
                        
                        # Add bot response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["output"] if isinstance(response, dict) else str(response),
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                        
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Sorry, I encountered an error: {str(e)}",
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                
                # Clear input and refresh
                st.rerun()
    
    else:
        st.info("🔧 Please initialize the agent using the sidebar to start chatting!")

with col1:
    if st.session_state.df is not None:
        st.markdown("### 📈 Data Preview")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        st.markdown("### 📋 Quick Stats")
        st.write(f"**Total Rows:** {len(st.session_state.df)}")
        st.write(f"**Total Columns:** {len(st.session_state.df.columns)}")
        st.write(f"**Memory Usage:** {st.session_state.df.memory_usage().sum() / 1024:.1f} KB")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 Powered by LangChain + Google Gemini + Streamlit</p>
</div>
""", unsafe_allow_html=True)