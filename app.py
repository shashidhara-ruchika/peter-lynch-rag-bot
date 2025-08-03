import streamlit as st
import logging
import warnings
from data_loader import load_qa, load_report_data
from embedding_store import init_embedding_store
from llm_model import load_llm
from rag import get_rag_answer

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("🚀 Starting Peter Lynch Trader Bot application...")

st.set_page_config(page_title="Peter Lynch Trader Bot", layout="wide")

logger.info("📱 Streamlit page configuration set")

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    Chat with the Peter Lynch-inspired AI bot and explore investment reports.
    Powered by open-source models and vector database.

    **Features:**
    - Peter Lynch Q&A chatbot (Retrieval-Augmented Generation)
    - Interactive stock/investment report viewer
    - Powered by Streamlit, ChromaDB, MiniLM/v2, and FLAN-T5
    """)

logger.info("📊 Loading Q&A data...")
# --- Load Data and Models ---
qa_data = load_qa()
logger.info(f"✅ Q&A data loaded successfully - {len(qa_data)} Q&A pairs")

logger.info("📈 Loading report data...")
report_df = load_report_data()
logger.info(f"✅ Report data loaded successfully - {len(report_df)} reports")

logger.info("🔍 Initializing embedding store...")
embedder, collection = init_embedding_store(qa_data)
logger.info("✅ Embedding store initialized successfully")

logger.info("🤖 Loading LLM model...")
llm = load_llm()
logger.info("✅ LLM model loaded successfully")

logger.info("🎯 All components loaded - Application ready!")

# --- Main: Reports Section ---
st.title("📊 Lynch InvestBot Reports & Chat")
st.subheader("Investment Reports")
st.dataframe(report_df)

# --- Chatbot Section ---
st.subheader("Ask Peter Lynch")

# Create two columns for input and button side by side
col1, col2 = st.columns([4, 1])

with col1:
    user_question = st.text_input("Enter your investing question here:", key="question_input", label_visibility="collapsed")

with col2:
    ask_button = st.button("Ask", type="primary", use_container_width=True)

# Process the question when button is clicked
if ask_button and user_question:
    logger.info(f"❓ User question received: {user_question[:50]}...")
    with st.spinner("Thinking like Peter Lynch..."):
        answer, context = get_rag_answer(user_question, embedder, collection, llm)
    logger.info("✅ RAG answer generated successfully")
    
    # Display the answer in a nice container
    st.markdown("### 🤖 Peter Lynch's Response")
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4;">
    {answer}
    </div>
    """, unsafe_allow_html=True)
    
    # Show context if available
    if context:
        st.markdown("### 📚 Related Questions & Answers")
        for idx, qa in enumerate(context, 1):
            with st.container():
                st.markdown(f"**{idx}. {qa['question']}**")
                st.markdown(f"*{qa['answer']}*")
                st.divider()

# Show a hint if button is clicked without question
elif ask_button and not user_question:
    st.warning("Please enter a question first!")

st.caption("Built with Streamlit, ChromaDB, SentenceTransformer, and FLAN-T5")
