import json
import pandas as pd
import streamlit as st
import logging
from config import QA_FILE

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=True)
def load_qa():
    logger.info(f"📖 Loading Q&A data from {QA_FILE}...")
    # Load CSV file with Peter Lynch Q&A data
    df = pd.read_csv(QA_FILE)
    logger.info(f"📊 CSV loaded with {len(df)} rows")
    
    # Convert DataFrame to list of dictionaries with 'question' and 'answer' keys
    qa_data = []
    for _, row in df.iterrows():
        qa_data.append({
            'question': row['Questions'],
            'answer': row['Answers']
        })
    
    logger.info(f"🔄 Converted {len(qa_data)} Q&A pairs to dictionary format")
    return qa_data
