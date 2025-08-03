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

@st.cache_data(show_spinner=True)
def load_report_data():
    logger.info("📈 Loading investment report data...")
    # Replace with your actual report CSV or data source if needed
    data = {
        "Ticker": ["AAPL", "AMGN", "AMZN", "AXP", "BA"],
        "Price": [214.10, 315.04, 194.95, 270.83, 172.83],
        "Recommendation": ["Buy", "Buy", "Strong Buy", "Hold", "Buy"],
        "EarningsGrowth": [0.10, 0.08, 0.15, 0.05, 0.07],
        "PE_Ratio": [19, 25, 32, 16, 20],
    }
    df = pd.DataFrame(data)
    logger.info(f"📊 Report data loaded with {len(df)} stocks")
    return df
