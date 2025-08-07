import streamlit as st
import logging
from transformers import pipeline
from config import LLM_MODEL_NAME

logger = logging.getLogger(__name__)

@st.cache_resource
def load_llm():
    logger.info(f"🤖 Loading LLM model: {LLM_MODEL_NAME}...")
    # device=-1 for CPU, 0 for GPU; adjust as needed
    model = pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        tokenizer=LLM_MODEL_NAME,
        device=0   # Change to 0 if using GPU
    )
    logger.info("✅ LLM model loaded successfully")
    return model
