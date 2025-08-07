import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import logging
from config import CHROMA_PERSIST_DIR, SENTENCE_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def init_embedding_store(qa_data):
    logger.info("🔧 Initializing SentenceTransformer model...")
    embedder = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)
    logger.info("✅ SentenceTransformer model loaded")
    
    logger.info(f"🗄️ Connecting to ChromaDB at {CHROMA_PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    logger.info("✅ ChromaDB client connected")
    
    # Load or create collection
    existing_collections = [c.name for c in client.list_collections()]
    logger.info(f"📚 Existing collections: {existing_collections}")
    
    if "lynch_qa" in existing_collections:
        logger.info("📖 Loading existing 'lynch_qa' collection...")
        collection = client.get_collection(name="lynch_qa")
        logger.info("✅ Existing collection loaded")
    else:
        logger.info("🆕 Creating new 'lynch_qa' collection...")
        collection = client.create_collection(name="lynch_qa")
        logger.info("✅ New collection created")
        
        logger.info("🔤 Encoding questions into embeddings...")
        questions = [qa['question'] for qa in qa_data]
        embeddings = embedder.encode(questions, show_progress_bar=True).tolist()
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        
        logger.info("💾 Adding embeddings to collection...")
        ids = [f"qa-{i}" for i in range(len(qa_data))]
        collection.add(
            embeddings=embeddings,
            documents=[qa['answer'] for qa in qa_data],
            metadatas=qa_data,
            ids=ids
        )
        logger.info(f"✅ Added {len(ids)} Q&A pairs to collection")
    
    return embedder, collection
