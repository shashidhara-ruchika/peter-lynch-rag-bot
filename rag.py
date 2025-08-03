import logging
import re
from config import TOP_K, MAX_NEW_TOKENS

logger = logging.getLogger(__name__)

def preprocess_query(query):
    """
    Preprocess the query to replace direct references to the bot with "Peter Lynch"
    to improve context matching in the RAG system.
    """
    # Convert to lowercase for matching
    query_lower = query.lower()
    
    # Define patterns to replace with "Peter Lynch"
    replacements = [
        (r'\b(you|your|yours)\b', 'Peter Lynch'),
        (r'\b(peter|lynch)\b', 'Peter Lynch'),
        (r'\b(his|him|he)\b', 'Peter Lynch'),
        (r'\b(this investor|this fund manager)\b', 'Peter Lynch'),
    ]
    
    # Apply replacements
    processed_query = query
    for pattern, replacement in replacements:
        processed_query = re.sub(pattern, replacement, processed_query, flags=re.IGNORECASE)
    
    # Clean up multiple "Peter Lynch" occurrences
    processed_query = re.sub(r'Peter Lynch\s+Peter Lynch', 'Peter Lynch', processed_query)
    processed_query = re.sub(r'Peter Lynch\s+and\s+Peter Lynch', 'Peter Lynch', processed_query)
    
    # Enhanced logging with clear formatting
    # logger.info("=" * 60)
    # logger.info("🔍 QUERY PREPROCESSING")
    # logger.info("=" * 60)
    logger.info(f"📝 Original Query:  '{query}'")
    logger.info(f"🔄 Processed Query: '{processed_query}'")
    # logger.info("=" * 60)
    
    return processed_query

def get_rag_answer(query, embedder, collection, llm):
    logger.info(f"🔍 Processing query: {query[:50]}...")
    
    # Preprocess the query
    processed_query = preprocess_query(query)
    
    logger.info("📊 Encoding query into embedding...")
    query_embedding = embedder.encode([processed_query])[0].tolist()
    logger.info("✅ Query encoded successfully")
    
    logger.info(f"🔎 Searching for top {TOP_K} similar Q&A pairs...")
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    logger.info(f"✅ Found {len(results['metadatas'][0]) if results['metadatas'] else 0} relevant Q&A pairs")

    context_data = results['metadatas'][0] if results['metadatas'] else []
    if not context_data:
        logger.warning("⚠️ No relevant context found for query")
        return "I'm sorry, but I don't have enough information to give you a proper answer to that question. Could you ask me something about my investment philosophy or strategies?", []

    logger.info("📝 Building context from relevant Q&A pairs...")
    context_text = "\n\n".join(
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in context_data
    )

    logger.info("🎭 Creating Peter Lynch persona prompt...")
    prompt = (
        "You are Peter Lynch"
        "Be conversational, wise, and share your personal experiences and insights. "
        "Use the following context from your own words and experiences to answer the question:\n\n"
        f"{context_text}\n\n"
        f"Investor's question: {query}\n"
        "Peter Lynch:"
    )

    logger.info("🤖 Generating response using LLM...")
    outputs = llm(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_p=0.9, top_k=40, temperature=0.7)
    answer = outputs[0]['generated_text'].strip()
    logger.info("✅ Response generated successfully")
    
    return answer, context_data
