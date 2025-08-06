import streamlit as st
import pandas as pd
import plotly.express as px
import random

from yahooquery import Ticker
import yahoo_fin.stock_info as si

from data_loader import load_qa
from embedding_store import init_embedding_store
from llm_model import load_llm
from rag import get_rag_answer

from numpy import where, unique
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore", category=Warning)
random.seed(42)

# --- Caching for efficient re-use
@st.cache_data
def get_dow_symbols():
    return si.tickers_dow()

@st.cache_data
def get_financial_df(stock_list):
    ticker = Ticker(stock_list)
    fin_data = ticker.financial_data
    df = pd.DataFrame.from_dict(fin_data, orient='index').T
    return df

# --- Sidebar: Stock Portfolio Management ---
st.sidebar.title("Stock Portfolio")
default_dow = get_dow_symbols()
user_stocks = st.sidebar.multiselect("Select stocks for your portfolio", options=default_dow, default=default_dow)
add_custom = st.sidebar.text_input("Add a custom stock symbol (e.g., TSLA):", "")
if add_custom and add_custom.upper() not in user_stocks:
    user_stocks.append(add_custom.upper())

# --- Main Title ---
st.title("📈 Lynch InvestBot: Chat, Ratios & Market Clusters")
st.markdown(
    "Chat with Peter Lynch's investing philosophy, analyze Dow Jones or custom stock ratios, and discover value/quality clusters for long/short ideas."
)

# --- 1. Peter Lynch Chatbot Section ---
st.header("1️⃣ Ask 'Peter Lynch'")
qa_data = load_qa()
embedder, collection = init_embedding_store(qa_data)
llm = load_llm()
with st.form("lynch_chat_form"):
    user_question = st.text_input("Enter your investing question for Peter Lynch:")
    submitted = st.form_submit_button("Ask!")
    if submitted and user_question.strip():
        answer, context = get_rag_answer(user_question, embedder, collection, llm)
        st.markdown(f"**Peter Lynch says:**\n\n{answer}")
        if context:
            with st.expander("Related Q&A Context"):
                for idx, qa in enumerate(context, 1):
                    st.markdown(f"**{idx}. {qa['question']}**\n*{qa['answer']}*")
    elif submitted and not user_question.strip():
        st.warning("Please enter a question.")

# --- 2. Financial Ratios Dashboard ---
st.header("2️⃣ Key Financial Ratios Dashboard")
fin_data_df = get_financial_df(user_stocks)

# --- Ratio descriptions ---
ratio_descriptions = {
    "trailingPE":       "Price-to-Earnings Ratio — The current share price relative to its per-share earnings.",
    "forwardPE":        "Forward P/E — Share price relative to predicted earnings (estimates).",
    "returnOnEquity":   "Return on Equity (ROE) — Efficiency at generating profit from shareholders’ equity.",
    "returnOnAssets":   "Return on Assets (ROA) — Profitability relative to all company assets.",
    "debtToEquity":     "Debt-to-Equity — Company's leverage: total liabilities divided by equity.",
    "dividendYield":    "Dividend Yield — Annual dividend as a % of stock price (shows income potential).",
    "currentRatio":     "Current Ratio — Ability to pay short-term obligations: current assets/current liabilities.",
    "quickRatio":       "Quick Ratio — Stricter liquidity: (current assets - inventories) / current liabilities.",
}

with st.expander("What do these ratios mean? (click to expand)"):
    for ratio, desc in ratio_descriptions.items():
        st.markdown(f"**{ratio}:** {desc}")

# Show only those ratios, handle missing gracefully
show_ratios = [r for r in ratio_descriptions if r in fin_data_df.index]
if show_ratios:
    st.dataframe(fin_data_df.loc[show_ratios].T.style.format('{:.2f}'))
else:
    st.warning("No financial ratios found for current selection. Try different or fewer stocks.")

# --- 3. K-Means Value-Quality Clustering and Recommendation ---
st.header("3️⃣ Value–Quality Stock Clustering & Recommendations")
if len(fin_data_df.columns) < 4:
    st.info("Select at least 4 stocks for meaningful clusters.")
else:
    # Preprocess: ratios, feature selection, impute/fix DE where needed
    temp_df = fin_data_df.copy()
    if "debtToEquity" in temp_df.index:
        temp_df.loc["debtToEquity"] = 1 / temp_df.loc["debtToEquity"].replace(0, float("nan"))

    data_full = temp_df.T.fillna(0)
    # Remove columns not for clustering (adapt to available cols):
    drop_cols = [c for c in [
        'maxAge','currentPrice','targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 
        'targetMedianPrice', 'recommendationMean', 'recommendationKey', 
        'numberOfAnalystOpinions','financialCurrency'
    ] if c in data_full.columns]
    data = data_full.drop(columns=drop_cols, errors="ignore").fillna(0)
    if data.shape[1] < 2:
        st.warning("Not enough financial features to cluster. Try different stock(s) or features.")
    else:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(data)
        n_clusters = 4 if X.shape[0] >= 4 else X.shape[0]  # Avoid error if few stocks
        model = KMeans(n_clusters=n_clusters, random_state=100)
        model.fit(X)
        yhat = model.predict(X)
        clusters = unique(yhat)

        # Interactive plot
        fig = px.scatter(
            x=X[:, 0], y=X[:, 1], color=[str(c) for c in yhat],
            hover_name=data.index,
            hover_data={
                "Stock": data.index,
                "Value": X[:, 0],
                "Quality": X[:, 1],
                "Cluster": yhat,
            },
            title="Value vs. Quality (K-Means Clustering)"
        )
        fig.update_layout(xaxis_title="Value Score", yaxis_title="Quality Score", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Show Long (cluster 3), Short (cluster 2) as in your logic
        st.success("#### Long Recommendation List (Cluster 3)")
        st.write(list(data.index[yhat == 3]) if 3 in clusters else "No stocks in this cluster.")
        st.error("#### Short Recommendation List (Cluster 2)")
        st.write(list(data.index[yhat == 2]) if 2 in clusters else "No stocks in this cluster.")

# --- Footer ---
st.caption("Built with Streamlit, yahooquery, yahoo_fin, ChromaDB, SentenceTransformer, FLAN-T5, and scikit-learn.")
