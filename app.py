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


# --- Utilities ---
@st.cache_data
def get_dow_symbols():
    return si.tickers_dow()

@st.cache_data
def get_financial_df(stock_list):
    ticker = Ticker(stock_list)
    fin_data = ticker.financial_data
    df = pd.DataFrame.from_dict(fin_data, orient='index').T
    return df

# Ratio explanations (extend as needed)
ratio_descriptions = {
    "trailingPE":       "Price-to-Earnings (P/E): Current share price divided by earnings per share.",
    "forwardPE":        "Forward P/E: Share price divided by projected future earnings.",
    "returnOnEquity":   "Return on Equity (ROE): Efficiency at generating profit from equity.",
    "returnOnAssets":   "Return on Assets (ROA): Net income divided by total assets.",
    "debtToEquity":     "Debt-to-Equity (D/E): Company leverage, higher means more debt vs. equity.",
    "dividendYield":    "Dividend Yield: Annual dividend divided by share price.",
    "currentRatio":     "Current Ratio: Current assets/current liabilities (liquidity measure).",
    "quickRatio":       "Quick Ratio: Stricter liquidity ratio—(assets minus inventories)/liabilities.",
}

# --- 1. SIDEBAR: STOCK PORTFOLIO WITH SELECTBOX TO ADD ---
st.sidebar.title("Stock Portfolio")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = get_dow_symbols()

dow_symbols = get_dow_symbols()
# Stocks available to add = All Dow symbols minus already in portfolio
available_to_add = sorted(list(set(dow_symbols) - set(st.session_state.portfolio)))

# Stock selector dropdown to add (only one at a time)
selected_to_add = st.sidebar.selectbox("Select a stock to add to your portfolio:", options=available_to_add)

if st.sidebar.button("Add Stock"):
    if selected_to_add not in st.session_state.portfolio:
        st.session_state.portfolio.append(selected_to_add)
        st.sidebar.success(f"Added {selected_to_add}!")
    else:
        st.sidebar.info("Stock already in portfolio.")

# Multiselect to remove stocks from portfolio
user_stocks = st.sidebar.multiselect("Select stocks for your portfolio (remove by unselecting):",
                                     options=st.session_state.portfolio,
                                     default=st.session_state.portfolio)

# Update portfolio to match user's current selection in multiselect
if set(user_stocks) != set(st.session_state.portfolio):
    st.session_state.portfolio = user_stocks

# --- 2. MAIN TITLE & INSTRUCTIONS ---
st.title("📈 Lynch InvestBot: Chat, Ratios & Market Clusters")
st.markdown(
    "Chat with Peter Lynch's investing philosophy, analyze Dow Jones or custom stock ratios, and discover value/quality clusters for long/short ideas."
)

# --- 3. PETER LYNCH CHATBOT ---
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

# --- 4. FINANCIAL RATIOS DASHBOARD ---
st.header("2️⃣ Key Financial Ratios Dashboard")

fin_data_df = get_financial_df(st.session_state.portfolio)
available_fin_ratios = [r for r in ratio_descriptions if r in fin_data_df.index]

chosen_ratios = st.multiselect(
    "Select Key Financial Ratios to display:",
    options=available_fin_ratios,
    default=available_fin_ratios[:4],
    help="Choose which ratios to view. Explanations below."
)

if chosen_ratios:
    st.dataframe(
        fin_data_df.loc[chosen_ratios].T.style.format("{:.2f}"),
        use_container_width=True,
        hide_index=False
    )
else:
    st.warning("Select at least one financial ratio to display.")

with st.expander("🔍 Explanations for all financial ratios (click to expand)"):
    for ratio, desc in ratio_descriptions.items():
        st.markdown(f"**{ratio}:** {desc}")

# --- 5. K-MEANS CLUSTERING SECTION ---
st.header("3️⃣ Value–Quality Stock Clustering & Recommendations")

if len(fin_data_df.columns) < 4:
    st.info("Select at least 4 stocks for meaningful clusters.")
else:
    temp_df = fin_data_df.copy()
    if "debtToEquity" in temp_df.index:
        temp_df.loc["debtToEquity"] = 1 / temp_df.loc["debtToEquity"].replace(0, float("nan"))
    data_full = temp_df.T.fillna(0)
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
        n_clusters = 4 if X.shape[0] >= 4 else X.shape[0]
        model = KMeans(n_clusters=n_clusters, random_state=100)
        model.fit(X)
        yhat = model.predict(X)
        clusters = unique(yhat)
        
        # Hover data only: symbol, cluster num, value & quality rounded to 2 decimals
        hover_data = {
            "Stock": data.index,
            "Cluster": yhat,
            "Value": [round(v, 2) for v in X[:, 0]],
            "Quality": [round(q, 2) for q in X[:, 1]],
        }

        fig = px.scatter(
            x=X[:, 0], y=X[:, 1], color=[str(c) for c in yhat],
            hover_name=data.index,
            hover_data=hover_data,
            title="Value vs. Quality (K-Means Clustering)"
        )
        fig.update_layout(xaxis_title="Value Score", yaxis_title="Quality Score", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Pretty display of stock symbols in bullet lists for Long (cluster 3) and Short (cluster 2)
        def display_stock_list(cluster_num, cluster_name):
            stocks = list(data.index[yhat == cluster_num])
            st.write(f"**{cluster_name} ({len(stocks)} stocks):**")
            if stocks:
                for s in stocks:
                    st.write(f"- {s}")
            else:
                st.write("_None._")

        st.success("Long Recommendation List (Cluster 3):")
        display_stock_list(3, "Long")

        st.error("Short Recommendation List (Cluster 2):")
        display_stock_list(2, "Short")


# --- Footer ---
st.caption("Built with Streamlit, yahooquery, yahoo_fin, ChromaDB, SentenceTransformer, FLAN-T5, and scikit-learn.")
