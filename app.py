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

# Full list of ratios including financialCurrency
all_financial_ratios = [
    "maxAge",
    "currentPrice",
    "targetHighPrice",
    "targetLowPrice",
    "targetMeanPrice",
    "targetMedianPrice",
    "recommendationMean",
    "recommendationKey",
    "numberOfAnalystOpinions",
    "totalCash",
    "totalCashPerShare",
    "ebitda",
    "totalDebt",
    "quickRatio",
    "currentRatio",
    "totalRevenue",
    "debtToEquity",
    "revenuePerShare",
    "returnOnAssets",
    "returnOnEquity",
    "grossProfits",
    "freeCashflow",
    "operatingCashflow",
    "earningsGrowth",
    "revenueGrowth",
    "grossMargins",
    "ebitdaMargins",
    "operatingMargins",
    "profitMargins",
    "financialCurrency",
]

# Ratio explanations (extend as needed)
ratio_descriptions = {
    "maxAge": "Max age of the data point (days)",
    "currentPrice": "Current market price",
    "targetHighPrice": "Analyst target high price",
    "targetLowPrice": "Analyst target low price",
    "targetMeanPrice": "Analyst mean target price",
    "targetMedianPrice": "Analyst median target price",
    "recommendationMean": "Mean analyst recommendation",
    "recommendationKey": "Key analyst recommendation",
    "numberOfAnalystOpinions": "Number of analyst opinions",
    "totalCash": "Total cash held by company",
    "totalCashPerShare": "Total cash per share",
    "ebitda": "Earnings before interest, taxes, depreciation, and amortization",
    "totalDebt": "Total debt",
    "quickRatio": "Quick ratio liquidity measure",
    "currentRatio": "Current ratio liquidity measure",
    "totalRevenue": "Total revenue",
    "debtToEquity": "Debt to equity ratio",
    "revenuePerShare": "Revenue per share",
    "returnOnAssets": "Return on assets",
    "returnOnEquity": "Return on equity",
    "grossProfits": "Gross profits",
    "freeCashflow": "Free cash flow",
    "operatingCashflow": "Operating cash flow",
    "earningsGrowth": "Earnings growth rate",
    "revenueGrowth": "Revenue growth rate",
    "grossMargins": "Gross margin percentage",
    "ebitdaMargins": "EBITDA margin percentage",
    "operatingMargins": "Operating margin percentage",
    "profitMargins": "Profit margin percentage",
    "financialCurrency": "Currency of financial values",
}

# --- 1. SIDEBAR: STOCK PORTFOLIO WITH SELECTBOX TO ADD ---
import streamlit as st
from yahooquery import Ticker
import yahoo_fin.stock_info as si

# Initialize portfolio in session_state once
if "portfolio" not in st.session_state:
    st.session_state.portfolio = si.tickers_dow()

if "stock_to_remove" not in st.session_state:
    st.session_state.stock_to_remove = None

def stock_exists(symbol: str) -> bool:
    """Check if a given stock symbol exists by querying yahooquery."""
    symbol = symbol.upper()
    try:
        t = Ticker(symbol)
        info = t.quote_type
        if symbol in info and info[symbol].get("quoteType"):
            return True
        else:
            return False
    except Exception:
        return False

st.sidebar.title("Stock Portfolio")

# --- Add Stock Form ---
with st.sidebar.form("add_stock_form", clear_on_submit=True):
    new_symbol = st.text_input("Type any stock symbol to add (e.g., TSLA):", max_chars=10)
    add_button = st.form_submit_button("Add Stock")

    if add_button:
        new_symbol = new_symbol.strip().upper()
        if not new_symbol:
            st.sidebar.warning("Please enter a stock symbol.")
        elif new_symbol in st.session_state.portfolio:
            st.sidebar.info(f"{new_symbol} is already in your portfolio.")
        elif stock_exists(new_symbol):
            st.session_state.portfolio.append(new_symbol)
            st.sidebar.success(f"Added {new_symbol} to portfolio!")
        else:
            st.sidebar.error(f"Stock symbol '{new_symbol}' does not exist. Please check and try again.")

# --- Display Current Portfolio with "x" Buttons for Removal ---
st.sidebar.write("### Current Portfolio (remove stocks by clicking 'x')")

user_stocks = st.sidebar.multiselect(
    "Select stocks for your portfolio (remove by unselecting):",
    options=st.session_state.portfolio,
    default=st.session_state.portfolio,
    help="Unselect stocks here to remove them from your portfolio."
)

# Update portfolio if user unselects any stocks
if set(user_stocks) != set(st.session_state.portfolio):
    st.session_state.portfolio = user_stocks

# --- 2. MAIN TITLE & INSTRUCTIONS ---
st.title("📈 Your Investments: Chat, Ratios & Market Clusters")
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
st.header("2️⃣ Financial Ratios Dashboard")

fin_data_df = get_financial_df(st.session_state.portfolio)
available_fin_ratios = [r for r in ratio_descriptions if r in fin_data_df.index]

chosen_ratios = st.multiselect(
    "Select Key Financial Ratios to display:",
    options=available_fin_ratios,
    default=["debtToEquity", "currentRatio", "returnOnEquity", "profitMargins", "revenueGrowth", "freeCashflow"],
    help="Choose which ratios to display"
)

with st.sidebar.expander("🔍 Explanations for all financial ratios (click to expand)"):
    for ratio, desc in ratio_descriptions.items():
        st.markdown(f"**{ratio}:** {desc}")


if chosen_ratios:
    with st.container():
        # st.markdown("### Financial Ratios Table (Scrollable)")
        # Use style to fix height and make scroll vertical
        styled_df = fin_data_df.loc[chosen_ratios].T.style.set_table_attributes(
            'style="max-height:400px; overflow-y:auto; display:block;"'
        )
        st.dataframe(styled_df, use_container_width=True)
else:
    st.warning("Select at least one financial ratio to display.")


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
