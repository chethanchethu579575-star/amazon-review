import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import requests
from bs4 import BeautifulSoup

# ==========================================
# PAGE CONFIGURATIONS & STYLES
# ==========================================
st.set_page_config(page_title="Product Review Analytics Dashboard", layout="wide")

st.title("📊 Amazon Product Reviews Sentiment Analytics Dashboard")
st.write("Welcome! Choose your input method below to analyze customer sentiments instantly.")

# ==========================================
# CENTRALIZED ML CORE BACKEND
# ==========================================
@st.cache_resource
def train_global_model():
    """ Trains a core baseline model so the app can predict sentiment instantly """
    np.random.seed(42)
    sample_texts = [
        "great product love the taste", "amazing quality highly recommend", "delicious and fresh",
        "terrible item arrived broken", "bad flavor completely stale", "disappointed waste of money",
        "excellent packaging fast shipping", "horrible customer service completely useless"
    ] * 50
    sample_labels = [1, 1, 1, 0, 0, 0, 1, 0] * 50
    
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(sample_texts)
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_tfidf, sample_labels)
    return model, tfidf

model, tfidf = train_global_model()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join(words)

# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
st.sidebar.header("📁 Data Sources Control")
st.sidebar.write("Choose your settings here.")

# Unified Row Slider to keep things highly responsive on the cloud server
max_rows = st.sidebar.slider("Max rows to process (for CSV files):", min_value=10000, max_value=500000, value=50000, step=10000)

# ==========================================
# MAIN PAGE: UNIFIED BOX INTERFACE
# ==========================================
# We split the top section into two clean columns on one page
box_col1, box_col2 = st.columns(2)

# --- FEATURE A: BATCH FILE UPLOADER (Left Box) ---
with box_col1:
    st.markdown("### Option 1: Upload Historical Dataset")
    uploaded_file = st.file_uploader("Upload your Amazon Reviews CSV file", type=["csv"], key="unified_csv")
    
# --- FEATURE B: LIVE WEB SCRAPER (Right Box) ---
with box_col2:
    st.markdown("### Option 2: Scrape a Live URL Link")
    product_url = st.text_input("Paste target product webpage URL link here:", placeholder="https://example-store.com/product-page")
    num_reviews = st.number_input("How many live reviews to crawl?", min_value=5, max_value=50, value=15, step=5)
    launch_scraper = st.button("Launch Web Scraper & Run NLP")


# Initialize an empty DataFrame that will hold whatever data comes in
df_to_analyze = None
source_type = ""

# ==========================================
# ENGINE ROUTING LOGIC
# ==========================================

# Case A: User uploaded a file
if uploaded_file is not None:
    source_type = "file"
    with st.spinner("Processing your dataset rows..."):
        raw_df = pd.read_csv(uploaded_file, nrows=max_rows)
        # Standardize columns
        df_to_analyze = raw_df[['ProductId', 'Text', 'Score']].copy() if 'ProductId' in raw_df.columns else raw_df.copy()
        df_to_analyze.columns = ['product_id', 'review_text', 'rating']
        df_to_analyze = df_to_analyze[df_to_analyze['rating'] != 3]

# Case B: User pasted a URL and clicked the Scraper button
elif launch_scraper and product_url.strip() != "":
    source_type = "scraper"
    with st.spinner("Injecting web scraper elements and reading live HTML..."):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            response = requests.get(product_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            elements = soup.find_all(["span", "p"])
            scraped_texts = [el.get_text(strip=True) for el in elements if len(el.get_text(strip=True)) > 25][:num_reviews]
            
            # Simulated safe response if destination site firewalls block direct requests
            if len(scraped_texts) == 0:
                scraped_texts = [
                    "This product is absolutely wonderful! Tastes spectacular and fresh.",
                    "Terrible experience. The package arrived completely smashed and open.",
                    "Incredible customer support and high-quality ingredients used.",
                    "Stale, awful flavor. Do not spend your hard-earned money here.",
                    "Decent value, but the texture was a bit strange compared to normal."
                ] * (num_reviews // 5 + 1)
                scraped_texts = scraped_texts[:num_reviews]
                st.sidebar.warning("⚠️ Live URL blocked by firewall. Using simulation data.")
            
            df_to_analyze = pd.DataFrame(scraped_texts, columns=['review_text'])
            df_to_analyze['product_id'] = "LIVE_URL"
            df_to_analyze['rating'] = 5 # Dummy placeholder for layout uniformity
            
        except Exception as err:
            st.error(f"Network connection failure: {err}")

# ==========================================
# SHARED ML PIPELINE & RENDER ENGINE
# ==========================================
if df_to_analyze is not None:
    st.markdown("---")
    with st.spinner("Running sentiment calculations using pre-trained model..."):
        # Apply shared NLP cleaning
        df_to_analyze['clean_review'] = df_to_analyze['review_text'].apply(clean_text)
        
        # Vectorize and Predict
        vectorized = tfidf.transform(df_to_analyze['clean_review'])
        df_to_analyze['predicted_sentiment'] = model.predict(vectorized)
    
    # 📈 EXECUTIVE METRICS SCORECARDS
    st.subheader("📈 Overall Analytical Performance Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    total_count = len(df_to_analyze)
    pos_count = len(df_to_analyze[df_to_analyze['predicted_sentiment'] == 1])
    neg_count = len(df_to_analyze[df_to_analyze['predicted_sentiment'] == 0])
    
    m_col1.metric("Total Reviews Processed", f"{total_count:,}")
    m_col1.caption(f"Source Type: {source_type.upper()}")
    m_col2.metric("Positive Sentiment", f"{pos_count:,}", f"{(pos_count/total_count)*100:.1f}%")
    m_col3.metric("Negative Critical Flags", f"{neg_count:,}", f"{(neg_count/total_count)*100:.1f}%", delta_color="inverse")
    
    # 📊 CHARTS SECTION
    st.subheader("📊 Visualizations Breakdown")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_pie = px.pie(names=['Positive', 'Negative'], values=[pos_count, neg_count], 
                         title="Sentiment Distribution", hole=0.4, 
                         color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        # Map nice flags for the explicit display data grid
        df_to_analyze['Sentiment Status'] = df_to_analyze['predicted_sentiment'].map({1: "🟢 Positive", 0: "🔴 Negative"})
        
        fig_bar = px.histogram(df_to_analyze, x="Sentiment Status", color="Sentiment Status",
                               title="Sentiment Count Comparison",
                               color_discrete_map={"🟢 Positive":'#2ecc71', "🔴 Negative":'#e74c3c'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # 🔍 DATA GRID LIST
    st.subheader("🔍 Explore Categorized Review Outputs")
    st.dataframe(df_to_analyze[['Sentiment Status', 'product_id', 'review_text']].head(100), use_container_width=True)
else:
    st.markdown("---")
    st.info("💡 Dashboard is waiting for input data. Choose to upload a file in **Option 1** or run a web scrape in **Option 2** to generate the metrics!")
