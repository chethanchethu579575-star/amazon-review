import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# PAGE CONFIGURATIONS & STYLES
# ==========================================
st.set_page_config(page_title="Amazon Review Analytics Dashboard", layout="wide")

st.title("📊 Amazon Product Reviews Sentiment Analytics Dashboard")
st.write("Upload your historical Amazon Reviews dataset file below to generate instant sentiment visualizations.")

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
# FILE UPLOADER INTERFACE
# ==========================================
st.markdown("### 📁 Upload Dataset")
uploaded_file = st.file_uploader("Drag and drop your Amazon Reviews CSV file here", type=["csv"])

# ==========================================
# ANALYSIS & RENDERING PIPELINE
# ==========================================
if uploaded_file is not None:
    st.markdown("---")
    with st.spinner("Processing dataset rows (Optimized Sample)..."):
        # Automatically read the first 50,000 rows to ensure blazing fast cloud performance
        raw_df = pd.read_csv(uploaded_file, nrows=50000)
        
        # Standardize columns dynamically
        if 'ProductId' in raw_df.columns:
            df = raw_df[['ProductId', 'Text', 'Score']].copy()
        else:
            df = raw_df.copy()
            
        df.columns = ['product_id', 'review_text', 'rating']
        
        # Filter out neutral reviews (3-star)
        df = df[df['rating'] != 3]
        
        # Clean text and run NLP predictions
        df['clean_review'] = df['review_text'].apply(clean_text)
        vectorized = tfidf.transform(df['clean_review'])
        df['predicted_sentiment'] = model.predict(vectorized)
        
    # 📈 EXECUTIVE METRICS SCORECARDS
    st.subheader("📈 Overall Analytical Performance Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    total_count = len(df)
    pos_count = len(df[df['predicted_sentiment'] == 1])
    neg_count = len(df[df['predicted_sentiment'] == 0])
    
    m_col1.metric("Total Reviews Processed", f"{total_count:,}")
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
        df['Sentiment Status'] = df['predicted_sentiment'].map({1: "🟢 Positive", 0: "🔴 Negative"})
        fig_bar = px.histogram(df, x="Sentiment Status", color="Sentiment Status",
                               title="Sentiment Count Comparison",
                               color_discrete_map={"🟢 Positive": '#2ecc71', "🔴 Negative": '#e74c3c'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # 🔍 DATA GRID LIST
    st.subheader("🔍 Explore Categorized Review Outputs")
    st.dataframe(df[['Sentiment Status', 'product_id', 'review_text']].head(100), use_container_width=True)

else:
    st.markdown("---")
    st.info("💡 Dashboard is waiting for data. Drop your CSV file into the box above to generate the analytical metrics charts!")
