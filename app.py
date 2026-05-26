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

# Set up page configurations
st.set_page_config(page_title="Amazon Review Batch Dashboard", layout="wide")

st.title("📊 Amazon Product Reviews Batch Analytics Dashboard")
st.write("Upload your full `Reviews.csv` file to analyze overall sentiments, top products, and customer trends.")

# ==========================================
# CACHED MACHINE LEARNING PIPELINE
# ==========================================
# We cache this function so the app doesn't re-train the model every time you click a button
@st.cache_resource
def train_pipeline_model(df_clean):
    X = df_clean['clean_review']
    y = df_clean['sentiment']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Model Training
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    return model, tfidf

@st.cache_data
def clean_text_batch(series_data):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    def clean_single(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        return ' '.join([w for w in words if w not in stop_words])
    
    return series_data.apply(clean_single)

# ==========================================
# FILE UPLOADER SIDEBAR
# ==========================================
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your Amazon Reviews CSV file", type=["csv"])

if uploaded_file is not None:
    # Read data
    with st.spinner("Reading CSV file... Please wait."):
        # reading a smaller chunk or full file. For testing, we can read a subset if memory is low
        raw_df = pd.read_csv(uploaded_file)
    
    st.sidebar.success(f"Successfully loaded {len(raw_df):,} reviews!")
    
    # --- PREPROCESSING DATA ---
    with st.spinner("Processing text and running sentiment calculations..."):
        # Create a working copy
        df = raw_df[['ProductId', 'Text', 'Score']].copy()
        df.columns = ['product_id', 'review_text', 'rating']
        
        # Filter neutral reviews and create sentiment column
        df = df[df['rating'] != 3]
        df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
        
        # Clean the text columns in batch
        df['clean_review'] = clean_text_batch(df['review_text'])
        df = df[df['clean_review'].str.strip() != '']
        
        # Train or load ML Pipeline on this dataset
        model, tfidf = train_pipeline_model(df)
        
        # Predict sentiments for the entire dataset to cross-verify model alignment
        vectorized_full = tfidf.transform(df['clean_review'])
        df['predicted_sentiment'] = model.predict(vectorized_full)
    
    # ==========================================
    # DASHBOARD METRICS SECTION
    # ==========================================
    st.subheader("📈 Overall Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_reviews = len(df)
    pos_reviews = len(df[df['predicted_sentiment'] == 1])
    neg_reviews = len(df[df['predicted_sentiment'] == 0])
    pos_percent = (pos_reviews / total_reviews) * 100
    
    col1.metric("Total Analyzed Reviews", f"{total_reviews:,}")
    col2.metric("Predicted Positive Reviews", f"{pos_reviews:,}", f"{pos_percent:.1f}%")
    col3.metric("Predicted Negative Reviews", f"{neg_reviews:,}", f"{(100-pos_percent):.1f}%", delta_color="inverse")
    col4.metric("Average Star Rating", f"⭐ {df['rating'].mean():.2f}")
    
    st.markdown("---")
    
    # ==========================================
    # VISUALIZATIONS SECTION
    # ==========================================
    st.subheader("📊 Sentiment Visualizations")
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown("#### Sentiment Distribution Split")
        sentiment_counts = df['predicted_sentiment'].map({1: 'Positive', 0: 'Negative'}).value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', color='Sentiment',
                         color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'}, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with v_col2:
        st.markdown("#### Ratings Breakdown vs Predicted Sentiment")
        fig_bar = px.histogram(df, x="rating", color=df['predicted_sentiment'].map({1: 'Positive', 0: 'Negative'}),
                               labels={'color': 'Predicted Sentiment', 'rating': 'Star Rating'},
                               color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'}, barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    
    # ==========================================
    # ADVANCED BUSINESS INSIGHTS
    # ==========================================
    st.subheader("🏪 Product-Level Deep Dive")
    bi_col1, bi_col2 = st.columns(2)
    
    with bi_col1:
        st.markdown("#### Top 5 Most Positively Reviewed Products")
        pos_df = df[df['predicted_sentiment'] == 1]
        top_pos = pos_df['product_id'].value_counts().head(5).reset_index()
        top_pos.columns = ['Product ID', 'Positive Review Count']
        st.dataframe(top_pos, use_container_width=True)
        
    with bi_col2:
        st.markdown("#### Top 5 Most Critically Criticized Products (Action Required!)")
        neg_df = df[df['predicted_sentiment'] == 0]
        top_neg = neg_df['product_id'].value_counts().head(5).reset_index()
        top_neg.columns = ['Product ID', 'Negative Review Count']
        st.dataframe(top_neg, use_container_width=True)

    st.markdown("---")
    
    # ==========================================
    # RAW DATA BROWSER
    # ==========================================
    st.subheader("🔍 Explore Categorized Reviews Data")
    sentiment_filter = st.selectbox("Filter Data Table By Sentiment:", ["All", "Positive Only", "Negative Only"])
    
    display_df = df[['product_id', 'rating', 'predicted_sentiment', 'review_text']]
    display_df['predicted_sentiment'] = display_df['predicted_sentiment'].map({1: '🔥 Positive', 0: '⚠️ Negative'})
    
    if sentiment_filter == "Positive Only":
        display_df = display_df[display_df['predicted_sentiment'] == '🔥 Positive']
    elif sentiment_filter == "Negative Only":
        display_df = display_df[display_df['predicted_sentiment'] == '⚠️ Negative']
        
    st.dataframe(display_df.head(100), use_container_width=True)

else:
    # Default message when app is waiting for user uploads
    st.info("💡 Please upload the `Reviews.csv` file from your sidebar panel to generate the automated batch metrics report.")