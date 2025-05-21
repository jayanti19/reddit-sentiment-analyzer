import streamlit as st
import pandas as pd
import numpy as np
import praw
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page configuration
st.set_page_config(
    page_title="Reddit Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better visual appearance
st.markdown("""
<style>
    /* Main styles */
    body {
        font-family: 'Arial', sans-serif;
    }
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        font-weight: 600;
    }
    
    /* Sentiment cards */
    .sentiment-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        height: 100%;
        border: 1px solid rgba(0,0,0,0.1);
    }
    .sentiment-card:hover {
        transform: translateY(-5px);
    }
    
    /* Text colors for better visibility */
    .sentiment-card h1, .sentiment-card h2 {
        font-weight: 600;
        text-shadow: 0px 0px 1px rgba(255,255,255,0.5);
    }
    .sentiment-card p {
        font-weight: 500;
    }
    
    /* Comments styling - removed as per request */
    
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 0.75rem;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 0.5rem;
    }
    
    /* Loading spinner */
    .stSpinner > div > div > div {
        border-color: #4CAF50 #f3f3f3 #f3f3f3 !important;
    }
    
    /* Header section */
    .header-container {
        background: linear-gradient(90deg, rgba(76,175,80,0.2) 0%, rgba(255,255,255,0) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 5px solid #4CAF50;
    }
    
    /* Results section */
    .results-container {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        border: 1px solid #eaeaea;
    }
    .results-container h2 {
        color: #333;
        font-weight: 600;
    }
    
    /* Dominant sentiment */
    .dominant-card {
        background: linear-gradient(120deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load tokenizer, encoder, and model
@st.cache_resource
def load_ml_components():
    try:
        tokenizer = joblib.load("tokenizer.pkl")
        encoder = joblib.load("label_encoder.pkl")
        model = load_model("sentiment_cnn_model.h5")
        return tokenizer, encoder, model
    except Exception as e:
        st.error(f"Error loading ML components: {e}")
        return None, None, None

tokenizer, encoder, model = load_ml_components()

# Emoji and color maps
emoji_map = {
    "positive": "😊",
    "negative": "😠",
    "neutral": "😐"
}
color_map = {
    "positive": "#e6f7e6",  # lighter green for better text visibility
    "negative": "#ffe6e6",  # lighter red for better text visibility
    "neutral": "#fff9e6"    # lighter yellow for better text visibility
}
text_color_map = {
    "positive": "#155724",  # dark green
    "negative": "#721c24",  # dark red
    "neutral": "#856404"    # dark yellow+
}

# Function to scrape posts and comments
@st.cache_data(ttl=300)
def scrape_subreddit_posts(subreddit_name):
    reddit = praw.Reddit(
        client_id='DZN8fwVqoCCpoYRfraQB2w',
        client_secret='43fzFXo_MawKif395JyM7HtBVxK-7w',
        user_agent='Jayanti Dwivedi'
    )

    data = []
    try:
        for submission in reddit.subreddit(subreddit_name).hot(limit=10):
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:40]:
                if hasattr(comment, "body"):
                    data.append({
                        "title": submission.title,
                        "comment_text": comment.body
                    })
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    
    return pd.DataFrame(data)

# Sentiment prediction function
@st.cache_data
def predict_sentiment(texts):
    if not tokenizer or not encoder or not model:
        st.error("ML components not loaded properly")
        return ["error"] * len(texts)
        
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=300)
    preds = model.predict(padded)
    labels = encoder.inverse_transform(np.argmax(preds, axis=1))
    return [label.lower() for label in labels]  # ensure lowercase match

# Header Section
st.markdown("""
<div class="header-container">
    <h1 style='text-align: center; margin-bottom: 20px;'>💬 Reddit Sentiment Analyzer</h1>
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
        Analyze the sentiment of comments on Reddit posts using deep learning (CNN)
    </p>
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # Subreddit input
    st.markdown("### 🔎 Enter Subreddit")
    subreddit_input = st.text_input(
        "Enter subreddit name (without r/):",
        key="sub_input",
        placeholder="Example: python, news, datascience"
    )

with col2:
    st.markdown("### ℹ️ About")
    st.info("""
    This tool analyzes the sentiment of 
    comments on Reddit posts as Positive,
    Neutral, or Negative using a CNN model.
    """)

# Process data
if subreddit_input:
    with st.spinner("Fetching data from Reddit..."):
        df = scrape_subreddit_posts(subreddit_input)

    if not df.empty:
        st.markdown("### 📝 Select a Post")
        unique_titles = df["title"].unique()
        
        # Calculate comment counts for each post
        title_counts = df["title"].value_counts().to_dict()
        title_options = [f"{title} ({title_counts[title]} comments)" for title in unique_titles]
        
        selected_title_option = st.selectbox(
            "Choose a post to analyze:",
            title_options,
            index=0,
            help="Select a post to analyze its comments' sentiment"
        )
        
        # Extract the actual title from the selection
        selected_title = selected_title_option.rsplit(" (", 1)[0]

        # Create two columns for button and loading indicator
        analyze_col, status_col = st.columns([3, 1])
        
        with analyze_col:
            # Button to analyze
            analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)
            
        if analyze:
            with status_col:
                with st.spinner(""):
                    st.markdown("⏳ **Analyzing...**")
            
            filtered_df = df[df["title"] == selected_title].copy()

            with st.spinner("Processing sentiments..."):
                # Predict sentiments for the comments
                filtered_df["sentiment"] = predict_sentiment(filtered_df["comment_text"])
                sentiment_counts = filtered_df["sentiment"].value_counts().to_dict()

                # Ensure all 3 sentiments are present
                for sentiment in ["positive", "neutral", "negative"]:
                    if sentiment not in sentiment_counts:
                        sentiment_counts[sentiment] = 0

                # Create results container
                st.markdown("""
                <div class="results-container">
                    <h2 style='text-align: center; margin-bottom: 20px; color: #333;'>📊 Analysis Results</h2>
                </div>
                """, unsafe_allow_html=True)

                # Determine dominant sentiment
                dominant = max(sentiment_counts, key=sentiment_counts.get)
                total_comments = sum(sentiment_counts.values())
                
                # Calculate percentages
                percentages = {
                    sentiment: round((count / total_comments) * 100, 1) 
                    for sentiment, count in sentiment_counts.items()
                }

                # Most dominant sentiment card
                st.markdown(f"""
                    <div class="dominant-card">
                        <h3 style="color: #333;">Overall Sentiment</h3>
                        <h1 style="color: {text_color_map[dominant]}; font-size: 3rem;">
                            {dominant.capitalize()} {emoji_map[dominant]}
                        </h1>
                        <p style="color: #666; font-size: 1.2rem;">
                            {percentages[dominant]}% of comments show {dominant} sentiment
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # Sentiment cards
                st.markdown("### 🔢 Sentiment Distribution")
                col1, col2, col3 = st.columns(3)
                for sentiment, col in zip(["positive", "neutral", "negative"], [col1, col2, col3]):
                    count = sentiment_counts[sentiment]
                    bg = color_map[sentiment]
                    text_color = text_color_map[sentiment]
                    emoji = emoji_map[sentiment]
                    col.markdown(f"""
                        <div class="sentiment-card" style="background-color: {bg};">
                            <h2 style="color: {text_color};">{emoji} {sentiment.capitalize()}</h2>
                            <h1 style="color: {text_color}; font-size: 2.5rem;">{count}</h1>
                            <p style="color: {text_color}; font-weight: bold;">{percentages[sentiment]}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Display sentiment chart
                st.markdown("### 📈 Sentiment Visualization")
                chart_data = pd.DataFrame({
                    'Sentiment': list(sentiment_counts.keys()),
                    'Count': list(sentiment_counts.values())
                })
                chart_data = chart_data.sort_values('Sentiment')
                
                # Use custom colors for the chart
                chart_colors = [color_map["positive"], color_map["neutral"], color_map["negative"]]
                
                # Plot the chart
                st.bar_chart(
                    chart_data.set_index('Sentiment'),
                    use_container_width=True
                )
                

    else:
        st.warning("⚠️ No data found or subreddit is private/restricted.")
else:
    # Show some examples if no subreddit is entered
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #666;">
        <h3>👆 Enter a subreddit name above to get started</h3>
        <p>Try popular subreddits like: 'worldnews', 'technology', 'science', etc.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #eaeaea;">
    <p style="color: #666; font-size: 0.9rem;">
        Powered by CNN Deep Learning Model • Created with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)