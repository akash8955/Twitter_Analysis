import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Initialize Nitter scraper
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# Function to create a colored card
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# Main app logic
def main():
    st.title("Social Media Sentiment Analysis")
    st.markdown("Analyze sentiment from text or Twitter user's tweets")

    # Load stopwords, model, vectorizer, and scraper only once
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()
    


    # User input: either text input or Twitter username
    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")

    elif option == "Get tweets from user":
        # Add option for demo mode
        use_demo = st.checkbox("Use Demo Mode (sample tweets)", help="Enable this to see how the analysis works with sample tweets")
        
        if use_demo:
            st.info("📝 Demo mode enabled - using sample tweets for demonstration")
            username = st.text_input("Enter any name for demo", value="demo_user")
            num_tweets = st.slider("Number of tweets to analyze", min_value=5, max_value=20, value=10, step=5)
            
            if st.button("Analyze Demo Tweets"):
                # Sample tweets for demonstration
                demo_tweets = [
                    "I absolutely love this product! Best purchase I've made this year. Highly recommend it to everyone!",
                    "This is terrible. Worst experience ever. I want my money back immediately.",
                    "Just had an amazing day at the beach with friends. Life is beautiful!",
                    "Feeling so frustrated and disappointed with the customer service. Never shopping here again.",
                    "The new update is fantastic! Everything works so smoothly now. Great job developers!",
                    "This app keeps crashing. So annoying and unreliable. Waste of time.",
                    "What a wonderful surprise! Exceeded all my expectations. Five stars!",
                    "Completely useless and overpriced. Don't waste your money on this garbage.",
                    "Having such a great time at the concert! The energy is incredible!",
                    "Stuck in traffic for hours. This is the worst commute of my life.",
                    "Just finished reading an inspiring book. Feeling motivated and energized!",
                    "The food was cold and tasteless. Very disappointing dining experience.",
                    "Congratulations to the team! Outstanding performance and well-deserved victory!",
                    "Another delay and no explanation. This airline is a complete disaster.",
                    "Beautiful sunset today. Nature never fails to amaze me.",
                    "Broken promises and poor quality. I expected so much better than this.",
                    "Thank you for the wonderful gift! You always know how to make me smile.",
                    "Rude staff and dirty facilities. Will never return to this place.",
                    "Excited about the new opportunities ahead! Can't wait to get started!",
                    "System is down again. How is this acceptable? Completely unreliable service."
                ]
                
                # Use only the requested number of tweets
                selected_tweets = demo_tweets[:num_tweets]
                
                with st.spinner(f'Analyzing {len(selected_tweets)} demo tweets...'):
                    # Analyze sentiment for all tweets
                    sentiments = []
                    for tweet_text in selected_tweets:
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                        sentiments.append(sentiment)
                    
                    # Calculate sentiment statistics
                    positive_count = sentiments.count("Positive")
                    negative_count = sentiments.count("Negative")
                    total_count = len(sentiments)
                    positive_percentage = (positive_count / total_count) * 100
                    negative_percentage = (negative_count / total_count) * 100
                    
                    # Display sentiment statistics
                    st.subheader("📊 Sentiment Analysis Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tweets", total_count)
                    with col2:
                        st.metric("Positive", f"{positive_count} ({positive_percentage:.1f}%)")
                    with col3:
                        st.metric("Negative", f"{negative_count} ({negative_percentage:.1f}%)")
                    
                    # Display sentiment distribution chart
                    st.subheader("📈 Sentiment Distribution")
                    chart_data = {"Sentiment": ["Positive", "Negative"], "Count": [positive_count, negative_count]}
                    st.bar_chart(chart_data, x="Sentiment", y="Count", color="Sentiment")
                    
                    # Display individual tweets
                    st.subheader("Individual Tweet Analysis")
                    for idx, (tweet_text, sentiment) in enumerate(zip(selected_tweets, sentiments), 1):
                        # Create and display the colored card for the tweet with numbering
                        card_html = f"""
                        <div style="background-color: {'#28a745' if sentiment == 'Positive' else '#dc3545'}; 
                                    padding: 15px; 
                                    border-radius: 8px; 
                                    margin: 10px 0;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h5 style="color: white; margin: 0 0 10px 0;">Tweet #{idx} - {sentiment} Sentiment</h5>
                            <p style="color: white; margin: 0; line-height: 1.5;">{tweet_text}</p>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
        
        else:
            username = st.text_input("Enter Twitter username (with or without @)")
            
            # Add slider to configure number of tweets to fetch
            num_tweets = st.slider("Number of tweets to fetch", min_value=10, max_value=100, value=50, step=10)
            
            if st.button("Fetch Tweets"):
                # Validate input
                if not username:
                    st.error("Please enter a Twitter username")
                else:
                    # Remove @ symbol if present
                    username = username.lstrip('@')
                    
                    # Show loading spinner while fetching tweets
                    with st.spinner(f'Fetching up to {num_tweets} tweets from @{username}...'):
                        try:
                            tweets_data = scraper.get_tweets(username, mode='user', number=num_tweets)
                            
                            if tweets_data and 'tweets' in tweets_data and len(tweets_data['tweets']) > 0:
                                tweets = tweets_data['tweets']
                                st.success(f"Successfully fetched {len(tweets)} tweets from @{username}")
                                
                                # Analyze sentiment for all tweets
                                sentiments = []
                                for tweet in tweets:
                                    tweet_text = tweet['text']
                                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                                    sentiments.append(sentiment)
                                
                                # Calculate sentiment statistics
                                positive_count = sentiments.count("Positive")
                                negative_count = sentiments.count("Negative")
                                total_count = len(sentiments)
                                positive_percentage = (positive_count / total_count) * 100
                                negative_percentage = (negative_count / total_count) * 100
                                
                                # Display sentiment statistics
                                st.subheader("📊 Sentiment Analysis Summary")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Tweets", total_count)
                                with col2:
                                    st.metric("Positive", f"{positive_count} ({positive_percentage:.1f}%)")
                                with col3:
                                    st.metric("Negative", f"{negative_count} ({negative_percentage:.1f}%)")
                                
                                # Display sentiment distribution chart
                                st.subheader("📈 Sentiment Distribution")
                                chart_data = {"Sentiment": ["Positive", "Negative"], "Count": [positive_count, negative_count]}
                                st.bar_chart(chart_data, x="Sentiment", y="Count", color="Sentiment")
                                
                                # Display individual tweets
                                st.subheader("🐦 Individual Tweet Analysis")
                                for idx, (tweet, sentiment) in enumerate(zip(tweets, sentiments), 1):
                                    tweet_text = tweet['text']
                                    
                                    # Create and display the colored card for the tweet with numbering
                                    card_html = f"""
                                    <div style="background-color: {'#28a745' if sentiment == 'Positive' else '#dc3545'}; 
                                                padding: 15px; 
                                                border-radius: 8px; 
                                                margin: 10px 0;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <h5 style="color: white; margin: 0 0 10px 0;">Tweet #{idx} - {sentiment} Sentiment</h5>
                                        <p style="color: white; margin: 0; line-height: 1.5;">{tweet_text}</p>
                                    </div>
                                    """
                                    st.markdown(card_html, unsafe_allow_html=True)
                            
                            else:
                                st.warning(f"No tweets found for @{username}. Please check if the username is correct and the account has public tweets.")
                                st.info("💡 **Tip**: Try enabling **Demo Mode** above to see how the sentiment analysis works with sample tweets!")
                        
                        except Exception as e:
                            st.error(f"An error occurred while fetching tweets: {str(e)}")
                            st.info("""
                            **This could be due to:**
                            - Invalid username
                            - Network connectivity issues
                            - Twitter/Nitter service unavailability (Nitter instances are often unreliable)
                            
                            **💡 Suggestion**: Enable **Demo Mode** (checkbox above) to see how the sentiment analysis works with sample tweets!
                            """)

if __name__ == "__main__":
    main()