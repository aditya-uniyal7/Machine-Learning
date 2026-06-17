import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from googleapiclient.discovery import build
import os
import re

# Page Config
st.set_page_config(page_title="YouTube Prediction Dashboard", layout="wide")

# Load Brain
YOUTUBE_API_KEY = "enter your api ketyyeyyyyeyyeyyyyeyyeyyeeyey ;<" 
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models/viral.pkl')
model = joblib.load(model_path) # Path resolved dynamically

# Helpers
def get_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def get_video_details(youtube_client, video_id):
    try:
        request = youtube_client.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()
        
        if not response.get('items'):
            return None
            
        item = response['items'][0]
        snippet = item['snippet']
        stats = item['statistics']
        
        details = {
            'title': snippet.get('title', 'Unknown Title'),
            'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', snippet.get('thumbnails', {}).get('default', {}).get('url', '')),
            'views': int(stats.get('viewCount', 0)),
            'likes': int(stats.get('likeCount', 0)),
            'comments': int(stats.get('commentCount', 0)),
            'tags': snippet.get('tags', []),
            'published_at': snippet.get('publishedAt', ''),
            'category_id': int(snippet.get('categoryId', 0))
        }
        return details
    except Exception:
        return None

def get_mock_video_details(url):
    video_id = get_video_id(url) or "dQw4w9WgXcQ"
    return {
        'title': "How to Train Your AI Model in 24 Hours! (Deep Learning Tutorial)",
        'thumbnail': f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        'views': 45672,
        'likes': 3820,
        'comments': 284,
        'tags': ['machine learning', 'ai', 'neural networks', 'python', 'tutorial'],
        'published_at': '2026-06-15T12:00:00Z',
        'category_id': 27 # Education
    }

st.title("🚀 YouTube Video Prediction Dashboard")

# UI Tabs
tab1, tab2 = st.tabs(["Live Prediction", "Channel Insights"])

with tab1:
    url = st.text_input("Paste YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    if st.button("Analyze Video"):
        if not url.strip():
            st.warning("Please paste a valid YouTube URL first!")
        else:
            with st.spinner("Fetching YouTube details and predicting..."):
                video_id = get_video_id(url)
                details = None
                is_mock = False
                
                if video_id:
                    details = get_video_details(youtube, video_id)
                
                if not details:
                    # Fallback to mock data
                    details = get_mock_video_details(url)
                    is_mock = True
                
                # Extract features for the model
                title_length = len(details['title'])
                tag_count = len(details['tags'])
                try:
                    upload_hour = pd.to_datetime(details['published_at']).hour
                except Exception:
                    upload_hour = 12
                category_id = details['category_id']
                
                # Make prediction
                data = pd.DataFrame([[title_length, tag_count, upload_hour, category_id]], 
                                    columns=['title_length', 'tag_count', 'upload_hour', 'category_id'])
                pred = model.predict(data)
                prob = model.predict_proba(data)[0][1]
                
                # Layout
                col_left, col_right = st.columns([1.2, 1.0])
                
                with col_left:
                    st.markdown(f"### 📺 {details['title']}")
                    if details['thumbnail']:
                        st.image(details['thumbnail'], use_container_width=True)
                    
                    st.markdown("#### 📊 Video Statistics")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Views", f"{details['views']:,}")
                    m_col2.metric("Likes", f"{details['likes']:,}")
                    m_col3.metric("Comments", f"{details['comments']:,}")
                    
                    if is_mock:
                        st.info("ℹ️ Running in Sandbox/Fallback Mode (Mock Statistics Shown).")
                        
                with col_right:
                    st.markdown("### 🎯 Prediction Analysis")
                    # Plotly Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        title = {'text': "Viral Potential (%)"},
                        gauge = {
                            'axis': {'range': [0, 100]}, 
                            'bar': {'color': "#FF0000"},
                            'steps': [
                                {'range': [0, 40], 'color': "#ECEFF1"},
                                {'range': [40, 70], 'color': "#FFE082"},
                                {'range': [70, 100], 'color': "#A5D6A7"}
                            ]
                        },
                    ))
                    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if pred[0] == 1:
                        st.success("🎉 **Verdict: Viral Potential HIGH! 🔥** \n\nThis video has all the right ingredients to trend!")
                    else:
                        st.warning("🐢 **Verdict: Performance likely AVERAGE.** \n\nThis video might perform moderately. Consider optimizing title length, tags, or upload hour.")

with tab2:
    st.subheader("Top 3 Viral Videos")
    # Yahan YouTube API se channel ki top videos fetch karne ka code aayega
    st.write("Fetching top performing videos...")
    # Example placeholders
    col1, col2, col3 = st.columns(3)
    col1.metric("Video 1", "GTA 5 Heist", "1.2M Views")
    col2.metric("Video 2", "Funny Moments", "800K Views")
    col3.metric("Video 3", "Montage #1", "500K Views")
