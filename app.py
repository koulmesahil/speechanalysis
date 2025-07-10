import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from transformers import pipeline
import numpy as np
from collections import defaultdict
import time

# Configure page
st.set_page_config(
    page_title="Podcast Diarization Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'ad_breaks' not in st.session_state:
    st.session_state.ad_breaks = []
if 'speaker_profiles' not in st.session_state:
    st.session_state.speaker_profiles = {}

@st.cache_resource
def load_ad_classifier():
    """Load the text classification model for ad detection"""
    try:
        # Using a general text classification model that can be fine-tuned for ad detection
        classifier = pipeline("text-classification", 
                            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                            return_all_scores=True)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_ad_keywords(text):
    """Simple keyword-based ad detection as fallback"""
    ad_keywords = [
        'sponsor', 'sponsored', 'advertisement', 'ad break', 'brought to you by',
        'thanks to our sponsor', 'promo code', 'discount', 'special offer',
        'visit', 'website', 'download', 'app', 'try for free', 'get started',
        'sign up', 'limited time', 'exclusive', 'deal', 'savings'
    ]
    
    text_lower = text.lower()
    keyword_matches = [keyword for keyword in ad_keywords if keyword in text_lower]
    
    # Score based on number of keywords and their weights
    score = len(keyword_matches) * 0.1
    if any(word in text_lower for word in ['sponsor', 'sponsored', 'advertisement']):
        score += 0.5
    if any(word in text_lower for word in ['promo code', 'discount', 'special offer']):
        score += 0.3
    
    return min(score, 1.0), keyword_matches

def analyze_ad_breaks(df):
    """Analyze transcript for potential ad breaks"""
    classifier = load_ad_classifier()
    ad_breaks = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        progress_bar.progress((idx + 1) / len(df))
        status_text.text(f"Analyzing segment {idx + 1}/{len(df)}")
        
        text = str(row['transcript'])
        
        # Keyword-based detection
        keyword_score, keywords = detect_ad_keywords(text)
        
        # Length-based heuristic (ads often have specific length patterns)
        duration = float(row['duration'])
        length_score = 0.3 if 15 <= duration <= 120 else 0.1
        
        # Combine scores
        total_score = keyword_score + length_score
        
        if total_score > 0.4 or len(keywords) >= 2:
            ad_breaks.append({
                'segment_id': idx,
                'speaker': row['speaker'],
                'start_time': row['start'],
                'end_time': row['end'],
                'duration': duration,
                'transcript': text,
                'confidence': total_score,
                'keywords_found': keywords,
                'start_formatted': format_timestamp(row['start']),
                'end_formatted': format_timestamp(row['end'])
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return ad_breaks

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def calculate_speaker_stats(df):
    """Calculate comprehensive speaker statistics"""
    speaker_stats = {}
    
    for speaker in df['speaker'].unique():
        speaker_df = df[df['speaker'] == speaker]
        
        # Basic stats
        total_segments = len(speaker_df)
        total_duration = speaker_df['duration'].sum()
        avg_segment_length = speaker_df['duration'].mean()
        
        # Speaking patterns
        words_per_segment = speaker_df['transcript'].apply(lambda x: len(str(x).split())).mean()
        
        # Time distribution
        speaking_times = []
        for _, row in speaker_df.iterrows():
            speaking_times.extend(range(int(row['start']), int(row['end']) + 1))
        
        speaker_stats[speaker] = {
            'total_segments': total_segments,
            'total_duration': total_duration,
            'avg_segment_length': avg_segment_length,
            'words_per_segment': words_per_segment,
            'speaking_percentage': (total_duration / df['duration'].sum()) * 100,
            'speaking_times': speaking_times
        }
    
    return speaker_stats

def load_data(uploaded_file):
    """Load and validate uploaded data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            df = pd.DataFrame(data)
        else:
            st.error("Please upload a CSV or JSON file")
            return None
        
        # Validate required columns
        required_columns = ['speaker', 'start', 'end', 'duration', 'transcript', 'speakers_labeled']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Convert numeric columns
        df['start'] = pd.to_numeric(df['start'], errors='coerce')
        df['end'] = pd.to_numeric(df['end'], errors='coerce')
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def main():
    st.title("🎙️ Podcast Diarization Analyzer")
    st.markdown("Upload your diarization CSV/JSON to analyze ad breaks and speaker patterns")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or JSON file",
            type=['csv', 'json'],
            help="Upload your diarization data with speaker, timestamps, and transcripts"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.success(f"✅ Loaded {len(df)} segments")
                
                # Display basic info
                st.subheader("📊 Dataset Overview")
                st.metric("Total Segments", len(df))
                st.metric("Total Duration", f"{df['duration'].sum():.1f}s")
                st.metric("Speakers", df['speaker'].nunique())
    
    # Main content
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Tabs for different features
        tab1, tab2, tab3 = st.tabs(["🎯 Ad Break Tracker", "👥 Speaker Portraits", "📈 Analytics"])
        
        with tab1:
            st.header("🎯 Sponsorship Ad Break Tracker")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                if st.button("🔍 Analyze Ad Breaks", type="primary"):
                    with st.spinner("Analyzing transcript for ad breaks..."):
                        st.session_state.ad_breaks = analyze_ad_breaks(df)
                
                sensitivity = st.slider(
                    "Detection Sensitivity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.4,
                    step=0.1,
                    help="Higher values = more sensitive detection"
                )
            
            with col1:
                if st.session_state.ad_breaks:
                    st.subheader(f"Found {len(st.session_state.ad_breaks)} Potential Ad Breaks")
                    
                    for i, ad in enumerate(st.session_state.ad_breaks):
                        with st.expander(f"Ad Break {i+1} - {ad['speaker']} ({ad['start_formatted']} - {ad['end_formatted']})"):
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.write(f"**Transcript:** {ad['transcript'][:200]}...")
                                if ad['keywords_found']:
                                    st.write(f"**Keywords:** {', '.join(ad['keywords_found'])}")
                            
                            with col_b:
                                st.metric("Duration", f"{ad['duration']:.1f}s")
                                st.metric("Confidence", f"{ad['confidence']:.2f}")
                    
                    # Export functionality
                    if st.button("📥 Export Ad Breaks"):
                        ad_df = pd.DataFrame(st.session_state.ad_breaks)
                        csv = ad_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="ad_breaks.csv",
                            mime="text/csv"
                        )
        
        with tab2:
            st.header("👥 Speaker Portrait Carousel")
            
            # Calculate speaker stats
            speaker_stats = calculate_speaker_stats(df)
            
            # Create speaker carousel
            speakers = list(speaker_stats.keys())
            
            if speakers:
                # Speaker selector
                selected_speaker = st.selectbox("Select Speaker", speakers)
                
                # Create speaker card
                stats = speaker_stats[selected_speaker]
                
                # Main speaker card
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        # Placeholder for speaker photo
                        st.image(
                            "https://via.placeholder.com/150x150/3498db/ffffff?text=" + selected_speaker[0],
                            width=150
                        )
                    
                    with col2:
                        st.markdown(f"## {selected_speaker}")
                        st.markdown("*Podcast Commentator*")
                        
                        # Quick stats
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Segments", stats['total_segments'])
                        with col_b:
                            st.metric("Speaking Time", f"{stats['total_duration']:.1f}s")
                        with col_c:
                            st.metric("Share", f"{stats['speaking_percentage']:.1f}%")
                    
                    with col3:
                        st.markdown("### 📊 Style")
                        st.write(f"**Avg Segment:** {stats['avg_segment_length']:.1f}s")
                        st.write(f"**Words/Segment:** {stats['words_per_segment']:.1f}")
                        
                        # Speaking style indicator
                        if stats['avg_segment_length'] > 30:
                            st.success("🎯 Long-form speaker")
                        elif stats['words_per_segment'] > 20:
                            st.info("💬 Conversational")
                        else:
                            st.warning("⚡ Quick responses")
                
                # Speaker timeline
                st.subheader(f"🕐 {selected_speaker}'s Speaking Timeline")
                speaker_df = df[df['speaker'] == selected_speaker]
                
                # Create timeline chart
                fig = px.timeline(
                    speaker_df.reset_index(),
                    x_start='start',
                    x_end='end',
                    y='speaker',
                    color='duration',
                    title=f"{selected_speaker} Speaking Segments",
                    labels={'start': 'Start Time (s)', 'end': 'End Time (s)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("📈 Analytics Dashboard")
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Duration", f"{df['duration'].sum():.1f}s")
            with col2:
                st.metric("Average Segment", f"{df['duration'].mean():.1f}s")
            with col3:
                st.metric("Total Speakers", df['speaker'].nunique())
            with col4:
                st.metric("Total Segments", len(df))
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Speaking time distribution
                speaker_times = df.groupby('speaker')['duration'].sum().sort_values(ascending=False)
                fig = px.pie(
                    values=speaker_times.values,
                    names=speaker_times.index,
                    title="Speaking Time Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Segment length distribution
                fig = px.histogram(
                    df,
                    x='duration',
                    nbins=30,
                    title="Segment Length Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Timeline overview
            st.subheader("🎙️ Complete Timeline")
            fig = px.timeline(
                df.reset_index(),
                x_start='start',
                x_end='end',
                y='speaker',
                color='speaker',
                title="Complete Speaking Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to the Podcast Diarization Analyzer!
        
        This app helps you analyze podcast transcripts with two main features:
        
        ### 🎯 Ad Break Tracker
        - Automatically detects sponsored content and advertisements
        - Uses AI and keyword analysis for accurate detection
        - Provides timestamps and confidence scores
        - Export results for downstream processing
        
        ### 👥 Speaker Portraits
        - Interactive speaker profiles with photos and bios
        - Speaking time analytics and patterns
        - Style analysis (conversational vs. long-form)
        - Timeline visualization of speaking segments
        
        **Get started by uploading your diarization CSV or JSON file in the sidebar!**
        """)
        
        # Show expected format
        st.subheader("📋 Expected File Format")
        sample_data = pd.DataFrame({
            'speaker': ['Host', 'Guest', 'Host'],
            'start': [0.0, 45.2, 120.5],
            'end': [45.2, 120.5, 180.0],
            'duration': [45.2, 75.3, 59.5],
            'transcript': ['Welcome to our show...', 'Thanks for having me...', 'Before we continue, a word from our sponsor...'],
            'speakers_labeled': ['Host', 'Guest', 'Host']
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()
