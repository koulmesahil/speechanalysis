import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import numpy as np
from collections import defaultdict
import time
import hashlib
import requests

# Configure page
st.set_page_config(
    page_title="Podcast Speaker Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'speaker_profiles' not in st.session_state:
    st.session_state.speaker_profiles = {}

def generate_avatar_url(speaker_name):
    """Generate a consistent avatar URL for each speaker using DiceBear API"""
    # Create a consistent hash for the speaker name
    speaker_hash = hashlib.md5(speaker_name.encode()).hexdigest()[:8]
    
    # Use DiceBear API for free avatars
    avatar_styles = ['avataaars', 'big-smile', 'bottts', 'identicon', 'initials', 'personas']
    style = avatar_styles[len(speaker_name) % len(avatar_styles)]
    
    return f"https://api.dicebear.com/7.x/{style}/svg?seed={speaker_hash}&size=150"

def analyze_speaking_patterns(df):
    """Analyze detailed speaking patterns for each speaker"""
    patterns = {}
    
    for speaker in df['speaker'].unique():
        speaker_df = df[df['speaker'] == speaker]
        
        # Calculate gaps between speaking segments
        speaking_gaps = []
        sorted_segments = speaker_df.sort_values('start')
        
        for i in range(1, len(sorted_segments)):
            prev_end = sorted_segments.iloc[i-1]['end']
            curr_start = sorted_segments.iloc[i]['start']
            gap = curr_start - prev_end
            speaking_gaps.append(gap)
        
        # Analyze transcript content
        transcripts = speaker_df['transcript'].astype(str)
        word_counts = transcripts.apply(lambda x: len(x.split()))
        
        # Calculate speaking intensity (words per second)
        speaking_intensity = word_counts / speaker_df['duration']
        speaking_intensity = speaking_intensity.replace([np.inf, -np.inf], 0)  # Handle division by zero
        
        patterns[speaker] = {
            'avg_gap': np.mean(speaking_gaps) if speaking_gaps else 0,
            'consistency': np.std(speaker_df['duration']) if len(speaker_df) > 1 else 0,
            'speaking_intensity': speaking_intensity.mean(),
            'vocabulary_richness': len(set(' '.join(transcripts).lower().split())),
            'interruption_rate': len(speaking_gaps) / len(speaker_df) if len(speaker_df) > 0 else 0
        }
    
    return patterns

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
        transcripts = speaker_df['transcript'].astype(str)
        words_per_segment = transcripts.apply(lambda x: len(x.split())).mean()
        total_words = transcripts.apply(lambda x: len(x.split())).sum()
        
        # Advanced metrics
        speaking_rate = total_words / total_duration if total_duration > 0 else 0  # words per second
        longest_segment = speaker_df['duration'].max()
        shortest_segment = speaker_df['duration'].min()
        
        # Engagement metrics
        segment_variance = speaker_df['duration'].var()
        consistency_score = 1 / (1 + segment_variance) if segment_variance > 0 else 1
        
        speaker_stats[speaker] = {
            'total_segments': total_segments,
            'total_duration': total_duration,
            'avg_segment_length': avg_segment_length,
            'words_per_segment': words_per_segment,
            'total_words': total_words,
            'speaking_rate': speaking_rate,
            'speaking_percentage': (total_duration / df['duration'].sum()) * 100,
            'longest_segment': longest_segment,
            'shortest_segment': shortest_segment,
            'consistency_score': consistency_score,
            'engagement_level': 'High' if speaking_rate > 2.5 else 'Medium' if speaking_rate > 1.5 else 'Low'
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
        required_columns = ['speaker', 'start', 'end', 'duration', 'transcript']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Convert numeric columns
        df['start'] = pd.to_numeric(df['start'], errors='coerce')
        df['end'] = pd.to_numeric(df['end'], errors='coerce')
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        
        # Remove rows with NaN values in critical columns
        df = df.dropna(subset=['start', 'end', 'duration'])
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def main():
    st.title("Podcast Speaker Analytics")
    st.markdown("Upload your diarization CSV/JSON to analyze speaker patterns and engagement")
    
    # Sidebar for file upload
    with st.sidebar:



        st.header("📁 Upload Transcripts")
        uploaded_file = st.file_uploader(
            "Choose a CSV or JSON file",
            type=['csv', 'json'],
            help="Upload your speaker diarization transcripts data results"
        )
        use_sample = st.sidebar.button("🔄 Use Sample CSV File")

        
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            elif use_sample:
                    df = pd.read_csv("Stephen A-transcript.csv")  # Adjust path if it's in a subfolder
            if df is not None:
                st.session_state.data = df
                st.success(f"✅ Loaded {len(df)} segments")
                
                # Display basic info
                st.subheader("📊 Dataset Overview")
                st.metric("Total Segments", len(df))
                st.metric("Total Duration", f"{df['duration'].sum():.1f}s")
                st.metric("Speakers", df['speaker'].nunique())
                
                # Quick insights
                st.subheader("🔍 Quick Insights")
                most_active = df.groupby('speaker')['duration'].sum().idxmax()
                st.write(f"**Most Active:** {most_active}")
                
                avg_segment = df['duration'].mean()
                st.write(f"**Avg Segment:** {avg_segment:.1f}s")


        st.header("About")
        
        st.markdown(
            """
            This web app takes speaker-diarization outputs (CSV or JSON) and turns them into **interactive dashboards**.
            """
        )        
        
        st.markdown(
            """
            **Key Features**  
            - **Speaker Turn Timelines**: Visualize exactly when each speaker talks  
            - **Interactive Charts**: Drill into metrics like speaking duration, word counts, and sentiment  
            """
        )
        
        st.markdown(
            """
            Simply upload your transcripts file and get a data-driven view of any multi-speaker audio—sports commentary or beyond.
            For example, we will use the same example csv file as generated in the ["Speech-to-Chat" Hugging Face Space Made by kobakhit](https://huggingface.co/spaces/kobakhit/speech-to-chat)

            """
        )
        st.markdown("### 📂 Download Sample CSV")

        # Get file content from GitHub
        url = "https://raw.githubusercontent.com/koulmesahil/speechanalysis/refs/heads/main/Stephen%20A-transcript.csv"
        response = requests.get(url)
        sample_csv = response.content
        
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=sample_csv,
            file_name="sample.csv",
            mime="text/csv"
        )
    
    # Main content
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Tabs for different features
        tab1, tab2 = st.tabs(["👥 Speaker Portraits", "📈 Advanced Analytics"])
        
        with tab1:
            st.header("👥 Speaker Portrait Gallery")
            
            # Calculate speaker stats
            speaker_stats = calculate_speaker_stats(df)
            speaking_patterns = analyze_speaking_patterns(df)
            
            # Display all speakers in a grid
            speakers = list(speaker_stats.keys())
            
            if speakers:
                # Create columns for speaker cards (3 per row)
                num_cols = 3
                for i in range(0, len(speakers), num_cols):
                    cols = st.columns(num_cols)
                    
                    for j, col in enumerate(cols):
                        if i + j < len(speakers):
                            speaker = speakers[i + j]
                            stats = speaker_stats[speaker]
                            patterns = speaking_patterns[speaker]
                            
                            with col:
                                # Speaker card container
                                with st.container():
                                    st.markdown(f"""
                                    <div style="
                                        border: 2px solid #e0e0e0;
                                        border-radius: 15px;
                                        padding: 20px;
                                        margin: 10px 0;
                                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                        text-align: center;
                                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                    ">
                                        <img src="{generate_avatar_url(speaker)}" 
                                             style="width: 120px; height: 120px; border-radius: 50%; margin-bottom: 15px;">
                                        <h3 style="margin: 10px 0; color: #2c3e50;">{speaker}</h3>
                                        <p style="color: #7f8c8d; font-style: italic;">Podcast Commentator</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Stats in expandable section
                                with st.expander("Detailed Stats", expanded=True):
                                    col_a, col_b = st.columns(2)
                                    
                                    with col_a:
                                        st.metric("Segments", stats['total_segments'])
                                        st.metric("Total Time", f"{stats['total_duration']:.1f}s")
                                        st.metric("Speaking Rate", f"{stats['speaking_rate']:.1f} w/s")
                                    
                                    with col_b:
                                        st.metric("Share", f"{stats['speaking_percentage']:.1f}%")
                                        st.metric("Avg Segment", f"{stats['avg_segment_length']:.1f}s")
                                        st.metric("Engagement", stats['engagement_level'])
                                    
                                    # Speaking style analysis
                                    st.markdown("**Speaking Style**")
                                    if stats['avg_segment_length'] > 45:
                                        st.success("🎪 Storyteller - Long detailed segments")
                                    elif stats['speaking_rate'] > 2.5:
                                        st.info("⚡ Fast Talker - High word density")
                                    elif stats['consistency_score'] > 0.8:
                                        st.info("🎯 Consistent - Even pacing")
                                    else:
                                        st.warning("💬 Conversational - Varied responses")
                                    
                                    # Vocabulary insight
                                    vocab_score = patterns['vocabulary_richness']
                                    if vocab_score > 200:
                                        st.success(f"📚 Rich Vocabulary ({vocab_score} unique words)")
                                    else:
                                        st.info(f"💭 Focused Language ({vocab_score} unique words)")
        
        with tab2:
            st.header("Advanced Analytics Dashboard")
            
            # Enhanced overview metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_duration = df['duration'].sum()
            total_words = df['transcript'].astype(str).apply(lambda x: len(x.split())).sum()
            
            with col1:
                st.metric("Total Duration", f"{total_duration:.0f}s")
            with col2:
                st.metric("Total Words", f"{total_words:,}")
            with col3:
                st.metric("Avg Speaking Rate", f"{total_words/total_duration:.1f} w/s")
            with col4:
                st.metric("Total Speakers", df['speaker'].nunique())
            with col5:
                st.metric("Engagement Score", f"{(total_words/total_duration)*10:.0f}/100")
            
            # Interactive Charts Section
            st.subheader("Interactive Charts")
            
            # Row 1: Speaking distribution and patterns
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced pie chart with hover data
                speaker_data = df.groupby('speaker').agg({
                    'duration': 'sum',
                    'transcript': lambda x: sum(len(str(text).split()) for text in x)
                }).reset_index()
                speaker_data.columns = ['Speaker', 'Duration', 'Words']
                
                fig = px.pie(
                    speaker_data,
                    values='Duration',
                    names='Speaker',
                    title="Speaking Time Distribution",
                    hover_data=['Words'],
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Speaking intensity heatmap
                speaker_stats = calculate_speaker_stats(df)
                intensity_data = []
                for speaker, stats in speaker_stats.items():
                    intensity_data.append({
                        'Speaker': speaker,
                        'Segments': stats['total_segments'],
                        'Avg Duration': stats['avg_segment_length'],
                        'Speaking Rate': stats['speaking_rate'],
                        'Consistency': stats['consistency_score']
                    })
                
                intensity_df = pd.DataFrame(intensity_data)
                fig = px.scatter(
                    intensity_df,
                    x='Avg Duration',
                    y='Speaking Rate',
                    size='Segments',
                    color='Consistency',
                    hover_name='Speaker',
                    title="Speaker Intensity Analysis",
                    labels={'Avg Duration': 'Avg Segment Duration (s)', 'Speaking Rate': 'Words per Second'},
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Row 2: Temporal analysis
            #st.subheader("⏱️ Temporal Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Speaking frequency over time
                df['time_bucket'] = (df['start'] // 60).astype(int)  # 1-minute buckets
                time_analysis = df.groupby(['time_bucket', 'speaker']).size().reset_index(name='segments')
                
                fig = px.bar(
                    time_analysis,
                    x='time_bucket',
                    y='segments',
                    color='speaker',
                    title="Speaking Activity Over Time",
                    labels={'time_bucket': 'Time (minutes)', 'segments': 'Number of Segments'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Segment duration distribution by speaker
                fig = px.box(
                    df,
                    x='speaker',
                    y='duration',
                    title="Segment Duration Distribution",
                    labels={'duration': 'Duration (seconds)', 'speaker': 'Speaker'},
                    color='speaker'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Row 3: Advanced insights
            #st.subheader("Advanced Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Word density analysis
                df['word_count'] = df['transcript'].astype(str).apply(lambda x: len(x.split()))
                df['word_density'] = df['word_count'] / df['duration']
                df['word_density'] = df['word_density'].replace([np.inf, -np.inf], 0)  # Handle division by zero
                
                fig = px.violin(
                    df,
                    x='speaker',
                    y='word_density',
                    title="🎵 Word Density Distribution",
                    labels={'word_density': 'Words per Second', 'speaker': 'Speaker'},
                    color='speaker'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Conversation flow analysis
                df_sorted = df.sort_values('start')
                df_sorted['speaker_change'] = df_sorted['speaker'] != df_sorted['speaker'].shift(1)
                changes_per_minute = df_sorted.groupby(df_sorted['start'] // 60)['speaker_change'].sum().reset_index()
                changes_per_minute.columns = ['minute', 'speaker_changes']
                
                fig = px.line(
                    changes_per_minute,
                    x='minute',
                    y='speaker_changes',
                    title="",
                    labels={'minute': 'Time (minutes)', 'speaker_changes': 'Speaker Changes'},
                    markers=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Timeline visualization
            st.subheader("Timeline")
            
            # Create timeline chart
            fig = px.timeline(
                df,
                x_start='start',
                x_end='end',
                y='speaker',
                color='speaker',
                title="🕐 Speaking Segments Timeline",
                labels={'start': 'Start Time (s)', 'end': 'End Time (s)'},
                hover_data=['duration', 'transcript']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary insights
            st.subheader("💡 Key Insights")
            
            # Calculate insights
            most_active = df.groupby('speaker')['duration'].sum().idxmax()
            most_words = df.groupby('speaker')['word_count'].sum().idxmax()
            most_consistent = max(speaker_stats.keys(), key=lambda x: speaker_stats[x]['consistency_score'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"🏆 **Most Active Speaker**: {most_active}")
                #st.write(f"Dominated {speaker_stats[most_active]['speaking_percentage']:.1f}% of the conversation")
            
            with col2:
                st.info(f"💬 **Most Talkative**: {most_words}")
                total_speaker_words = df[df['speaker'] == most_words]['word_count'].sum()
                #st.write(f"Contributed {total_speaker_words:,} words to the discussion")
            
            with col3:
                st.info(f"🎯 **Most Consistent**: {most_consistent}")
                #st.write(f"Maintained steady pacing throughout the conversation")
            
            # Export functionality
            st.subheader("📥 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export Speaker Stats"):
                    speaker_stats_df = pd.DataFrame(speaker_stats).T
                    csv = speaker_stats_df.to_csv()
                    st.download_button(
                        label="Download Speaker Statistics CSV",
                        data=csv,
                        file_name="speaker_statistics.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 Export Analytics Data"):
                    analytics_df = df.copy()
                    analytics_df['word_count'] = analytics_df['transcript'].astype(str).apply(lambda x: len(x.split()))
                    analytics_df['word_density'] = analytics_df['word_count'] / analytics_df['duration']
                    analytics_df['word_density'] = analytics_df['word_density'].replace([np.inf, -np.inf], 0)
                    csv = analytics_df.to_csv(index=False)
                    st.download_button(
                        label="Download Enhanced Analytics CSV",
                        data=csv,
                        file_name="enhanced_analytics.csv",
                        mime="text/csv"
                    )
    
    else:
        # Welcome screen

        
        # Show expected format
        st.subheader("📋 Expected File Format")
        sample_data = pd.DataFrame({
            'speaker': ['Host', 'Guest', 'Host'],
            'start': [0.0, 45.2, 120.5],
            'end': [45.2, 120.5, 180.0],
            'duration': [45.2, 75.3, 59.5],
            'transcript': ['Welcome to our show...', 'Thanks for having me...', 'Let me ask you about...']
        })
        st.dataframe(sample_data)
        
        st.markdown("""
        **Required columns:**
        - `speaker`: Speaker name/identifier
        - `start`: Start time in seconds
        - `end`: End time in seconds  
        - `duration`: Segment duration in seconds
        - `transcript`: Text content of the segment
        """)

if __name__ == "__main__":
    main()
