import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import re

# Page configuration
st.set_page_config(
    page_title="🎬 Emotion-Based Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for better styling
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    
    /* Apply font globally */
    html, body, [class*="block-container"], [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Hero Section with Background Image */
    .hero-section {
        position: relative;
        background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 8rem 2rem 8rem 2rem;
        margin: -3rem -3rem 3rem -3rem;
        border-radius: 0 0 40px 40px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.35);
        z-index: 1;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
    }
    
    /* Global Styles */
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Header - ABSOLUTELY MASSIVE! */
    .main-header, h1.main-header {
        font-size: 5rem !important;
        font-weight: 900 !important;
        text-align: center !important;
        color: white !important;
        text-shadow: 
            5px 5px 15px rgba(0,0,0,1),
            -2px -2px 10px rgba(0,0,0,0.9),
            0 0 60px rgba(0,0,0,0.8),
            0 0 80px rgba(102,126,234,0.6),
            0 0 100px rgba(245,87,108,0.4) !important;
        margin-bottom: 0.5rem !important;
        margin-top: 0 !important;
        padding: 2rem 0 1rem 0 !important;
        letter-spacing: -3px !important;
        line-height: 1.05 !important;
    }
    
    @media (max-width: 768px) {
        .main-header, h1.main-header {
            font-size: 5rem !important;
        }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sub Header */
    .sub-header {
        text-align: center;
        color: white;
        text-shadow: 
            3px 3px 10px rgba(0,0,0,1),
            -1px -1px 5px rgba(0,0,0,0.9),
            0 0 30px rgba(0,0,0,0.7);
        margin-bottom: 1rem;
        font-size: 2.2rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Decorative Line */
    .decorative-line {
        width: 200px;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #f5576c, #ffa69e);
        margin: 1rem auto 1.5rem auto;
        border-radius: 3px;
        box-shadow: 0 0 20px rgba(245,87,108,0.6);
    }
    
    /* Movie Card Styling */
    .movie-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 2px solid #667eea30;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    /* Movie Title */
    .movie-title {
        font-size: 1.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #667eea 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Emotion Tags */
    .emotion-tag {
        display: inline-block;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        transition: all 0.2s ease;
    }
    
    .emotion-tag:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Match Score */
    .match-score {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 1rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea40;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea08 0%, #764ba208 100%);
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #f5576c);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #667eea15;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border: 2px solid #667eea30;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Constants - 7 EMOTIONS
EMOS = ["joy", "sadness", "anger", "fear", "surprise", "neutral", "disgust"]
EMOTION_COLORS = {
    "joy": "#FFD700",
    "sadness": "#4169E1", 
    "anger": "#DC143C",
    "fear": "#8B008B",
    "surprise": "#FF69B4",
    "neutral": "#808080",
    "disgust": "#228B22"
}
EMOTION_EMOJIS = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐",
    "disgust": "🤢"
}

# Context mode presets
CONTEXT_MODES = {
    "Custom": None,
    "Date Night 💕": {"joy": 0.5, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.3, "neutral": 0.2, "disgust": 0.0},
    "Comfort Watch 🛋️": {"joy": 0.6, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.2, "neutral": 0.2, "disgust": 0.0},
    "Adrenaline Rush ⚡": {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.5, "surprise": 0.5, "neutral": 0.0, "disgust": 0.0},
    "Emotional Journey 🎭": {"joy": 0.2, "sadness": 0.4, "anger": 0.1, "fear": 0.1, "surprise": 0.2, "neutral": 0.0, "disgust": 0.0},
    "Comedy Vibes 😂": {"joy": 0.7, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.3, "neutral": 0.0, "disgust": 0.0},
    "Thriller Mode 🔪": {"joy": 0.0, "sadness": 0.1, "anger": 0.2, "fear": 0.5, "surprise": 0.2, "neutral": 0.0, "disgust": 0.0},
    "Dark Comedy 😈": {"joy": 0.4, "sadness": 0.2, "anger": 0.1, "fear": 0.05, "surprise": 0.15, "neutral": 0.05, "disgust": 0.05},
    "Feel-Good 🌈": {"joy": 0.7, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.2, "neutral": 0.1, "disgust": 0.0},
    "Deep Drama 🎬": {"joy": 0.1, "sadness": 0.5, "anger": 0.2, "fear": 0.1, "surprise": 0.05, "neutral": 0.05, "disgust": 0.0},
    "Existential Crisis 🤯": {"joy": 0.0, "sadness": 0.3, "anger": 0.1, "fear": 0.3, "surprise": 0.2, "neutral": 0.1, "disgust": 0.0},
    "Chaos Mode 🌪️": {"joy": 0.2, "sadness": 0.2, "anger": 0.2, "fear": 0.2, "surprise": 0.2, "neutral": 0.0, "disgust": 0.0}
}

# Helper functions
def clean_movie_name(name):
    """Clean movie name - remove IMDb suffixes and normalize"""
    name = str(name)
    # Remove IMDb suffixes
    name = re.sub(r'\s*-\s*User reviews\s*-\s*IMDb', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*IMDb', '', name, flags=re.IGNORECASE)
    # Remove year in parentheses
    name = re.sub(r'\s*\(\d{4}\)', '', name)
    # Clean whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name.upper()

def normalize_emotion_vector(vec):
    """L2 normalize an emotion vector"""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def fuse_vector(src_vecs, alpha=0.3, beta=0.5, gamma=0.2):
    """Fuse emotion vectors from different sources with weights"""
    w = {"critic": alpha, "user": beta, "yt": gamma}
    avail = {k: v for k, v in src_vecs.items() if k in w and not k.startswith('meta_')}
    if not avail:
        return None
    wsum = sum(w[k] for k in avail)
    fused = sum((w[k]/wsum) * avail[k] for k in avail)
    return normalize_emotion_vector(fused)

def make_query_vector(joy=0, sadness=0, anger=0, fear=0, surprise=0, neutral=0, disgust=0):
    """Create normalized query vector from emotion values"""
    q = np.array([joy, sadness, anger, fear, surprise, neutral, disgust], float)
    q_sum = q.sum()
    q = q / q_sum if q_sum > 0 else q / len(q)
    return normalize_emotion_vector(q)

def rank_movies(movie_sources, q, alpha=0.3, beta=0.5, gamma=0.2, top_k=10, min_rating=0, hidden_gems=False, only_complete=False, selected_platforms=None):
    """Rank movies based on emotion match with optional filters"""
    rows = []
    for mv, srcs in movie_sources.items():
        # Filter for complete sources if requested
        if only_complete:
            has_all_sources = all(k in srcs for k in ['user', 'critic', 'yt'])
            if not has_all_sources:
                continue
        
        # Filter by streaming platforms if specified
        if selected_platforms and len(selected_platforms) > 0:
            if 'streaming' not in srcs:
                continue  # Skip movies without streaming data
            
            # Check if movie is on any of the selected platforms
            has_platform = False
            for platform in selected_platforms:
                if srcs['streaming']['platforms'].get(platform, False):
                    has_platform = True
                    break
            
            if not has_platform:
                continue
        
        v = fuse_vector(srcs, alpha, beta, gamma)
        if v is None:
            continue
        
        score = float(cosine_similarity([q], [v])[0, 0])
        
        # Apply filters
        meta = srcs.get('meta_user', {})
        if min_rating > 0 and meta.get('avg_rating', 0) < min_rating:
            continue
        
        # Hidden gems: boost less popular movies
        if hidden_gems:
            popularity = meta.get('popularity', 0.5)
            score = 0.7 * score + 0.3 * (1 - popularity)
        
        rows.append((mv, score, v, srcs))
    
    return sorted(rows, key=lambda x: x[1], reverse=True)[:top_k]

def explain_match(q, v, movie_name, weights, srcs, emos=EMOS, topn=2):
    """Explain why a movie matches the query"""
    contrib = q * v
    idx = np.argsort(contrib)[::-1][:topn]
    
    # Get metadata from all sources
    meta_user = srcs.get('meta_user', {})
    meta_critic = srcs.get('meta_critic', {})
    meta_yt = srcs.get('meta_yt', {})
    
    return {
        "movie": movie_name,
        "match_pct": float(cosine_similarity([q], [v])[0, 0]) * 100,
        "top_emotions": [emos[i] for i in idx],
        "top_scores": [float(contrib[i]) for i in idx],
        "weights": weights,
        "emotion_breakdown": {emos[i]: float(v[i]) for i in range(len(emos))},
        "sentiment_user": meta_user.get('sentiment'),
        "sentiment_critic": meta_critic.get('sentiment'),
        "sentiment_yt": meta_yt.get('sentiment'),
        "rating": meta_user.get('avg_rating'),
        "dominant_user": meta_user.get('dominant'),
        "dominant_critic": meta_critic.get('dominant'),
        "dominant_yt": meta_yt.get('dominant'),
        "sources": list([k for k in srcs.keys() if k in ['user', 'critic', 'yt']])
    }

@st.cache_data
def load_data():
    """Load and process emotion vectors with streaming data"""
    try:
        # Load CSVs
        user_df = pd.read_csv("user_emotions_simple.csv")
        critic_df = pd.read_csv("critic_emotions_simple.csv")
        yt_df = pd.read_csv("youtube_emotions_simple.csv")
        
        # Try to load streaming data if available
        streaming_data = {}
        try:
            streaming_df = pd.read_csv("imdb_top250_streaming_full.csv")
            streaming_df['movie_normalized'] = streaming_df['movie'].str.upper()
            # Store streaming data by normalized movie name
            for idx, row in streaming_df.iterrows():
                streaming_data[row['movie_normalized']] = {
                    'platforms': {
                        'Netflix': row['Netflix'] == 'Yes',
                        'Amazon Prime': row['Amazon Prime'] == 'Yes',
                        'Hulu': row['Hulu'] == 'Yes',
                        'Disney+': row['Disney+'] == 'Yes',
                        'HBO Max': row['HBO Max'] == 'Yes',
                        'Apple TV+': row['Apple TV+'] == 'Yes',
                        'Paramount+': row['Paramount+'] == 'Yes'
                    },
                    'all_platforms': row['all_platforms'],
                    'total_platforms': row['total_platforms'],
                    'price_tier': row['price_tier'],
                    'year': row.get('year'),
                    'runtime_min': row.get('runtime_min')
                }
        except FileNotFoundError:
            st.warning("ℹ️ Streaming data not found. Guess you'll have to use *gasp* multiple apps to find where to watch.")
        
        # Clean movie names
        user_df['movie'] = user_df['movie_clean'].apply(clean_movie_name)
        critic_df['movie'] = critic_df['movie_clean'].apply(clean_movie_name)
        yt_df['movie'] = yt_df['movie_clean'].apply(clean_movie_name)
        
        # Build movie sources dictionary
        movie_sources = {}
        
        # Process user data
        for idx, row in user_df.iterrows():
            movie = row['movie']
            vec = row[EMOS].values.astype(float)
            vec = normalize_emotion_vector(vec)  # NORMALIZE HERE
            
            movie_sources.setdefault(movie, {})['user'] = vec
            movie_sources[movie]['meta_user'] = {
                'sentiment': row.get('sentiment_score'),
                'avg_rating': row.get('avg_user_rating'),
                'dominant': row.get('dominant_emotion'),
                'dominant_strength': row.get('dominant_strength')
            }
            
            # Add streaming data if available
            if movie in streaming_data:
                movie_sources[movie]['streaming'] = streaming_data[movie]
        
        # Process critic data
        for idx, row in critic_df.iterrows():
            movie = row['movie']
            vec = row[EMOS].values.astype(float)
            vec = normalize_emotion_vector(vec)  # NORMALIZE HERE
            
            movie_sources.setdefault(movie, {})['critic'] = vec
            movie_sources[movie]['meta_critic'] = {
                'sentiment': row.get('sentiment_score'),
                'dominant': row.get('dominant_emotion'),
                'dominant_strength': row.get('dominant_strength')
            }
        
        # Process YouTube data
        for idx, row in yt_df.iterrows():
            movie = row['movie']
            vec = row[EMOS].values.astype(float)
            vec = normalize_emotion_vector(vec)  # NORMALIZE HERE
            
            movie_sources.setdefault(movie, {})['yt'] = vec
            movie_sources[movie]['meta_yt'] = {
                'sentiment': row.get('sentiment_score'),
                'dominant': row.get('dominant_emotion'),
                'dominant_strength': row.get('dominant_strength')
            }
        
        return movie_sources, user_df, streaming_data
        
    except FileNotFoundError as e:
        st.error(f"⚠️ Data files not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        st.exception(e)
        return None, None, None

def create_emotion_radar(emotions_dict, title="Emotion Profile"):
    """Create a beautiful radar chart for emotion visualization"""
    fig = go.Figure()
    
    emotions = list(emotions_dict.keys())
    values = list(emotions_dict.values())
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        name='Emotions',
        line=dict(color='rgb(102, 126, 234)', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, max(values) * 1.2] if values else [0, 1],
                gridcolor='rgba(102, 126, 234, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(102, 126, 234, 0.2)'
            )
        ),
        showlegend=False,
        title=dict(text=title, font=dict(size=16, color='#667eea', family='Poppins')),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Initialize session state
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []
if 'weights' not in st.session_state:
    st.session_state.weights = {"alpha": 0.3, "beta": 0.5, "gamma": 0.2}

# Main App
def main():
    # Load and encode the background image
    import base64
    from pathlib import Path
    import os
    
    # Try to load the movie wall background
    bg_image_base64 = ""
    
    # Check multiple possible locations (Colab-friendly)
    possible_paths = [
        "movie_wall_bg.png",  # Same directory as script
        "./movie_wall_bg.png",
        "app_movie_bg.png",
        "/content/movie_wall_bg.png",  # Colab default
        Path("movie_wall_bg.png"),
        Path("/mnt/user-data/outputs/movie_wall_bg.png"),
        Path("/mnt/user-data/uploads/1761691473043_image.png")
    ]
    
    loaded_from = None
    for path in possible_paths:
        path_str = str(path)
        if os.path.exists(path_str):
            try:
                with open(path_str, "rb") as image_file:
                    bg_image_base64 = base64.b64encode(image_file.read()).decode()
                loaded_from = path_str
                break
            except Exception as e:
                continue
    
    if not bg_image_base64:
        st.warning("⚠️ Background image not found! Save your image as 'movie_wall_bg.png' in the same folder as this script.")
    
    # Hero Section with background
    hero_style = f"""
        <style>
        .hero-section {{
            background-image: url('data:image/png;base64,{bg_image_base64}');
        }}
        </style>
    """ if bg_image_base64 else ""
    
    st.markdown(hero_style, unsafe_allow_html=True)
    
    st.markdown('''
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="main-header">🎬 EMOTION-BASED<br>MOVIE RECOMMENDER</h1>
            <div class="decorative-line"></div>
            <p class="sub-header">Because your therapist said you need to "feel your feelings" 🎭✨</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Add spacing after hero
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load data
    movie_sources, user_df, streaming_data = load_data()
    
    if movie_sources is None:
        st.stop()
    
    # Count movies with each source combination
    with_all_3 = sum(1 for srcs in movie_sources.values() if all(k in srcs for k in ['user', 'critic', 'yt']))
    with_2 = sum(1 for srcs in movie_sources.values() if sum(1 for k in ['user', 'critic', 'yt'] if k in srcs) == 2)
    with_1 = sum(1 for srcs in movie_sources.values() if sum(1 for k in ['user', 'critic', 'yt'] if k in srcs) == 1)
    
    st.markdown(f'<div class="info-box">📊 We analyzed <b>{with_all_3}</b> movies so thoroughly they filed restraining orders • <b>{with_2}</b> got partial analysis (we got bored) • <b>{with_1}</b> barely made the cut</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 style="text-align: center;">⚙️ Settings</h2>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Context Mode
        st.subheader("🎭 Context Mode")
        selected_mode = st.selectbox("Choose a mood preset", list(CONTEXT_MODES.keys()), index=0)
        
        st.markdown("---")
        
        # Emotion Sliders
        st.subheader("🎨 Emotion Mix")
        
        if selected_mode != "Custom" and CONTEXT_MODES[selected_mode] is not None:
            emotion_values = CONTEXT_MODES[selected_mode]
            st.info(f"✨ Using **{selected_mode}** preset\n\n*(We're basically your emotional sommelier right now)*")
        else:
            emotion_values = {}
            for emo in EMOS:
                emoji = EMOTION_EMOJIS.get(emo, "")
                emotion_values[emo] = st.slider(
                    f"{emoji} {emo.capitalize()}",
                    0.0, 1.0, 0.14, 0.05,
                    key=f"emotion_{emo}"
                )
        
        st.markdown("---")
        
        # Perspective Weights
        st.subheader("👥 Perspective Weights")
        st.caption("Who gets to judge your movie choices? (Spoiler: everyone)")
        alpha = st.slider("🎓 Critics", 0.0, 1.0, st.session_state.weights["alpha"], 0.05)
        beta = st.slider("👤 Audience", 0.0, 1.0, st.session_state.weights["beta"], 0.05)
        gamma = st.slider("🎥 YouTube", 0.0, 1.0, st.session_state.weights["gamma"], 0.05)
        
        total = alpha + beta + gamma
        if total > 0:
            alpha, beta, gamma = alpha/total, beta/total, gamma/total
        
        st.session_state.weights = {"alpha": alpha, "beta": beta, "gamma": gamma}
        
        st.markdown("---")
        
        # Streaming Platform Filter
        st.subheader("📺 Streaming Platforms")
        st.caption("Because you're paying for like 8 subscriptions anyway:")
        
        platforms = ['Netflix', 'Amazon Prime', 'Hulu', 'Disney+', 'HBO Max', 'Apple TV+', 'Paramount+']
        selected_platforms = []
        
        cols = st.columns(2)
        for i, platform in enumerate(platforms):
            with cols[i % 2]:
                if st.checkbox(platform, key=f"platform_{platform}"):
                    selected_platforms.append(platform)
        
        st.markdown("---")
        
        # Advanced Filters
        st.subheader("🔍 Advanced Filters")
        st.caption("For the picky people (you know who you are)")
        
        top_k = st.number_input("Number of recommendations", 5, 50, 10, 5)
        
        only_complete = st.checkbox("✨ Only movies with all 3 sources", value=True, 
                                     help="We did ALL the homework on these ones")
        
        enable_hidden_gems = st.checkbox("💎 Hidden Gems Mode", 
                                         help="For hipsters who claim they 'liked it before it was cool'")
        
        st.markdown("---")
        
        if st.button("🔄 Reset All Settings", use_container_width=True):
            st.session_state.weights = {"alpha": 0.3, "beta": 0.5, "gamma": 0.2}
            st.rerun()
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown('<p class="section-header">🎯 Your Emotional Baggage</p>', unsafe_allow_html=True)
        query_fig = create_emotion_radar(emotion_values, "What You're Looking For (Good Luck!)")
        st.plotly_chart(query_fig, use_container_width=True)
        
        st.markdown('<p class="section-header">⚖️ The Judge Panel</p>', unsafe_allow_html=True)
        weight_df = pd.DataFrame({
            "Source": ["Critics", "Audience", "YouTube"],
            "Weight": [alpha, beta, gamma]
        })
        fig_weights = px.pie(
            weight_df, 
            values='Weight', 
            names='Source',
            color_discrete_sequence=['#667eea', '#764ba2', '#f5576c'],
            hole=0.4
        )
        fig_weights.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=12
        )
        fig_weights.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins', size=12)
        )
        st.plotly_chart(fig_weights, use_container_width=True)
    
    with col1:
        st.markdown('<p class="section-header">🎬 Your Recommendations</p>', unsafe_allow_html=True)
        
        if st.button("🔍 Find My Soulmate Movies", type="primary", use_container_width=True):
            with st.spinner("✨ Consulting the movie gods and doing some emotional math..."):
                q = make_query_vector(**emotion_values)
                results = rank_movies(movie_sources, q, alpha, beta, gamma, top_k, 
                                    hidden_gems=enable_hidden_gems, 
                                    only_complete=only_complete,
                                    selected_platforms=selected_platforms)
                
                if not results:
                    st.warning("🔍 No movies found. Your emotional requirements are... ambitious. Try lowering your standards?")
                else:
                    st.success(f"🎉 Found {len(results)} movies that totally get you (probably better than your friends do)!")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    for idx, (movie, score, v, srcs) in enumerate(results, 1):
                        explanation = explain_match(q, v, movie, st.session_state.weights, srcs)
                        
                        # Movie Card Container
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                        
                        st.markdown(f'<p class="movie-title">#{idx} • {explanation["movie"]}</p>', unsafe_allow_html=True)
                        
                        col_a, col_b, col_c = st.columns([2, 2, 1])
                        
                        with col_a:
                            st.markdown(f'<p class="match-score">{explanation["match_pct"]:.1f}% Match</p>', unsafe_allow_html=True)
                            
                            # Show which sources were used
                            sources_text = " + ".join(explanation["sources"])
                            st.caption(f"📊 **Sources:** {sources_text.upper()}")
                            
                            # Show streaming platforms if available
                            if 'streaming' in srcs:
                                streaming_info = srcs['streaming']
                                available_on = [p for p, avail in streaming_info['platforms'].items() if avail]
                                
                                if available_on:
                                    st.write("**📺 Available on:**")
                                    platform_colors = {
                                        'Netflix': '#E50914',
                                        'Amazon Prime': '#00A8E1',
                                        'Hulu': '#1CE783',
                                        'Disney+': '#113CCF',
                                        'HBO Max': '#B100CD',
                                        'Apple TV+': '#000000',
                                        'Paramount+': '#0064FF'
                                    }
                                    
                                    platforms_html = ""
                                    for platform in available_on:
                                        color = platform_colors.get(platform, '#666')
                                        platforms_html += f'<span class="emotion-tag" style="background-color: {color}; color: white; font-size: 0.75rem;">{platform}</span>'
                                    st.markdown(platforms_html, unsafe_allow_html=True)
                                    
                                    # Show price tier
                                    if streaming_info.get('price_tier'):
                                        st.caption(f"💰 **Price Tier:** {streaming_info['price_tier']}")
                            
                            # Sentiment (averaged across available sources)
                            sentiments = [s for s in [explanation['sentiment_user'], explanation['sentiment_critic'], explanation['sentiment_yt']] if s is not None]
                            if sentiments:
                                avg_sent = np.mean(sentiments)
                                sent_emoji = "😊" if avg_sent > 0.3 else "😐" if avg_sent > -0.3 else "😞"
                                st.write(f"**Overall Sentiment:** {sent_emoji} {avg_sent:.2f}")
                            
                            # Rating
                            if explanation['rating']:
                                st.write(f"⭐ **IMDb Rating:** {explanation['rating']:.1f}/10")
                        
                        with col_b:
                            st.write("**🎯 Key Emotions:**")
                            for emo in explanation["top_emotions"]:
                                emoji = EMOTION_EMOJIS[emo]
                                color = EMOTION_COLORS[emo]
                                st.markdown(
                                    f'<span class="emotion-tag" style="background-color: {color}; color: white;">'
                                    f'{emoji} {emo.capitalize()}</span>',
                                    unsafe_allow_html=True
                                )
                            
                            # Dominant emotion
                            if explanation['dominant_user']:
                                dom = explanation['dominant_user']
                                dom_emoji = EMOTION_EMOJIS.get(dom, "")
                                st.write(f"**🎭 Dominant:** {dom_emoji} {dom.capitalize()}")
                        
                        with col_c:
                            st.write("**Feedback:**")
                            col_up, col_down = st.columns(2)
                            with col_up:
                                st.button("👍", key=f"up_{idx}_{movie}")
                            with col_down:
                                st.button("👎", key=f"down_{idx}_{movie}")
                        
                        # Detailed breakdown - SHOW INDIVIDUAL SOURCES
                        with st.expander("📊 Detailed Emotion Breakdown by Source"):
                            # Show which sources are available
                            available_sources = explanation["sources"]
                            
                            # Create tabs for each source
                            if len(available_sources) == 3:
                                tab1, tab2, tab3 = st.tabs(["👤 User Reviews", "🎓 Critic Reviews", "🎥 YouTube Reactions"])
                                tabs = [tab1, tab2, tab3]
                                source_names = ["user", "critic", "yt"]
                            elif len(available_sources) == 2:
                                tab1, tab2 = st.tabs([f"📊 {available_sources[0].upper()}", 
                                                       f"📊 {available_sources[1].upper()}"])
                                tabs = [tab1, tab2]
                                source_names = available_sources
                            else:
                                tab1 = st.container()
                                tabs = [tab1]
                                source_names = available_sources
                            
                            # Display each source's emotion profile
                            for tab, source_key in zip(tabs, source_names):
                                with tab:
                                    if source_key in srcs and not source_key.startswith('meta_'):
                                        vec = srcs[source_key]
                                        
                                        # Verify normalization
                                        l2_norm = np.linalg.norm(vec)
                                        
                                        st.caption(f"✅ **L2 Norm:** {l2_norm:.3f} (normalized for cosine similarity)")
                                        
                                        # Show emotions
                                        emotion_dict = {EMOS[i]: float(vec[i]) for i in range(len(EMOS))}
                                        
                                        col_a, col_b = st.columns(2)
                                        
                                        with col_a:
                                            st.write("**Emotion Values:**")
                                            for emo, val in emotion_dict.items():
                                                emoji = EMOTION_EMOJIS[emo]
                                                st.progress(val, text=f"{emoji} **{emo.capitalize()}:** {val:.3f}")
                                        
                                        with col_b:
                                            radar = create_emotion_radar(emotion_dict, f"{source_key.upper()} Profile")
                                            st.plotly_chart(radar, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
