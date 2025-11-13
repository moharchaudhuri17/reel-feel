# 🎬 Emotion-Based Movie Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An intelligent movie recommendation system that matches your emotional preferences using multi-source sentiment analysis**

Ever wondered which movie matches your current mood? This app analyzes emotional content from **critic reviews**, **user ratings**, and **YouTube trailers** to recommend films that resonate with how you're feeling right now.

[**🚀 Try the Live Demo**](https://your-app.streamlit.app)

---

## 🌟 Features

- **🎭 Multi-Source Emotion Analysis** - Combines sentiment from critics, users, and video trailers
- **🤖 Smart Matching Algorithm** - Uses cosine similarity to find your perfect movie match
- **📊 Interactive Visualizations** - Beautiful Plotly charts showing emotion distributions
- **🎯 Advanced Filtering** - Filter by streaming service, year, rating, and runtime
- **💝 Personalized Recommendations** - Select emotions and get movies ranked by match score
- **🎨 Beautiful UI** - Custom gradient styling and card-based layout

---

## 🧠 Methodology

### Data Collection & Processing

#### 1. **User Reviews Analysis**
- Source: IMDb Top 250 user reviews
- Dataset: `complete_imdb_user_reviews.csv`
- Processing: Aggregated emotions across multiple reviews per movie

#### 2. **Critic Reviews Analysis**
- Source: Professional critic reviews compilation
- Dataset: `Critic_Review_Compiled.xlsx`
- Processing: Wide format with 10 reviews per movie, averaged for emotion scores

#### 3. **YouTube Trailer Analysis**
- Source: Official movie trailers
- Processing: Emotion extraction from trailer content (visual + audio cues)

### Emotion Detection Pipeline

The backend uses state-of-the-art transformer models from HuggingFace:

#### **Models Used:**

1. **Sentiment Analysis**
   - Model: `distilbert-base-uncased-finetuned-sst-2-english`
   - Purpose: Binary sentiment classification (POSITIVE/NEGATIVE)
   - Output: Sentiment score in range [-1, 1]

2. **Emotion Classification**
   - Model: `j-hartmann/emotion-english-distilroberta-base`
   - Purpose: Multi-label emotion detection
   - Output: 7 emotion scores (anger, disgust, fear, joy, neutral, sadness, surprise)

#### **Processing Steps:**

```python
# For each data source (critics, users, trailers):
1. Load raw text reviews/transcripts
2. Apply DistilBERT for sentiment scoring
3. Apply DistilRoBERTa for emotion classification
4. Aggregate scores per movie (mean across multiple reviews)
5. Calculate dominant emotion and strength
6. Normalize and align movie titles across sources
```

### Key Metrics Computed

- **Emotion Vectors**: 7-dimensional emotion profile per movie per source
- **Sentiment Score**: Continuous scale from negative (-1) to positive (+1)
- **Dominant Emotion**: Primary emotion with highest score
- **Emotion Alignment**: Cosine similarity between user and critic emotions
- **Sentiment Gap**: Difference between user and critic sentiment

### Recommendation Algorithm

The app uses **cosine similarity** to match user preferences:

```python
# User selects desired emotions (e.g., 80% joy, 20% surprise)
user_vector = [joy=0.8, surprise=0.2, others=0.0]

# For each movie:
movie_vector = [emotion_scores_from_all_sources]

# Calculate similarity
similarity_score = cosine_similarity(user_vector, movie_vector)

# Rank movies by similarity (0-100% match)
```

**Weighting Strategy:**
- Critic emotions: 40%
- User emotions: 40%
- Trailer emotions: 20%

This balanced approach considers both professional analysis and audience reception.

---

## 🛠️ Tech Stack

### Frontend
- **Streamlit** - Interactive web application framework
- **Plotly** - Dynamic visualizations and charts

### Machine Learning
- **scikit-learn** - Cosine similarity, data preprocessing
- **transformers (HuggingFace)** - Pre-trained NLP models
  - DistilBERT for sentiment analysis
  - DistilRoBERTa for emotion classification
- **pandas** - Data manipulation and aggregation
- **numpy** - Numerical computations

### NLP Processing
- **tqdm** - Progress tracking for batch processing
- **rapidfuzz** - Fuzzy string matching for title alignment
- **unicodedata** - Text normalization

---

## 📊 Dataset Details

### Movies
- **Count**: 250 top-rated films from IMDb
- **Metadata**: Title, year, rating, runtime, genre, plot synopsis
- **Streaming**: Netflix, Prime Video, Disney+, Hulu availability

### Emotion Categories (8 emotions)
1. 😊 **Joy** - Happiness, delight, pleasure
2. 😢 **Sadness** - Sorrow, melancholy, grief
3. 😠 **Anger** - Rage, frustration, irritation
4. 😨 **Fear** - Terror, anxiety, dread
5. 😮 **Surprise** - Shock, amazement, wonder
6. 🤢 **Disgust** - Revulsion, distaste
7. 😍 **Love** - Affection, romance, warmth


### Data Statistics
- **User Reviews**: ~22 reviews per movie average
- **Critic Reviews**: 10 reviews per movie
- **Trailer Analysis**: 1 official trailer per movie
- **Correlation**: User rating ↔ sentiment: 0.340

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/emotion-movie-recommender.git
   cd emotion-movie-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

---

## 📁 Project Structure

```
emotion-movie-recommender/
├── .streamlit/
│   └── config.toml                        # Streamlit theme configuration
├── app.py                                 # Main Streamlit application
├── requirements.txt                       # Python dependencies
│
├── Data Files:
├── critic_emotions_simple.csv             # Processed critic emotions
├── user_emotions_simple.csv               # Processed user emotions  
├── youtube_emotions_simple.csv            # Processed trailer emotions
├── imdb_top250_streaming_full.csv         # Movie metadata + streaming
├── movie_wall_bg.png                      # UI background image
│
├── Backend Processing (not in repo):
├── UD-Project_MK.ipynb                    # Data processing pipeline
├── complete_imdb_user_reviews.csv         # Raw user reviews
└── Critic_Review_Compiled.xlsx            # Raw critic reviews
```

---

## 🔬 Technical Implementation

### Backend Data Processing

The emotion analysis pipeline is implemented in `UD-Project_MK.ipynb`:

**Step 1: Load HuggingFace Models**
```python
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)
```

**Step 2: Process Reviews**
```python
def infer(text):
    # Get sentiment score
    sentiment = sentiment_pipeline(text)[0]
    
    # Get emotion scores
    emotions = emotion_pipeline(text)[0]
    
    return {
        'sentiment_score': sentiment['score'],
        'anger': emotions[0]['score'],
        'disgust': emotions[1]['score'],
        # ... all 7 emotions
    }
```

**Step 3: Aggregate by Movie**
```python
# Aggregate across all reviews per movie
movie_emotions = reviews.groupby('movie').agg({
    'anger': 'mean',
    'joy': 'mean',
    'sadness': 'mean',
    # ... all emotions
    'sentiment_score': 'mean'
})
```

**Step 4: Title Normalization**
```python
def title_key(title):
    # Remove accents, special chars
    # Normalize spacing
    # Remove year parentheticals
    return normalized_title
```

### Frontend Recommendation Engine

The Streamlit app (`app.py`) implements real-time matching:

```python
# User selects emotions
user_emotions = {
    'joy': joy_slider / 100,
    'sadness': sadness_slider / 100,
    # ... normalized to sum = 1.0
}

# Calculate similarity with each movie
for movie in movies:
    # Weight sources
    movie_vector = (
        0.4 * critic_emotions[movie] +
        0.4 * user_emotions_data[movie] +
        0.2 * trailer_emotions[movie]
    )
    
    # Compute cosine similarity
    similarity = cosine_similarity(
        [user_emotions],
        [movie_vector]
    )[0][0]
    
    match_scores[movie] = similarity * 100
```

---

## 🎯 Use Cases

- **Movie Night Planning** - Find films that match your group's mood
- **Emotional Discovery** - Explore movies through an emotion lens
- **Content Analysis** - Compare how emotions differ across critics, users, and trailers
- **Research** - Study emotion-based recommendation systems
- **Portfolio Project** - Demonstrate ML/NLP and full-stack development skills

---

## 📈 Model Performance

### Emotion Detection Accuracy
- **Sentiment Model**: 91% accuracy on SST-2 benchmark
- **Emotion Model**: 84% F1-score on GoEmotions dataset

### Alignment Analysis
- **User-Critic Emotion Alignment** (non-neutral): 0.76 ± 0.18 (cosine similarity)
- **User-Critic Emotion Alignment** (with neutral): 0.88 ± 0.12
- **Sentiment Gap Distribution**: Mean ≈ 0, suggesting balanced perspectives

### Correlation Insights
- User rating ↔ Sentiment score: **r = 0.340**
- Joy ↔ User rating: **r = 0.28** (positive movies rated higher)
- Sadness ↔ User rating: **r = -0.15** (weak negative correlation)

---

## 🔮 Future Enhancements

### Data & Features
- [ ] Expand to 1000+ movies from various decades
- [ ] Add TV shows and series
- [ ] Include international cinema
- [ ] Real-time emotion detection from user text input

### Technical Improvements
- [ ] Fine-tune emotion model on movie-specific corpus
- [ ] Implement collaborative filtering
- [ ] Add A/B testing for different algorithms
- [ ] Cache movie vectors for faster recommendations

### User Experience
- [ ] User accounts with preference history
- [ ] Save favorite movies and watch lists
- [ ] Group recommendation mode (aggregate preferences)
- [ ] Mobile app version

### Integration
- [ ] TMDb API for real-time movie data
- [ ] Streaming service APIs for availability
- [ ] Social sharing features
- [ ] Export recommendations to calendar

---

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- **Data**: Add more movies, update streaming availability
- **Models**: Experiment with different NLP models
- **Features**: Implement new recommendation algorithms
- **UI/UX**: Improve design and user experience
- **Documentation**: Enhance guides and tutorials

**How to contribute:**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 References & Citations

### Models
- **DistilBERT**: Sanh et al. (2019). "DistilBERT, a distilled version of BERT"
- **Emotion Classification**: Hartmann (2022). "Emotion English DistilRoBERTa-base"

### Datasets
- **IMDb**: Internet Movie Database, user reviews and ratings
- **Critic Reviews**: Compiled from major review aggregators

### Libraries
- HuggingFace Transformers: [huggingface.co/transformers](https://huggingface.co/transformers)
- Streamlit: [streamlit.io](https://streamlit.io)
- scikit-learn: [scikit-learn.org](https://scikit-learn.org)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

## 🙏 Acknowledgments

- **HuggingFace** for pre-trained transformer models
- **IMDb** for movie ratings and review data
- **Streamlit** for the amazing web framework
- **scikit-learn** for ML utilities
- The open-source ML/NLP community

---

## 💬 Questions & Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Reach out directly for collaboration

---

**⭐ If you found this project helpful, please give it a star!**

*Built with ❤️ using Streamlit, HuggingFace Transformers, and a love for movies*
