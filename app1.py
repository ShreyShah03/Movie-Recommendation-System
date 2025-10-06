import streamlit as st
from main1 import all_genres, all_moods, recommend_movies, get_ai_recommendation

# 🎬 Streamlit Frontend
st.set_page_config(page_title="Gen Z Movie Recommender", page_icon="🎬", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #1f1c2c, #928dab);
        color: white;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .movie-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.02);
        background: rgba(255, 255, 255, 0.15);
    }
    .stButton button {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-size: 16px;
    }
    .stButton button:hover {
        background: linear-gradient(45deg, #ff4b2b, #ff416c);
        transform: scale(1.05);
    }
    .header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        padding: 20px 0;
        color: #ffcc70;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header">🎬 Gen Z Movie Recommender</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Make your free time awesome by finding the perfect movie based on your <b>mood, genre, or favorite celebrity</b> 🍿✨</p>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🔍 Filters")

selected_moods = st.sidebar.multiselect("Choose your mood(s):", options=all_moods)
selected_genres = st.sidebar.multiselect("Choose genres:", options=all_genres)
search_query = st.sidebar.text_input("Search (Movie, Actor, Director)")

# --- Button Action ---
if st.sidebar.button("Get Recommendations 🎥"):
    recommendations = recommend_movies(selected_genres, selected_moods, search_query)

    if recommendations:
        st.subheader("✨ Your Movie Recommendations ✨")

        # Show results in styled cards
        for movie in recommendations:
            st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
            st.markdown(f"### 🎞️ {movie['title']}")
            st.write(f"**Genres:** {', '.join(movie['genres'])}")
            st.write(f"**Actors:** {', '.join(movie['actors'][:3])}")
            st.write(f"**Director:** {movie['director']}")
            st.write(f"**Overview:** {movie['overview'][:200]}...")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.warning("😔 No movies matched your filters. Try adjusting your choices!")

    # --- AI Pick ---
    st.subheader("🤖 AI Special Pick")
    st.success(get_ai_recommendation(recommendations))
else:
    st.info("👈 Set your filters and click **Get Recommendations 🎥** to start.")
