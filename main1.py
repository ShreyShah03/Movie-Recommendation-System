# movie_recommender.py

import pandas as pd
import ast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# --- 1. DATA LOADING AND PREPROCESSING ---
print("Loading and processing movie data from tmdb_5000_movies.csv and tmdb_5000_credits.csv...")

try:
    # Load the datasets
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits_df = pd.read_csv("tmdb_5000_credits.csv")

    # Merge the two dataframes on 'title'
    merged_df = pd.merge(movies_df, credits_df, on='title', suffixes=('_movie', '_credit'))

    # Rename 'id' from movies_df to 'movie_id' for clarity
    merged_df.rename(columns={'id_movie': 'movie_id'}, inplace=True)

    # Convert stringified JSON columns to Python objects
    json_cols = ['genres', 'keywords', 'cast', 'crew']
    for col in json_cols:
        merged_df[col] = merged_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Function to extract names from JSON columns
    def get_names(data_list):
        return [item['name'] for item in data_list]

    # Function to extract director's name from the crew list
    def get_director(crew_list):
        for member in crew_list:
            if member.get('job') == 'Director':
                return member.get('name', 'N/A')
        return "N/A"

    # Apply the extraction functions
    merged_df['genres'] = merged_df['genres'].apply(get_names)
    merged_df['keywords'] = merged_df['keywords'].apply(get_names)
    merged_df['actors'] = merged_df['cast'].apply(lambda x: get_names(x) if x is not None else [])
    merged_df['director'] = merged_df['crew'].apply(lambda x: get_director(x) if x is not None else "N/A")

    # Add a 'moods' column for the recommendation logic
    # This is a simple mapping for demonstration purposes
    mood_mapping = {
        'Action': 'Excited', 'Adventure': 'Excited', 'Fantasy': 'Whimsical',
        'Science Fiction': 'Thoughtful', 'Crime': 'Intense', 'Drama': 'Serious',
        'Thriller': 'Intense', 'Comedy': 'Happy', 'Family': 'Happy', 'Romance': 'Romantic',

    }
    def get_moods(genres):
        return [mood_mapping.get(g, 'Relaxed') for g in genres]
    merged_df['moods'] = merged_df['genres'].apply(get_moods)

    # Create a simplified list of dictionaries for the recommendation engine
    MOVIES_DB = merged_df[['title', 'genres', 'keywords', 'actors', 'director', 'overview', 'popularity', 'moods']].to_dict('records')

    # --- GET AVAILABLE OPTIONS FOR UI ---
    all_genres = sorted(list(set(g for movie in MOVIES_DB for g in movie['genres'])))
    all_moods = sorted(list(set(m for movie in MOVIES_DB for m in movie['moods'])))

except FileNotFoundError:
    print("Error: Could not find the required CSV files. Please ensure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the same directory.")
    exit()

print("Data processing complete. Ready to recommend!")

# --- 2. AI SYSTEM: CONTENT-BASED RECOMMENDER ---

# The AI system is now a content-based recommender using TF-IDF.
# It creates a vector for each movie based on its genres, keywords, and overview.
# This part of the code should only be run once after data loading.

# Create a 'combined_features' string for each movie
def create_combined_features(movie):
    # Combine genres, keywords, and overview into a single string for vectorization
    genres = ' '.join(movie.get('genres', []))
    keywords = ' '.join(movie.get('keywords', []))
    overview = movie.get('overview', '')
    return f"{genres} {keywords} {overview}".lower()

merged_df['combined_features'] = merged_df.apply(create_combined_features, axis=1)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Create the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df['combined_features'])

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

def get_ai_recommendation(movies):
    """Generates a creative recommendation based on the given movie list."""
    if not movies:
        return "The AI assistant suggests that sometimes the best movie is the one you make with your imagination! Try a different combination of filters."

    # Get the top movie by popularity from the list to highlight
    top_movie = max(movies, key=lambda x: x['popularity'])

    # Find the index of the top movie in the original dataframe
    movie_index = merged_df[merged_df['title'] == top_movie['title']].index[0]

    # Get the similarity scores for that movie with all other movies
    sim_scores = list(enumerate(cosine_sim[movie_index]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 2 most similar movies (excluding itself)
    sim_scores = sim_scores[1:3]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Get the titles of the most similar movies
    similar_movies = merged_df.loc[movie_indices, 'title'].tolist()

    return (
        f"Based on your choices, the AI has a special recommendation for you: '{top_movie['title']}'! "
        f"Since you like this kind of movie, the AI also thinks you might enjoy '{similar_movies[0]}' or '{similar_movies[1]}'."
    )

# --- 3. CORE LOGIC FOR RECOMMENDATIONS ---
def get_recommendation_score(movie, query, selected_genres, selected_moods):
    """Calculates a recommendation score for a single movie based on user inputs."""
    score = 0
    query_lower = query.lower()

    # High score for a direct match in the search bar
    if query and (query_lower in movie['title'].lower() or
                  query_lower in movie['director'].lower() or
                  any(query_lower in str(actor).lower() for actor in movie['actors'])):
        score += 100

    # Score for matching selected genres
    movie_genres_lower = [g.lower() for g in movie['genres']]
    for genre in selected_genres:
        if genre.lower() in movie_genres_lower:
            score += 5

    # Score for matching selected moods
    movie_moods_lower = [m.lower() for m in movie['moods']]
    for mood in selected_moods:
        if mood.lower() in movie_moods_lower:
            score += 5

    return score

def recommend_movies(genres, moods, search_query):
    """
    Recommends movies based on moods, genres, and an optional search query.

    Args:
        genres (list): A list of desired genres.
        moods (list): A list of desired moods.
        search_query (str): A string to search for in movie titles, actors, or directors.

    Returns:
        list: A sorted list of movie dictionaries that match the criteria.
    """
    ranked_movies = []
    selected_genres_lower = {g.lower() for g in genres}
    selected_moods_lower = {m.lower() for m in moods}

    for movie in MOVIES_DB:
        score = get_recommendation_score(movie, search_query, selected_genres_lower, selected_moods_lower)
        if score > 0:
            ranked_movies.append({**movie, 'score': score})

    # Sort by score in descending order
    ranked_movies.sort(key=lambda x: x['score'], reverse=True)

    # Return a maximum of 10 movies for readability
    return ranked_movies[:10]

# --- 4. BASIC UI (COMMAND-LINE) ---
def main():
    """Main function to run the command-line interface."""
    print("🎬 Welcome to the Gen Z Movie Recommender! 🎬")
    print("Let's find the perfect movie for your free time.")
    print("-" * 50)

    # Get user input for filters
    print("First, choose your mood(s).")
    print("Available moods:", ', '.join(all_moods))
    user_moods_str = input("Enter moods (comma-separated, e.g., 'happy, excited'): ").strip()
    user_moods = [m.strip() for m in user_moods_str.split(',') if m.strip()]

    print("\nNext, select some genres.")
    print("Available genres:", ', '.join(all_genres))
    user_genres_str = input("Enter genres (comma-separated, e.g., 'action, comedy'): ").strip()
    user_genres = [g.strip() for g in user_genres_str.split(',') if g.strip()]

    print("\nFinally, if you have a specific movie, actor, or director in mind, use the search bar.")
    user_search_query = input("Enter a name (e.g., 'Christopher Nolan') or press Enter to skip: ").strip()

    print("\nSearching for movies based on your preferences...")

    # Get recommendations
    recommended_list = recommend_movies(user_genres, user_moods, user_search_query)

    print("\n" + "=" * 50)
    print("✨ Here are your personalized movie recommendations: ✨")
    if recommended_list:
        for movie in recommended_list:
            print(f"\n🎥 Title: {movie['title']}")
            print(f"  Genres: {', '.join(movie['genres']).title()}")
            print(f"  Actors: {', '.join(movie['actors'][:3])}") # Show top 3 actors
            print(f"  Director: {movie['director']}")
            print(f"  Overview: {movie['overview'][:100]}...") # Truncate overview for brevity
    else:
        print("\n😔 Oops! No movies matched your criteria. Try adjusting your filters!")

    print("\n" + "=" * 50)

    # Display the AI service recommendation
    print("\n🤖 The AI has a special recommendation for you:")
    print(get_ai_recommendation(recommended_list))
    print("-" * 50)

if __name__ == "__main__":
    main()
