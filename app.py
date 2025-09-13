import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------------------
# Helper function to clean titles
# ---------------------------
def clean_title(title):
    """
    Lowercase the title, remove years in parentheses, collapse multiple spaces.
    """
    title = title.lower().strip()                    # lowercase + strip spaces
    title = re.sub(r"\(\d{4}\)", "", title)         # remove year like (1995)
    title = re.sub(r"\s+", " ", title)             # collapse multiple spaces into one
    return title

# ---------------------------
# Load dataset
# ---------------------------
# Make sure 'movies.csv' is in the same folder as app.py
movies = pd.read_csv("movies.csv")
movies['title'] = movies['title'].fillna('')

# ---------------------------
# Preprocess titles
# ---------------------------
movies['clean_title'] = movies['title'].apply(clean_title)

# ---------------------------
# TF-IDF Vectorization
# ---------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['clean_title'])

# ---------------------------
# Compute cosine similarity matrix
# ---------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------------------
# Create indices for fast lookup
# ---------------------------
indices = pd.Series(movies.index, index=movies['clean_title'])

# ---------------------------
# Recommendation function
# ---------------------------
def recommend_movies(user_input, cosine_sim=cosine_sim):
    user_input = clean_title(user_input)  # clean user input
    
    if user_input not in indices:
        # Optional: return close matches using fuzzy matching if not found
        return ["Movie not found"]
    
    idx = indices[user_input]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices].tolist()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name to get top 5 recommendations.")

user_input = st.text_input("Movie Name:")

if st.button("Recommend"):
    if not user_input:
        st.write("Please enter a movie name.")
    else:
        recommendations = recommend_movies(user_input)
        st.write("Recommended Movies:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
