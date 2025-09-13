import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import get_close_matches

# ---------------------------
# Clean titles
# ---------------------------
def clean_title(title):
    title = str(title).lower().strip()
    title = re.sub(r"\(\d{4}\)", "", title)      # remove year
    title = re.sub(r"\s+", " ", title)          # collapse spaces
    title = title.encode('ascii', errors='ignore').decode()  # remove hidden chars
    return title

# ---------------------------
# Load dataset
# ---------------------------
movies = pd.read_csv("movies.csv", encoding='utf-8-sig')
movies['title'] = movies['title'].fillna('')
movies['clean_title'] = movies['title'].apply(clean_title)

# ---------------------------
# TF-IDF & Cosine similarity
# ---------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['clean_title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------------------
# Indices
# ---------------------------
indices = {title: idx for idx, title in enumerate(movies['clean_title'])}

# ---------------------------
# Recommendation function with fuzzy fallback
# ---------------------------
def recommend_movies(user_input):
    user_input_clean = clean_title(user_input)

    if user_input_clean not in indices:
        # Fuzzy matching fallback
        possible = get_close_matches(user_input_clean, indices.keys(), n=1, cutoff=0.6)
        if possible:
            user_input_clean = possible[0]
        else:
            return ["Movie not found"]

    idx = indices[user_input_clean]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5
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
