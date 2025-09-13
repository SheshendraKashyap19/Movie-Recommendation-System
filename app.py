import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")
movies['title'] = movies['title'].fillna('')

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['title'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(title, cosine_sim=cosine_sim):
    # Make indices lowercase and strip spaces
    indices = pd.Series(movies.index, index=movies['title'].str.lower().str.strip())
    # Process user input
    title = title.lower().strip()
    
    if title not in indices:
        return ["Movie not found"]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()


# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name to get top 5 recommendations.")

user_input = st.text_input("Movie Name:")
if st.button("Recommend"):
    recommendations = recommend_movies(user_input)
    st.write("Recommended Movies:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
