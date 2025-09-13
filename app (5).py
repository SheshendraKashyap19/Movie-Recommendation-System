
import streamlit as st
import pickle

# Load the portable model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

global_mean = model["global_mean"]
movie_bias = model["movie_bias"]
user_bias = model["user_bias"]

st.title("Movie Rating Predictor")

user_id = st.number_input("Enter User ID", min_value=1, step=1)
movie_id = st.number_input("Enter Movie ID", min_value=1, step=1)

if st.button("Predict Rating"):
    pred = global_mean + movie_bias.get(movie_id, 0) + user_bias.get(user_id, 0)
    pred = min(max(pred, 0.5), 5.0)  # clip rating to 0.5-5.0
    st.success(f"Predicted Rating: {pred:.2f}")
