
import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Movie Rating Predictor")

user_id = st.number_input("Enter User ID", min_value=1, step=1)
movie_id = st.number_input("Enter Movie ID", min_value=1, step=1)

if st.button("Predict Rating"):
    global_mean = model["global_mean"]
    movie_bias = model["movie_bias"]
    user_bias = model["user_bias"]

    m_bias = movie_bias.get(movie_id, 0)
    u_bias = user_bias.get(user_id, 0)

    pred = global_mean + m_bias + u_bias
    pred = min(max(pred, 0.5), 5.0)  # clip to 0.5-5.0 range

    st.success(f"Predicted Rating: {pred:.2f}")
