
import pandas as pd

def predict_vectorized(df, model):
    global_mean = model["global_mean"]
    movie_bias = model["movie_bias"]
    user_bias = model["user_bias"]

    m_bias = df["movieId"].map(movie_bias).fillna(0)
    u_bias = df["userId"].map(user_bias).fillna(0)

    return global_mean + m_bias + u_bias
