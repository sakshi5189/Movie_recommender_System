import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page setup
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")

# Load dataset
movies = pd.read_csv("movies.csv")

# Keep required columns
movies = movies[['title', 'overview', 'genres', 'keywords']]
movies.dropna(inplace=True)

# ---------------- PREPROCESSING ----------------

# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Convert genres & keywords (string → list)
movies['genres'] = movies['genres'].apply(lambda x: x.split())
movies['keywords'] = movies['keywords'].apply(lambda x: x.split())

# Combine all features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']

# Convert list → string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# ---------------- VECTOR + SIMILARITY ----------------

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

# ---------------- RECOMMEND FUNCTION ----------------

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    return [movies.iloc[i[0]].title for i in movie_list]

# ---------------- UI ----------------

movie_list = movies['title'].values

selected_movie_name = st.selectbox("🎥 Select a movie", movie_list)

if st.button("🚀 Recommend"):
    recommendations = recommend(selected_movie_name)

    st.subheader("🎬 Recommended Movies:")
    for movie in recommendations:
        st.write(movie)