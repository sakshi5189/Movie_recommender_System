import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('🎬 Movie Recommender System')

# Load dataset
movies = pd.read_csv("movies.csv")

# ----------- FUNCTIONS -----------

def convert(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except:
        pass
    return L

def convert3(obj):
    L = []
    count = 0
    try:
        for i in ast.literal_eval(obj):
            if count != 3:
                L.append(i['name'])
                count += 1
            else:
                break
    except:
        pass
    return L

def fetch_director(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        pass
    return L

# ----------- PREPROCESSING -----------

movies = movies[['title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine all features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Convert list → string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# ----------- VECTOR + SIMILARITY -----------

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

# ----------- RECOMMEND FUNCTION -----------

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    return [movies.iloc[i[0]].title for i in movie_list]

# ----------- UI -----------

movie_list = movies['title'].values

selected_movie_name = st.selectbox('Select a movie', movie_list)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)

    st.subheader("🎥 Recommended Movies:")
    for movie in recommendations:
        st.write(movie)