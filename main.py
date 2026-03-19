import streamlit as st
import pickle

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommendation function
def recommend(movie):
    recommended_movies = []

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# UI
st.title('🎬 Movie Recommender System')

movie_list = movies['title'].values

selected_movie_name = st.selectbox(
    'Select a movie',
    movie_list
)

# Button
if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)

    for movie in recommendations:
        st.write(movie)