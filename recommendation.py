
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = {
    'Movie': ['Inception', 'Interstellar', 'The Dark Knight', 'Prestige', 'Memento'],
    'Genre': ['Sci-Fi', 'Sci-Fi', 'Action', 'Drama', 'Thriller']
}


df = pd.DataFrame(movies)

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df['Genre'])


similarity = cosine_similarity(genre_matrix)


def recommend(movie_name):
    if movie_name not in df['Movie'].values:
        return "Movie not found. Please check the name."

    index = df[df['Movie'] == movie_name].index[0]
    score_list = list(enumerate(similarity[index]))
    score_list = sorted(score_list, key=lambda x: x[1], reverse=True)[1:]
    suggested = [df['Movie'][i[0]] for i in score_list[:3]]
    return suggested


movie_you_like = input("Enter a movie name you like: ")
print("You may also like:", recommend(movie_you_like))
