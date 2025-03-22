import streamlit as st
import requests
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- TMDB CONFIG ---
TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMTQ3ZDUzNmE4ODZlNzhmZWVhODEwM2MwMTQ2MDFiMiIsIm5iZiI6MTc0MjU4NjY0Mi4yNjEsInN1YiI6IjY3ZGRjMzEyMDUxY2JhOTA2NmY1NjVlOSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.s65IRbAthvobyw0BlQBz0qbisHXz5Zu9_xDqNAeSp94"  # Replace with your TMDB API Read Access Token

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json;charset=utf-8"
}

# --- GENRE MAP ---
GENRE_MAP = {
    "action": 28, "comedy": 35, "drama": 18, "romance": 10749,
    "horror": 27, "thriller": 53, "fantasy": 14, "animation": 16,
    "science fiction": 878, "family": 10751, "mystery": 9648
}

MOOD_GENRE_MAP = {
    "funny": ["comedy"], "feel-good": ["comedy", "family"], "dark": ["thriller", "drama"],
    "uplifting": ["drama", "family"], "sad": ["drama"], "scary": ["horror"],
    "romantic": ["romance"], "suspenseful": ["thriller"], "trippy": ["science fiction", "fantasy"],
    "cozy": ["animation", "family"]
}

# --- TMDB FUNCTIONS ---
def get_movie_details(movie_id):
    url = f"{TMDB_API_BASE_URL}/movie/{movie_id}"
    params = {"append_to_response": "credits"}
    response = requests.get(url, headers=headers, params=params)
    if not response.ok:
        return None
    data = response.json()
    genres = [g["name"].lower() for g in data.get("genres", [])]
    credits = data.get("credits", {})
    cast = [c["name"] for c in credits.get("cast", [])[:5]]
    director = next((c["name"] for c in credits.get("crew", []) if c["job"] == "Director"), "N/A")
    return {
        "id": movie_id, "title": data.get("title", ""), "overview": data.get("overview", ""),
        "genres": genres, "rating": data.get("vote_average", 0), "cast": cast,
        "director": director, "poster_path": data.get("poster_path")
    }

def get_movies_from_prompt(prompt, pages=3):
    prompt = prompt.lower()
    matched_genres = set()
    for keyword, genres in MOOD_GENRE_MAP.items():
        if keyword in prompt:
            matched_genres.update(genres)
    genre_ids = [str(GENRE_MAP[g]) for g in matched_genres if g in GENRE_MAP]
    genre_filter = ",".join(genre_ids) if genre_ids else None

    all_movies = []
    for page in range(1, pages + 1):
        params = {"sort_by": "popularity.desc", "page": page}
        if genre_filter:
            params["with_genres"] = genre_filter
        url = f"{TMDB_API_BASE_URL}/discover/movie"
        response = requests.get(url, headers=headers, params=params)
        if not response.ok:
            continue
        for movie in response.json().get("results", []):
            details = get_movie_details(movie["id"])
            if details:
                all_movies.append(details)
    return all_movies

def get_movies_like_reference(prompt):
    url = f"{TMDB_API_BASE_URL}/search/movie"
    params = {"query": prompt}
    response = requests.get(url, headers=headers, params=params)
    results = response.json().get("results", []) if response.ok else []
    if not results:
        return None, []
    main_movie = get_movie_details(results[0]["id"])
    candidates = get_movies_from_prompt("", pages=3)
    return main_movie, candidates

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def genre_similarity(g1, g2):
    s1, s2 = set(g1), set(g2)
    return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0

def compute_relevance(prompt_text, candidates, reference_genres=None):
    corpus = [prompt_text] + [m["overview"] for m in candidates]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    text_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    final = []
    for i, m in enumerate(candidates):
        genre_score = genre_similarity(reference_genres, m["genres"]) if reference_genres else 0
        final_score = 0.6 * text_scores[i] + 0.4 * genre_score
        final.append((m, final_score))
    return sorted(final, key=lambda x: x[1], reverse=True)

# --- STREAMLIT UI ---
st.title("🎬 AI Movie Recommender")
st.write("Find movies based on mood, genre, or by entering a movie you like!")

mode = st.radio("Search Mode", ["Prompt-Based", "Movie Title Reference"])
user_input = st.text_input("Enter a movie description or title:")

if st.button("Get Recommendations") and user_input:
    if mode == "Prompt-Based":
        movies = get_movies_from_prompt(user_input)
        if not movies:
            st.error("❌ No movies found.")
        else:
            ranked = compute_relevance(user_input, movies)[:6]
            st.markdown(f"### Prompt-Based Recommendations for: *{user_input}*")
    else:
        reference, candidates = get_movies_like_reference(user_input)
        if not reference:
            st.error("❌ Movie not found.")
        else:
            ranked = compute_relevance(reference["overview"], candidates, reference["genres"])[:6]
            st.markdown(f"### Movies similar to: *{reference['title']}*")

    for m, score in ranked:
        st.subheader(f"{m['title']} — Score: {score:.2f}")
        if m["poster_path"]:
            st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}")
        st.markdown(f"""
        **Genres:** {', '.join(m['genres'])}  
        **Rating:** {m['rating']} | **Director:** {m['director']}  
        **Cast:** {', '.join(m['cast'])}  
        **Overview:** {m['overview']}  
        """)