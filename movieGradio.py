import gradio as gr
import requests
import re
import random
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMTQ3ZDUzNmE4ODZlNzhmZWVhODEwM2MwMTQ2MDFiMiIsIm5iZiI6MTc0MjU4NjY0Mi4yNjEsInN1YiI6IjY3ZGRjMzEyMDUxY2JhOTA2NmY1NjVlOSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.s65IRbAthvobyw0BlQBz0qbisHXz5Zu9_xDqNAeSp94"  # Replace with your TMDB API Read Access Token

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json;charset=utf-8"
}

# TMDB genre keyword to ID mapping
GENRE_MAP = {
    "action": 28,
    "comedy": 35,
    "drama": 18,
    "romance": 10749,
    "horror": 27,
    "thriller": 53,
    "fantasy": 14,
    "animation": 16,
    "science fiction": 878,
    "family": 10751,
    "mystery": 9648
}

# Keyword → Genre mapping
MOOD_GENRE_MAP = {
    "funny": ["comedy"],
    "feel-good": ["comedy", "family"],
    "dark": ["thriller", "drama"],
    "uplifting": ["drama", "family"],
    "sad": ["drama"],
    "scary": ["horror"],
    "romantic": ["romance"],
    "suspenseful": ["thriller"],
    "trippy": ["science fiction", "fantasy"],
    "cozy": ["animation", "family"]
}

# --- TMDB Movie Details ---
def get_movie_details(movie_id):
    url = f"{TMDB_API_BASE_URL}/movie/{movie_id}"
    params = {"append_to_response": "credits"}
    response = requests.get(url, headers=headers, params=params)
    if not response.ok:
        return None
    data = response.json()
    genres = [g["name"].lower() for g in data.get("genres", [])]
    overview = data.get("overview", "")
    credits = data.get("credits", {})
    cast = [c["name"] for c in credits.get("cast", [])[:5]]
    director = next((c["name"] for c in credits.get("crew", []) if c["job"] == "Director"), "N/A")
    return {
        "id": movie_id,
        "title": data.get("title", ""),
        "overview": overview,
        "genres": genres,
        "rating": data.get("vote_average", 0),
        "cast": cast,
        "director": director,
        "poster_path": data.get("poster_path")
    }

# --- Get Movies by Genre Filters (Prompt Mode) ---
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
        results = response.json().get("results", [])
        for movie in results:
            movie_details = get_movie_details(movie["id"])
            if movie_details:
                all_movies.append(movie_details)
    return all_movies

# --- Get Movies by Reference Title (Title Mode) ---
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

# --- NLP and Relevance Ranking ---
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def genre_similarity(genres1, genres2):
    set1 = set(genres1)
    set2 = set(genres2)
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

def compute_relevance(prompt_text, movies, reference_genres=None):
    corpus = [prompt_text] + [m["overview"] for m in movies]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    text_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    final_scores = []
    for i, m in enumerate(movies):
        genre_score = genre_similarity(reference_genres, m["genres"]) if reference_genres else 0
        score = 0.6 * text_scores[i] + 0.4 * genre_score
        final_scores.append((m, score))

    return sorted(final_scores, key=lambda x: x[1], reverse=True)

# --- Main Logic With Toggle ---
def smart_recommender(prompt, mode):
    if mode == "Prompt-Based":
        movies = get_movies_from_prompt(prompt, pages=3)
        if not movies:
            return "❌ No movies found. Try a different prompt."
        ranked = compute_relevance(prompt, movies)[:6]
        heading = f"## Prompt-Based Recommendations for: *{prompt}*\n\n"

    elif mode == "Movie Title Reference":
        reference, candidates = get_movies_like_reference(prompt)
        if not reference:
            return "❌ Could not find a movie by that title. Try again."
        ranked = compute_relevance(reference["overview"], candidates, reference["genres"])[:6]
        heading = f"## Movies similar to: *{reference['title']}*\n\n"

    else:
        return "❌ Invalid mode."

    output = heading
    for movie, score in ranked:
        polarity, subjectivity = analyze_sentiment(movie["overview"])
        poster_url = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if movie["poster_path"] else ""
        output += f"### {movie['title']} — Score: {score:.2f}\n"
        if poster_url:
            output += f"<img src='{poster_url}' width='120'><br>\n"
        output += f"**Genres:** {', '.join(movie['genres'])}  \n"
        output += f"**Rating:** {movie['rating']} | **Director:** {movie['director']}  \n"
        output += f"**Sentiment:** Polarity {polarity:.2f}, Subjectivity {subjectivity:.2f}  \n"
        output += f"**Overview:** {movie['overview']}  \n---\n"
    return output

# --- Gradio UI ---
demo = gr.Interface(
    fn=smart_recommender,
    inputs=[
        gr.Textbox(label="Enter a movie description or title"),
        gr.Radio(["Prompt-Based", "Movie Title Reference"], label="Search Mode", value="Prompt-Based")
    ],
    outputs=gr.Markdown(),
    title="🎬 Smart Movie Recommender",
    description="Choose 'Prompt-Based' for mood/genre queries like 'funny feel-good movie', or 'Movie Title Reference' to get movies similar to an existing one like 'Interstellar' or 'Spider-Man'."
)

demo.launch()