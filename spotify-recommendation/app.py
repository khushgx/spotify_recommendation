from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up authentication to use the Spotify API
load_dotenv()
client_id = os.environ["SPOTIFY_CLIENT_ID"]
client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

genius_token = os.environ["GENIUS_TOKEN"]

# Set up authentication to use the Genius API for lyrics
genius = lyricsgenius.Genius(genius_token)

class Graph:
    def __init__(self, n):
        if n <= 0:
            raise ValueError("n must be positive.")
        self.adj_list = [{} for _ in range(n)]

    def size(self):
        return len(self.adj_list)

    def add_edge(self, u, v, weight):
        if u < 0 or u >= self.size() or v < 0 or v >= self.size() or u == v:
            raise ValueError("Invalid vertices.")
        self.adj_list[u][v] = weight

    def out_neighbors(self, v):
        if v < 0 or v >= self.size():
            raise ValueError("Invalid vertex.")
        return self.adj_list[v].keys()

    def in_neighbors(self, v):
        if v < 0 or v >= self.size():
            raise ValueError("Invalid vertex.")
        incoming = []
        for i in range(self.size()):
            if v in self.adj_list[i]:
                incoming.append(i)
        return incoming

    def pagerank(self, d=0.85, max_iter=100, tol=1e-6):
        n = self.size()
        ranks = [1 / n for _ in range(n)]
        new_ranks = [0 for _ in range(n)]
        for _ in range(max_iter):
            for node in range(n):
                local_rank = sum(
                    [ranks[neighbor] / len(self.out_neighbors(neighbor)) for neighbor in self.in_neighbors(node)])
                new_ranks[node] = (1 - d) / n + d * local_rank

            if sum(abs(new_ranks[i] - ranks[i]) for i in range(n)) < tol:
                break

            ranks, new_ranks = new_ranks, ranks

        return ranks


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['user_id']
        category = request.form['category']
        print(category)
        playlists, public_playlists, playlist_names = get_playlists(user_id, category)
        similarity_matrix = calculate_similarity_matrix(playlists, public_playlists)
        recommendations = recommend_playlists(similarity_matrix, playlist_names, len(playlists))
        return render_template('index.html', recommendations=recommendations, limit = 1)
    return render_template('index.html')

def get_playlists(user_id, category):
    # Retrieve the user's playlists from the Spotify API
    playlists = sp.user_playlists(user_id)['items']

    public_playlists = sp.category_playlists(category_id=category, country='US')['playlists']['items']
    public_playlists = [playlist for playlist in public_playlists if playlist is not None]
    playlist_names = []

    for playlist in playlists + public_playlists:
        playlist_names.append(playlist['name'])

    return playlists, public_playlists, playlist_names


def calculate_audio_similarity_matrix(playlists, public_playlists):
    playlist_audio_features = {}
    max_features = 0
    for playlist in playlists + public_playlists:
        playlist_id = playlist['id']
        tracks = sp.playlist_tracks(playlist_id)['items']
        tracks = [track for track in tracks if track.get('track') is not None]
        track_uris = []
        for track in tracks:
            if track is not None and 'track' in track and 'uri' in track['track']:
                track_uris.append(track['track']['uri'])
        track_audio_features = sp.audio_features(track_uris)
        track_audio_features_values = [[float(value) for key, value in track.items() if key != 'uri' and isinstance(value, float)] for track in track_audio_features]
        max_features = max(max_features, len(track_audio_features_values[0]))
        playlist_audio_features[playlist_id] = np.mean(track_audio_features_values, axis=0)

    # Pad the audio features with zeros
    padded_audio_features = []
    for features in playlist_audio_features.values():
        num_features = len(features)
        if num_features < max_features:
            padded_features = np.zeros(max_features)
            padded_features[:num_features] = features
        else:
            # Remove extra features, only keeping the first max_features
            padded_features = features[:max_features]
        padded_audio_features.append(padded_features)

    # Print the shapes of the arrays in padded_audio_features
    for idx, arr in enumerate(padded_audio_features):
        print(f"Array {idx}: {arr.shape}")

    audio_feature_values = np.array(padded_audio_features, dtype=np.float32)
    audio_similarity_matrix = cosine_similarity(audio_feature_values)

    return audio_similarity_matrix




def calculate_lyrics_similarity_matrix(playlists, public_playlists):
    playlist_lyrics = {}
    for playlist in playlists + public_playlists:
        playlist_id = playlist['id']
        tracks = sp.playlist_tracks(playlist_id)['items']
        playlist_lyrics[playlist_id] = ''
        for track in tracks:
            try:
                song = genius.search_song(track['track']['name'], track['track']['artists'][0]['name'])
                playlist_lyrics[playlist_id] += song.lyrics + '\n'
            except:
                pass

    # Convert the lyrics to consistent-length vectors using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    lyrics_vectors = vectorizer.fit_transform(playlist_lyrics.values())

    lyrics_similarity_matrix = cosine_similarity(lyrics_vectors)

    return lyrics_similarity_matrix


def calculate_similarity_matrix(playlists, public_playlists):
    audio_similarity_matrix = calculate_audio_similarity_matrix(playlists, public_playlists)
    lyrics_similarity_matrix = calculate_lyrics_similarity_matrix(playlists, public_playlists)
    similarity_matrix = (audio_similarity_matrix + lyrics_similarity_matrix) / 2

    return similarity_matrix



def recommend_playlists(similarity_matrix, playlist_names, num_user_playlists):
    G = Graph(len(similarity_matrix))
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            similarity_score = similarity_matrix[i][j]
            G.add_edge(i, j, similarity_score)
            G.add_edge(j, i, similarity_score)

    pagerank_scores = G.pagerank()
    sorted_pagerank_scores = sorted(enumerate(pagerank_scores), key=lambda x: x[1], reverse=True)

    # Recommend similar playlists to the user
    num_recommendations = 5
    recommendations = []
    for i, (playlist_id, score) in enumerate(sorted_pagerank_scores):
        # Skip user's own playlists
        if playlist_id < num_user_playlists:
            continue

        recommended_playlist_name = playlist_names[playlist_id]
        recommendations.append(recommended_playlist_name)

        if len(recommendations) == num_recommendations:
            break

    return recommendations

if __name__ == '__main__':
    app.run(debug=True)