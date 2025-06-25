from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Build user-item matrix and similarity matrix
user_item_matrix = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
similarity_matrix = cosine_similarity(user_item_matrix.fillna(0))
sim_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Recommendation function
def predict_rating(user_id, movie_id, k=20, threshold=0.5):
    user_id = int(user_id)
    movie_id = int(movie_id)
    if movie_id not in user_item_matrix.columns or user_id not in sim_df.index:
        return 0

    sims = sim_df[user_id].drop(user_id, errors='ignore')
    sims = sims[sims > threshold].sort_values(ascending=False).head(k)

    total_sim = 0
    total_rating = 0
    for other_user, sim in sims.items():
        try:
            rating = user_item_matrix.at[other_user, movie_id]
            if pd.isna(rating):
                continue
            total_rating += sim * rating
            total_sim += sim
        except KeyError:
            continue
    return total_rating / total_sim if total_sim != 0 else 0

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    user_id = request.args.get('user_id')
    movie_id = request.args.get('movie_id')
    rating = predict_rating(user_id, movie_id)
    return jsonify({'user_id': user_id, 'movie_id': movie_id, 'predicted_rating': round(rating, 2)})
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    N = int(request.args.get('top_n', 5))  # default = top 5
    k = int(request.args.get('k', 30))
    threshold = float(request.args.get('threshold', 0.2))
    user_rated = user_item_matrix.loc[user_id].dropna().index.tolist()
    predictions = []

    for movie_id in user_item_matrix.columns:
        if movie_id not in user_rated:
            pred = predict_rating(user_id, movie_id)
            if pred > 0:
                predictions.append((movie_id, pred))

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:N]
    result = [{"movie_id": int(mid), "predicted_rating": round(r, 2)} for mid, r in top_n]
    return jsonify({"user_id": user_id, "recommendations": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
