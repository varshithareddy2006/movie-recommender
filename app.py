from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv("rating1.csv")
movies = pd.read_csv("movie_new.csv")

# Build user-item matrix and similarity matrix
user_item_matrix = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
similarity_matrix = cosine_similarity(user_item_matrix.fillna(0))
sim_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Recommendation function
def predict_rating(user_id, movie_id, k=30, threshold=0.2):
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
