from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data of ratings and movies
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
# Build user-item matrix and similarity matrix
user_item_matrix = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
similarity_matrix = cosine_similarity(user_item_matrix.fillna(0))
# Create a DataFrame for the similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Predict rating for a given user and movie using user-based collaborative filtering
def predict_rating(user_id, movie_id, k=40, threshold=0.1):
    user_id = int(user_id)
    movie_id = int(movie_id)
    # if user or movie is not present in the data then return 0
    if movie_id not in user_item_matrix.columns or user_id not in similarity_df.index:
        return 0
    # Get similarity scores of other users to the target user (excluding self)
    sims = similarity_df[user_id].drop(user_id, errors='ignore')
    # Filter by threshold and keep top-k similar users
    sims = sims[sims > threshold].sort_values(ascending=False).head(k)

    total_sim = 0
    total_rating = 0
    # Iterate through similar users and calculate weighted ratings
    for other_user, sim in sims.items():
        try:
            # Get the rating of the movie by the other user
            rating = user_item_matrix.at[other_user, movie_id]
            # If the rating is NaN, skip to the next user
            if pd.isna(rating):
                continue
            #else compute total_rating and total_sim
            total_rating += sim * rating
            total_sim += sim
        except KeyError:
            continue
    # If no similar users or ratings, return 0
    if total_sim == 0:
        return 0  
    else:
        # else return the predicted rating as total_rating divided by total_sim
        return total_rating / total_sim 

# Flask app
app = Flask(__name__)

# Route to predict the rating for a specific user and movie
@app.route('/predict', methods=['GET'])
def predict():
    user_id = request.args.get('user_id')
    movie_id = request.args.get('movie_id')
    # compute the predicted rating for the user and movie
    rating = predict_rating(user_id, movie_id)
    # return predicted rating as a JSON response
    return jsonify({'user_id': user_id, 'movie_id': movie_id, 'predicted_rating': round(rating, 2)})

# Route to recommend movies for a specific user
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    N = int(request.args.get('top_n', 5)) 
    # Check if user_id is in the user-item matrix index
    # if not, return an error message
    if user_id not in user_item_matrix.index:
        return jsonify({"error": f"User ID {user_id} not found in matrix."}), 404
    # Get list of movies the user has already rated
    user_rated = user_item_matrix.loc[user_id].dropna().index.tolist()
    # Find movies the user hasn't rated yet
    unseen_movies = [m for m in user_item_matrix.columns if m not in user_rated]
    # To reduce computation, sample first 300 unseen movies
    sample_movies = unseen_movies[:300]

    predictions = []
    # Predict ratings for sampled unseen movies
    for movie_id in sample_movies:
        pred = predict_rating(user_id, movie_id)
        if pred > 0:
            predictions.append((movie_id, pred))
    # If no predictions are possible, return a message
    if not predictions:
        return jsonify({
            "user_id": user_id,
            "recommendations": [],
            "note": "No recommendations available — try another user or adjust similarity settings."
        })
    else:
        # Sort predictions by rating and take top N
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:N]
        result = [{"movie_id": int(mid), "predicted_rating": round(r, 2)} for mid, r in top_n]
    # Return the recommendations as a JSON response
        return jsonify({"user_id": user_id, "recommendations": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
