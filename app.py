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
def nid_movies(user_id):
  user_id=int(user_id)
  if user_id not in user_item_matrix.index:
    print(f"User {user_id} not found in matrix.")
    return 
  id=[]
  nid=[]
  for m in movies.movieId:
    try:
      rating = user_item_matrix.at[user_id, m]
      if not pd.isna(rating):
        id.append(m)
    except KeyError:
      continue
  for i in movies.movieId:
    if i not in id:
      nid.append(i)
  return nid
# Flask app
app = Flask(__name__)

@app.route('/recommend',methods=['GET'])
def recommend():
    recommend_movies=[]
    user_id=request.args.get('user_id')
    N= int(request.args.get('N', 10))
    n=nid_movies(user_id)
    if n is None:
        return jsonify({"error": "User not found or no movies available for recommendation."}), 404
    else:
        for m in movies.movieId:
            m=int(m)
            if m in n:
                rating= predict_rating(user_id, m)
                if rating > 0:
                    title_row = movies[movies['movieId'] == m]
                    if not title_row.empty:
                        title = title_row.iloc[0]['title']
                    recommend_movies.append({
                        'title': title,
                        'predicted_rating': rating
                    })
    recommend_movies = sorted(recommend_movies, key=lambda x: x['predicted_rating'], reverse=True)[:N]
    return jsonify(recommend_movies)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
