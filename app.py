
from flask import Flask, request, jsonify
from recommendation import predicted_ratings, user_item_matrix, movies

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    user_id = request.args.get('user_id')
    movie_id = request.args.get('movie_id')
    rating = predicted_ratings(user_id, movie_id)
    return jsonify({'user_id': user_id, 'movie_id': movie_id, 'predicted_rating': round(rating, 2)})

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    N = int(request.args.get('top_n', 5))  # default = top 5

    if user_id not in user_item_matrix.index:
        return jsonify({"error": f"User ID {user_id} not found in matrix."}), 404

    user_rated = user_item_matrix.loc[user_id].dropna().index.tolist()
    unseen_movies = [m for m in user_item_matrix.columns if m not in user_rated]
    sampled_movies = unseen_movies[:300]

    predictions = []
    for movie_id in sampled_movies:
        pred = predicted_ratings(user_id, movie_id)
        if pred > 0:
            predictions.append((movie_id, pred))
    if not predictions:
        return jsonify({
            "user_id": user_id,
            "recommendations": [],
            "note": "No recommendations available — try another user or adjust similarity settings."
        })
    else:
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:N]
        result = [{"movie_id": int(mid), "predicted_rating": round(r, 2)} for mid, r in top_n]
        return jsonify({"user_id": user_id, "recommendations": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=8080)
