import pandas as pd
ratings=pd.read_csv('ratings.csv')
movies=pd.read_csv('movies.csv')
from sklearn.metrics.pairwise import cosine_similarity

# Example: User-Item Matrix
user_item_matrix = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
# Example: Calculate similarity matrix
similarity_matrix = cosine_similarity(user_item_matrix.fillna(0), user_item_matrix.fillna(0))
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
def predicted_ratings(user_id, movie_id):
  if movie_id not in user_item_matrix.columns:
    return 0
  else:
    total_sim=0
    total_rating=0
    for other_user_id,row in similarity_df.iterrows():
      if other_user_id!=user_id:
        rating=user_item_matrix[movie_id][other_user_id]
        if pd.isna(rating):
            continue
        total_rating+=(similarity_df[user_id][other_user_id]*rating)
        total_sim+=similarity_df[user_id][other_user_id]
    if total_sim==0:
      return 0
    else:
      return total_rating/total_sim
print(predicted_ratings(1,1))
y_true = []
y_pred = []
for _, row in test_data.iterrows():
    user = row['userId']
    movie = row['movieId']
    actual = row['rating']
    predicted = predicted_ratings(user, movie)
    if predicted!=0:
      y_true.append(actual)
      y_pred.append(predicted)
from sklearn.metrics import mean_squared_error
import numpy as np
if y_pred:
  mse=mean_squared_error(y_true,y_pred)
  rmse=np.sqrt(mse)
  print(rmse)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
kf=KFold(n_splits=3,shuffle=True,random_state=42)
k=30
threshold=[0.15,0.2,0.3,0.5,0.7]
result=[]
for i in threshold:
  rmses=[]
  for train_index,test_index in kf.split(ratings):
    train_data=ratings.iloc[train_index]
    test_data=ratings.iloc[test_index]
    user_item_matrix = pd.pivot_table(train_data, values='rating', index='userId', columns='movieId')
    similarity_matrix = cosine_similarity(user_item_matrix.fillna(0),user_item_matrix.fillna(0))
    sim_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    def predict_rating(user_id, movie_id):
      user_id=int(user_id)
      movie_id=int(movie_id)
      if movie_id not in user_item_matrix.columns:
        return 0
      else:
        total_sims=0
        total_ratings=0
        for other_user_id,row in sim_df.iterrows():
          other_user_id=int(other_user_id)
          if other_user_id!=user_id:
            rating=user_item_matrix[movie_id][other_user_id]
            sim=sim_df[other_user_id][user_id]
            if pd.isna(rating) or sim<i:
              continue
            total_ratings+=(sim_df[user_id][other_user_id]*rating)
            total_sims+=sim_df[user_id][other_user_id]
        if total_sims==0:
          return 0
        else:
          return total_ratings/total_sims
    y1_true = []
    y1_pred = []
    for _, row in test_data.iterrows():
        user = row['userId']
        movie = row['movieId']
        actual_value = row['rating']
        predict = predict_rating(user, movie)
        if predict!=0:
          y1_true.append(actual_value)
          y1_pred.append(predict)
    if y1_pred:
      mse=mean_squared_error(y1_true,y1_pred)
      rmse=np.sqrt(mse)
      rmses.append(rmse)
  if len(rmses)!=0:
    avg_rmse=sum(rmses)/len(rmses)
  else:
    avg_rmse=0
  result.append({'threshold': i, 'rmse': avg_rmse})

