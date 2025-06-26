import pandas as pd
# read csv files
ratings=pd.read_csv('ratings.csv')
movies=pd.read_csv('movies.csv')
from sklearn.metrics.pairwise import cosine_similarity
#create user-item matrix
user_item_matrix = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
# Create similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(user_item_matrix.fillna(0), user_item_matrix.fillna(0))
# Create a DataFrame for the similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

from sklearn.model_selection import train_test_split
#split the data into training and testing sets with 20% of data as test data
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
#create a function to predict ratings of unrated movies for a user using the ratings of other users based of their similarity with user
def predicted_ratings(user_id, movie_id):
  user_id=int(user_id)
  movie_id=int(movie_id)
  # id movie_id is not in user_item_matrix columns, return 0
  if movie_id not in user_item_matrix.columns:
    return 0
  else:
    total_sim=0
    total_rating=0
    # itreate through users in similarity_df
    for other_user_id,row in similarity_df.iterrows():
      # check the rating if it is other than user_id
      if other_user_id!=user_id:
        rating=user_item_matrix[movie_id][other_user_id]
        # if rating is NaN, continue
        if pd.isna(rating):
            continue
        # add it to total_rating and total_sim as shown
        total_rating+=(similarity_df[user_id][other_user_id]*rating)
        total_sim+=similarity_df[user_id][other_user_id]
    # if the user has no similar users or no ratings, return 0
    if total_sim==0:
      return 0
    else:
      #else return the total_rating/total_sim
      return total_rating/total_sim
    
y_true = []
y_pred = []
for _, row in test_data.iterrows():
    # get userId, movieId and rating from the row of test_data
    user = row['userId']
    movie = row['movieId']
    actual = row['rating']
    #predict ratings for a particular user and movie
    predicted = predicted_ratings(user, movie)
    # if predicted rating is not 0, append actual and predicted ratings to y_true and y_pred respectively
    if predicted!=0:
      y_true.append(actual)
      y_pred.append(predicted)
from sklearn.metrics import mean_squared_error
import numpy as np
# if y_pred is not empty, calculate root mean squared error
if y_pred:
  mse=mean_squared_error(y_true,y_pred)
  rmse=np.sqrt(mse)
# if rmse is low, it means the model is performing well

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
# Create KFold cross-validation
kf=KFold(n_splits=3,shuffle=True,random_state=42)
# set the number of neighbors as 40 to consider for prediction
k=30
# set the threshold values that will be checked
threshold=[0.15,0.2,0.3,0.5,0.7]
result=[]
#for every threshold value
for i in threshold:
  #create a list to store the rmse values for th particular threshold
  rmses=[]
  # split ratings into train and test sets
  for train_index,test_index in kf.split(ratings):
    # get tarin datausing the indices provided by kfold
    train_data=ratings.iloc[train_index]
    # get test data using the indices provided by kfold
    test_data=ratings.iloc[test_index]
    # create user-item matrix and similarity matrix for the train data
    user_item_matrix = pd.pivot_table(train_data, values='rating', index='userId', columns='movieId')
    similarity_matrix = cosine_similarity(user_item_matrix.fillna(0),user_item_matrix.fillna(0))
    sim_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    # create a function to predict ratings of unrated movies for a user 
    def predict_rating(user_id, movie_id):
      user_id=int(user_id)
      movie_id=int(movie_id)
      if movie_id not in user_item_matrix.columns:
        return 0
      else:
        total_sims=0
        total_ratings=0
        # iterate through every user
        for other_user_id,row in sim_df.iterrows():
          other_user_id=int(other_user_id)
          # if other_user_id is not equal to user_id
          if other_user_id!=user_id:
            # get the rating of the movie by other_user_id
            rating=user_item_matrix[movie_id][other_user_id]
            #and similarity of user_id and other_user_id
            sim=sim_df[other_user_id][user_id]
            # if rating is NaN or similarity is less than threshold, continue
            if pd.isna(rating) or sim<i:
              continue
            #else add the rating and similarity to total_ratings and total_sims respectively
            total_ratings+=(sim_df[user_id][other_user_id]*rating)
            total_sims+=sim_df[user_id][other_user_id]
        # if total_sims is 0, return 0
        if total_sims==0:
          return 0
        # else return total_ratings/total_sims
        else:
          return total_ratings/total_sims
    y1_true = []
    y1_pred = []
    # iterate through the test data
    for _, row in test_data.iterrows():
        # get userId, movieId and rating from the row of test_data
        user = row['userId']
        movie = row['movieId']
        actual_value = row['rating']
        #predict ratings for a particular user and movie
        predict = predict_rating(user, movie)
        # if predicted rating is not 0, append actual and predicted ratings to y1_true and y1_pred respectively
        if predict!=0:
          y1_true.append(actual_value)
          y1_pred.append(predict)
    # if y1_pred is not empty, calculate root mean squared error
    if y1_pred:
      mse=mean_squared_error(y1_true,y1_pred)
      rmse=np.sqrt(mse)
      rmses.append(rmse)
  # if rmses are not empty the calculate the average rmse for the threshold
  if len(rmses)!=0:
    avg_rmse=sum(rmses)/len(rmses)
  else:
    # if rmses are empty, set avg_rmse to 0
    avg_rmse=0
  # append the threshold and avg_rmse to the result list
  # by this we can compare the performance of the model with different threshold values
  result.append({'threshold': i, 'rmse': avg_rmse})


