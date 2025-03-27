import pandas as pd
from typing import Literal
import os

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
class MovieLensDataset(Dataset):
    users: dict
    ratings: pd.DataFrame
    users_min_number_of_ratings: pd.DataFrame
    split : str
    def __init__(self, ml_path: str, split: Literal["train", "val", "test"], seed: int):
        super().__init__()
        
        self.scaler = MinMaxScaler()
        
        data = pd.read_csv(os.path.join(ml_path,"u.data"),
                 sep='\t',    
                 header=None,  
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

        data['timestamp'] = (data['timestamp'] - pd.Timestamp("1970-01-01").second )/(3600*24*365)
        
        genre = pd.read_csv(os.path.join(ml_path,"u.genre"),
                 sep='|',    
                 header=None,  
                 names=['genre_name', 'genre_id'])
        
        columns = [
            'movie_id',
            'movie_title',
            'release_date',
            'video_release_date',
            'IMDb_URL'
        ] + genre["genre_name"].tolist()
        
        
        movie = pd.read_csv(
            os.path.join(ml_path,"u.item"),
            sep='|',
            header=None,        
            names=columns,     
            encoding='latin-1'  
        ).drop(columns=['video_release_date', 'IMDb_URL'])
        
        movie["release_date"] = pd.to_datetime(movie["release_date"], errors='coerce') - pd.Timestamp("1970-01-01")
        movie['years_1970'] = movie['release_date'].dt.days / 365
        movie = movie.drop(columns=['release_date', "movie_title"])
        
        users = pd.read_csv(os.path.join(ml_path,"u.user"),
                            sep='|',
                            header=None,        
                            names= "user id | age | gender | occupation | zip code".split(" | "),     
                            encoding='latin-1')
        
        users["gender"] = users["gender"].apply(lambda x: 1 if x == "M" else 0)
        users["occupation"] = users["occupation"]
        users["occupation"] = pd.Categorical(users["occupation"])
        occupation_dummies = pd.get_dummies(users['occupation'], prefix='occupation').astype(int)
        users = pd.concat([users, occupation_dummies], axis=1).drop(columns=["occupation", "zip code"])
        
        ratings = pd.merge(
            data,           
            movie,          
            how='left',     
            left_on='item_id',
            right_on='movie_id'
        )
        ratings.head()
        ratings.drop(columns=["item_id", "movie_id"], inplace=True)
        ratings["years_since_review"] = ratings["timestamp"] - ratings["years_1970"]
        ratings.drop(columns=["years_1970"], inplace=True)
        
        self.ratings = ratings

        self.users_min_number_of_ratings = ratings["user_id"].value_counts().min() - 1
        
        users_train_val, users_test = train_test_split(
            users, 
            test_size=0.2, 
            random_state=seed
        )

        users_train, users_val = train_test_split(
            users_train_val,
            test_size=0.1,
            random_state=seed
        )
        self.users = {}
        self.users["val"] = users_val
        self.users["test"] = users_test
        self.users["train"] = users_train
        
        self.split = split
        
    def __len__(self): return len(self.ratings)
    
    def __getitem__(self, index):
        user = self.users[self.split].iloc[index].astype(int)
        user_id = user["user id"]
        user_data = user.drop("user id")
        
        rating_hist = self.ratings[self.ratings["user_id"] == user_id].drop(columns=["user_id"]).sort_values(by='timestamp', ascending=True)
        rating_hist["timestamp"] = self.scaler.fit_transform(rating_hist[["timestamp"]])
        rating_hist["years_since_review"] = self.scaler.fit_transform(rating_hist[["years_since_review"]])
        rating_hist["rating"] = self.scaler.fit_transform(rating_hist[["rating"]])
        
        rating_train = rating_hist.head(self.users_min_number_of_ratings)
        rating_test = rating_hist.tail(len(rating_hist)- self.users_min_number_of_ratings)
        
        cols_to_multiply = rating_test.columns.difference(["timestamp", "rating", "years_since_review"])
        rating_test = rating_test.loc[:, cols_to_multiply].mul(rating_test["rating"], axis=0).astype('float').sum()
        
        user_data_tensor = torch.tensor(user_data.values)
        rating_train_tensor = torch.tensor(rating_train.values)
        rating_test_tensor = torch.tensor(rating_test.values)
        
        min_val = rating_test_tensor.min()
        max_val = rating_test_tensor.max()
        
        rating_test_tensor = (rating_test_tensor - min_val) / (max_val - min_val)
        
        return user_data_tensor, rating_train_tensor, rating_test_tensor