import pandas as pd
from typing import Literal
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
class MovieLensDataset(Dataset):
    users: dict
    ratings: pd.DataFrame
    users_min_number_of_ratings: pd.DataFrame
    split : str
    transpose_ratings : bool
    def __init__(self, ml_path: str, split: Literal["train", "val", "test"], transpose_ratings: bool, seed: int):
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
        self.transpose_ratings = transpose_ratings
        
    def __len__(self): return len(self.users[self.split])
    
    def __getitem__(self, index):
        user = self.users[self.split].iloc[index].astype(int)
        user_id = user["user id"]
        user_data = user.drop("user id")
        
        rating_hist = self.ratings[self.ratings["user_id"] == user_id].drop(columns=["user_id"]).sort_values(by='timestamp', ascending=True)
        rating_hist["timestamp"] = self.scaler.fit_transform(rating_hist[["timestamp"]])
        rating_hist["timestamp"] = np.around(rating_hist[["timestamp"]] * 100)
        
        rating_hist["years_since_review"] = self.scaler.fit_transform(rating_hist[["years_since_review"]])
        rating_hist["years_since_review"] = np.around(rating_hist["years_since_review"]*100)
        
        rating_hist["rating"] = np.around(self.scaler.fit_transform(rating_hist[["rating"]]) * 5)
        
        rating_train = rating_hist.head(self.users_min_number_of_ratings).fillna(0)
        rating_test = rating_hist.tail(len(rating_hist)- self.users_min_number_of_ratings)
        
        cols_to_multiply = rating_test.columns.difference(["timestamp", "rating", "years_since_review"])
        rating_test = rating_test.loc[:, cols_to_multiply].mul(rating_test["rating"], axis=0).astype('float').sum()
    
        user_data_tensor = torch.tensor(user_data.values, dtype=torch.int32)
        rating_train_tensor = torch.tensor(rating_train.values, dtype=torch.int32)
        rating_test_tensor = torch.tensor(rating_test.values, dtype=torch.float)
        
        min_val = rating_test_tensor.min()
        max_val = rating_test_tensor.max()
        
        divider = (max_val - min_val) if (max_val - min_val) != 0 else 1
        
        rating_test_tensor = (rating_test_tensor - min_val) / divider
        
        #print(rating_hist)
        # Comprobación de NaNs y valores negativos en user_data_tensor
        if torch.isnan(user_data_tensor).any():
            print("Found NaNs in user_data_tensor")
        if (user_data_tensor < 0).any():
            print("Found negative values in user_data_tensor")

        # Comprobación de NaNs y valores negativos en rating_train_tensor
        if torch.isnan(rating_train_tensor).any():
            print("Found NaNs in rating_train_tensor")
        if (rating_train_tensor < 0).any():
            print("Found negative values in rating_train_tensor")
            negative_columns = []

            for column in rating_train.columns:
                if (rating_train[column] < 0).any():
                    negative_columns.append(column)
                    print(f"En la columna '{column}' se encontraron valores negativos.")

        # Comprobación de NaNs y valores negativos en rating_test_tensor
        if torch.isnan(rating_test_tensor).any():
            print("Found NaNs in rating_test_tensor")
        if (rating_test_tensor < 0).any():
            print("Found negative values in rating_test_tensor")
    
        return user_data_tensor, rating_train_tensor.t() if self.transpose_ratings else rating_train_tensor, rating_test_tensor

if __name__ == '__main__':
    val_dataset = MovieLensDataset(ml_path="submission2/ml-100k", split="val", transpose_ratings=True, seed=55)
    user_data_tensor, rating_train_tensor, rating_test_tensor = val_dataset.__getitem__(1)

    print("User Data Tensor Size:", user_data_tensor.size())
    print("Rating Train Tensor Size:", rating_train_tensor.size())
    print("Rating Test Tensor Size:", rating_test_tensor.size())