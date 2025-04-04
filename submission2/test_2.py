# %%
import pandas as pd
from typing import Literal
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.loader import MovieLensDataset
from arquitecture.Recommender import Recommender_2
import torch.optim as optim

SEED = 55
BATCH = 678
NUN_THREADS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"
NUM_EPOCH = 400
LEARNING_RATE = 0.0004


# %%
ml_path = "submission2/ml-100k"
train_dataset = MovieLensDataset(ml_path=ml_path, split="train", transpose_ratings=True, seed=SEED)
test_dataset = MovieLensDataset(ml_path=ml_path, split="test", transpose_ratings=True, seed=SEED)
val_dataset = MovieLensDataset(ml_path=ml_path, split="val", transpose_ratings=True, seed=SEED)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=NUN_THREADS)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=True, num_workers=NUN_THREADS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True, num_workers=NUN_THREADS)

for user_data_tensor, rating_train_tensor, rating_test_tensor in train_dataloader:
    print("Rating Tensor Shape:", rating_train_tensor.size())
    print("User Data Tensor Shape:", user_data_tensor.size())
    print(f"Max index in rating tensor: {rating_train_tensor.max().item()}")
    print(f"Min index in rating tensor: {rating_train_tensor.min().item()}")
    
    # Get true num_ratings from data
    batch_size, num_ratings_actual, rating_size = rating_train_tensor.size()
    user_batch_size, user_data_dim = user_data_tensor.size()
    
    print(f"Actual num_ratings from tensor: {num_ratings_actual}")
    print(f"Actual user_data_input_dim: {user_data_dim}")
    break

# %%
    
model = Recommender_2(ratings_num_embeddings = 101, 
                 ratings_embedding_dim = 30, 
                 ratings_num_ratings = 22, #Fixed
                 ratings_lstm_hidden_size  = 34 , 
                 ratings_lstm_num_layers = 1, 
                 ratings_word_size = 25,
                 ratings_final_mlp_factor = 3,
                 ratings_embedding_output = 36 ,
                 user_num_embeddings = 100,
                 user_embedding_dim = 23,
                 user_embedding_output = 21,
                 user_data_input_dim = 23, #Fixed
                 user_factor = 14,
                 final_output_size = 19, #Fixed
                 expert_factor = 7
                 ).to(DEVICE) 

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

for user_data_tensor, rating_train_tensor, rating_test_tensor in train_dataloader:
    print(rating_train_tensor.size())
    print(user_data_tensor.size())
    print(f"Max index in tensor: {rating_train_tensor.max().item()}")
    model(rating_train_tensor.to(DEVICE), user_data_tensor.to(DEVICE)).to(DEVICE)
    break
# %%
# Crear el optimizador, por ejemplo, usando Adam con una tasa de aprendizaje de 0.001
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Crear la función de pérdida para regresión (Mean Squared Error Loss)
criterion = nn.MSELoss(reduction="sum")

# %%
for epoch in range(NUM_EPOCH):
    # --- Training Phase ---
    model.train()  # Set model to training mode
    running_train_loss = 0.0
    
    for user_data_tensor, rating_train_tensor, rating_test_tensor in train_dataloader:
        optimizer.zero_grad()
        
        outputs = model(rating_train_tensor.to(DEVICE), user_data_tensor.to(DEVICE)).to(DEVICE)        
        loss = criterion(outputs.to(DEVICE), rating_test_tensor.to(DEVICE)) 
        
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update model parameters
        running_train_loss += loss.item() * rating_test_tensor.size()[1]
    
    epoch_train_loss = running_train_loss / len(train_dataloader.dataset)

    # --- Validation Phase ---32
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():
        for user_data_tensor, rating_train_tensor, rating_test_tensor in val_dataloader:
            
            outputs = model(rating_train_tensor.to(DEVICE), user_data_tensor.to(DEVICE))        
            loss = criterion(outputs, rating_test_tensor.to(DEVICE)) 
            
            running_val_loss += loss.item() * rating_test_tensor.size()[1]
    
    epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCH}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")


