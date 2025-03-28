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
from arquitecture.Recommender import Recommender
from arquitecture.components.PatterAnalyzer import PatternAnalyzer
import torch.optim as optim

SEED = 55
BATCH = 2000
NUN_THREADS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCH = 400
LEARNING_RATE = 0.001

# %%
train_dataset = MovieLensDataset(ml_path="submission2/ml-100k", split="train", seed=SEED)
test_dataset = MovieLensDataset(ml_path="submission2/ml-100k", split="test", seed=SEED)
val_dataset = MovieLensDataset(ml_path="submission2/ml-100k", split="val", seed=SEED)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=NUN_THREADS)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=True, num_workers=NUN_THREADS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True, num_workers=NUN_THREADS)

# %%
height=19
width=22
user_input_size = 23
    # Crear instancia del modelo
pattern_analyzer = PatternAnalyzer(conv_structure=[1,4,8,16,32,32,32],
                            input_size=torch.Size((height, width)),
                            pool_depth=3,
                            expert_hidden_size=16,
                            expert_output_len=4,
                            final_mlp_factor=2,
                            final_mlp_output_len=15,
                            ).to(DEVICE)
    
model = Recommender(
        user_data_input_size=user_input_size,
        user_data_analizer_factor = 2,
        user_data_analizer_output_size = 15,
        pattern_analyzer=pattern_analyzer,
        final_regressor_factor=1,
        final_regressor_output_len=19
    ).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

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

        outputs = model(user_data_tensor.to(DEVICE), rating_train_tensor.to(DEVICE)).to(DEVICE)        
        loss = criterion(outputs.to(DEVICE), rating_test_tensor.to(DEVICE)) 
        
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update model parameters
        running_train_loss += loss.item() * rating_test_tensor.size()[1]
    
    epoch_train_loss = running_train_loss / len(train_dataloader.dataset)

    # --- Validation Phase ---
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():
        for user_data_tensor, rating_train_tensor, rating_test_tensor in val_dataloader:
            
            outputs = model(user_data_tensor.to(DEVICE), rating_train_tensor.to(DEVICE))        
            loss = criterion(outputs, rating_test_tensor.to(DEVICE)) 
            
            running_val_loss += loss.item() * rating_test_tensor.size()[1]
    
    epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCH}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")



