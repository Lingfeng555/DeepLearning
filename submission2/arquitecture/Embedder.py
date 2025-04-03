import torch
import torch.nn as nn

from .components.VeryBasicLSTM import LSTMNet
from .components.VeryBasicMLP import VeryBasicMLP

class RatingsEmbedder(nn.Module):
    
    embedding_block : nn.ModuleList
    
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 num_ratings: int, 
                 lstm_hidden_size : int, 
                 lstm_num_layers: int, 
                 word_size: int,
                 final_mlp_factor: int,
                 output_size: int
                 ):
        super(RatingsEmbedder, self).__init__()
        
        self.embedding_block = nn.ModuleList()
        for i in range(num_ratings):
            self.embedding_block.append(
                nn.Sequential(
                    nn.Embedding(num_embeddings, embedding_dim),
                    LSTMNet(input_size=embedding_dim,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            output_size=word_size
                            )
                )
            )
        
        self.final_mlp = VeryBasicMLP(input_size=num_ratings*word_size, 
                                      output_size=output_size,
                                      factor=final_mlp_factor
                                      )     
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_ratings, rating_size = x.size()
        x = torch.stack( [
            self.embedding_block[i]( x[:, i ,:] ) for i in range(num_ratings)
        ], dim=1 )
        x = x.flatten(start_dim=1)
        return self.final_mlp(x)
    
    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())

class UserEmbedder (nn.Module):
    
    user_embedding : nn.Embedding
    user_analyzer : VeryBasicMLP
    
    def __init__(self, num_embeddings: int, embedding_dim: int, input_dim: int, output_size : int, factor : int):
        super(UserEmbedder, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings = num_embeddings, 
                                           embedding_dim = embedding_dim)
        self.user_analyzer = VeryBasicMLP(input_size = input_dim*embedding_dim,
                                          output_size = output_size,
                                          factor  = factor)
    
    def forward(self, x):
        x = self.user_embedding(x)
        x = self.user_analyzer(torch.flatten(x, start_dim=1))
        return x
    
    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())
        
        
if __name__ == '__main__':
    
    num_embeddings = 100
    embedding_dim = 16
    num_ratings = 22
    lstm_hidden_size = 32
    lstm_num_layers = 8
    word_size = 8
    final_mlp_factor = 2
    embedding_output = 19
    
    model = RatingsEmbedder(num_embeddings = num_embeddings, 
                     embedding_dim = embedding_dim, 
                     num_ratings = num_ratings,
                     lstm_hidden_size=lstm_hidden_size,
                     lstm_num_layers=lstm_num_layers,
                     word_size=word_size,
                     final_mlp_factor = final_mlp_factor,
                     output_size=embedding_output
                     )
    
    print("Test ratings Embedder")
    sample_ratings = torch.randint(low=12, high=num_embeddings, size=(10,22,19), dtype=torch.long)
    print("Input ratings:", sample_ratings.size())
    embedding_output = model(sample_ratings)
    print("Output Indices:", embedding_output.size())
    print(f"Model parameters: {model.n_parameters()}")
    
    print()
    
    print("Test User Embedder")
    user_embedder = UserEmbedder( num_embeddings=100, 
                                 embedding_dim=10,
                                 input_dim=23,
                                 output_size=24,
                                 factor=2
                                 )
    sample_user_data = torch.randint(low=12, high=num_embeddings, size=(10,23), dtype=torch.long)
    print("Input ratings:", sample_user_data.size())
    embedding_output = user_embedder(sample_user_data)
    print("Output Indices:", embedding_output.size())
    print(f"Model parameters: {user_embedder.n_parameters()}")