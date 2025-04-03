import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.PatterAnalyzer import PatternAnalyzer
from .components.VeryBasicMLP import VeryBasicMLP
from .Embedder import RatingsEmbedder, UserEmbedder

class Recommender(nn.Module):
    
    user_data_analyzer : VeryBasicMLP
    pattern_analyzer : PatternAnalyzer
    final_regressor : VeryBasicMLP
    
    def __init__(
                self, 
                user_data_input_size: int, 
                user_data_analizer_factor: int,
                user_data_analizer_output_size: int,
                pattern_analyzer: PatternAnalyzer,
                final_regressor_factor: int, 
                final_regressor_output_len: int
                ):
        super().__init__()
        
        self.user_data_analyzer = VeryBasicMLP(input_size=user_data_input_size,
                                               output_size=user_data_analizer_output_size,
                                               factor=user_data_analizer_factor
                                               )
        self.pattern_analyzer = pattern_analyzer
        
        self.final_regressor = VeryBasicMLP(
            input_size=user_data_analizer_output_size + self.pattern_analyzer.output_size, 
            output_size=final_regressor_output_len,
            factor=final_regressor_factor
            )
    
    def forward(self, user_data, ratings_matrix):
        
        user_output = self.user_data_analyzer(user_data)
        #print("User outputs have NA", torch.isnan(user_output).any(), user_output.size(), user_output.dtype, user_output.requires_grad)
        
        ratings_patters_information = self.pattern_analyzer(ratings_matrix)
        #print("Rating patterns have NA", torch.isnan(ratings_patters_information).any(), ratings_patters_information.size(), ratings_patters_information.dtype, ratings_patters_information.requires_grad)
        
        concatenated = torch.concat([user_output, ratings_patters_information], dim=-1)
        #print("Concatenated have NA", torch.isnan(concatenated).any(), concatenated.size())

        ret = self.final_regressor(concatenated)
        #print("Regression have NA", torch.isnan(ret).any(), ret.size())
        return ret
    
class Recommender_2 (nn.Module):
    
    ratings_embedder: RatingsEmbedder
    user_embedder: UserEmbedder
    experts: nn.ModuleList
    
    def __init__(self,                  
                 ratings_num_embeddings: int, 
                 ratings_embedding_dim: int, 
                 ratings_num_ratings: int, 
                 ratings_lstm_hidden_size : int, 
                 ratings_lstm_num_layers: int, 
                 ratings_word_size: int,
                 ratings_final_mlp_factor: int,
                 ratings_embedding_output: int ,
                 user_num_embeddings: int,
                 user_embedding_dim: int,
                 user_embedding_output:int,
                 user_data_input_dim: int,
                 user_factor : int,
                 final_output_size: int,
                 expert_factor: int
                 ):
        super(Recommender_2, self).__init__()
        
        self.ratings_embedder = RatingsEmbedder(
                     num_embeddings = ratings_num_embeddings, 
                     embedding_dim = ratings_embedding_dim, 
                     num_ratings = ratings_num_ratings,
                     lstm_hidden_size= ratings_lstm_hidden_size,
                     lstm_num_layers= ratings_lstm_num_layers,
                     word_size= ratings_word_size,
                     final_mlp_factor = ratings_final_mlp_factor,
                     output_size= ratings_embedding_output   
        )
        
        self.user_embedder = UserEmbedder(
            num_embeddings=user_num_embeddings, 
            embedding_dim=user_embedding_dim,
            input_dim=user_data_input_dim,
            output_size=user_embedding_output,
            factor=user_factor
            )
        
        self.experts = nn.ModuleList()
        for i in range(final_output_size):
            self.experts.append(
                VeryBasicMLP(input_size=ratings_embedding_output+user_embedding_output,
                             output_size=1, factor=expert_factor)
            )
            
    def forward(self, ratings_tensor, user_tensor):
        ratings_embedding = self.ratings_embedder(ratings_tensor)
        user_embedding = self.user_embedder(user_tensor)
        x = torch.concat([user_embedding, ratings_embedding], dim=-1)
        x = torch.stack( [ expert(x) for expert in self.experts ], dim=1)
        x = torch.flatten(x, start_dim=1)
        return x
     
    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())

def test_recommender():
    height=19
    width=22
    user_input_size = 23
    # Crear instancia del modelo
    pattern_analyzer = PatternAnalyzer(conv_structure=[1,4,8,16,32,32,32],
                            input_size=torch.Size((height, width)),
                            pool_depth=3,
                            expert_hidden_size=32,
                            expert_output_len=8,
                            final_mlp_factor=2,
                            final_mlp_output_len=10,
                            )
    
    model = Recommender(
        user_data_input_size=user_input_size,
        user_data_analizer_factor = 2,
        user_data_analizer_output_size = 10,
        pattern_analyzer=pattern_analyzer,
        final_regressor_factor=2,
        final_regressor_output_len=19
    )
    ratings = torch.randn(628, height, width)  
    user_data = torch.randn(628, user_input_size)
    

    # Pasar el tensor por el modelo
    output = model(user_data, ratings)
    print(output)
    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

def test_recommender_2():
    model = Recommender_2(ratings_num_embeddings = 100, 
                 ratings_embedding_dim = 16, 
                 ratings_num_ratings = 22, #Fixed
                 ratings_lstm_hidden_size  = 32 , 
                 ratings_lstm_num_layers = 16, 
                 ratings_word_size = 8,
                 ratings_final_mlp_factor = 16,
                 ratings_embedding_output = 40 ,
                 user_num_embeddings = 100,
                 user_embedding_dim = 16,
                 user_embedding_output = 15,
                 user_data_input_dim = 23, #Fixed
                 user_factor = 16,
                 final_output_size = 19, #Fixed
                 expert_factor = 6
                 )
    ratings_tensor = torch.randint(0, 100, size=(678, 22, 19))
    user_data_tensor = torch.randint(0, 100, size=(678, 23))

    print("Tamaño de ratings_tensor:", ratings_tensor.size())
    print("Tamaño de user_data_tensor:", user_data_tensor.size())

    output = model(ratings_tensor, user_data_tensor)
    
    print("Tamaño de la salida del modelo:", output.size())
    
if __name__ == "__main__":
    test_recommender_2()