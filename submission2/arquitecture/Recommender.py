import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.PatterAnalyzer import PatternAnalyzer
from .components.VeryBasicMLP import VeryBasicMLP

class Recommender(nn.Module):
    
    user_data_analyzer : VeryBasicMLP
    pattern_analyzer : PatternAnalyzer
    final_regressor : VeryBasicMLP
    
    def __init__(self, 
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
        #print("User analyzer output: ", user_output)
        ratings_patters_information = self.pattern_analyzer(ratings_matrix)
        #print("ratings_patters_information output: ", ratings_patters_information)
        concatenated = torch.cat([user_output, ratings_patters_information], dim=1)
        
        ret = self.final_regressor(concatenated)
        
        return ret
    
if __name__ == "__main__":
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
        