import torch
import torch.nn as nn

from .CNNBlock import CNNBlock
from .PatternExpert import PatternExpert
from .AttentionBlock import AttentionBlock
from .VeryBasicMLP import VeryBasicMLP

class PatternAnalyzer(nn.Module):
    
    cnn_block : CNNBlock
    experts : nn.ModuleList
    attention_block : AttentionBlock
    wighted_sum : VeryBasicMLP
    output_size : int
    
    def __init__(self, 
                 input_size: torch.Size, 
                 conv_structure : list, 
                 pool_depth: int, 
                 expert_hidden_size: int, 
                 expert_output_len: int,
                 final_mlp_factor: int,
                 final_mlp_output_len: int, 
                 ):
        super(PatternAnalyzer, self).__init__()
        
        self.cnn_block = CNNBlock(
                                    feature= conv_structure,
                                    height=input_size[0], 
                                    width=input_size[1],
                                    pool_depth = pool_depth
                                   )
        
        self.experts = nn.ModuleList()
        for _ in range(conv_structure[-1]):
            self.experts.append(
                PatternExpert(
                    input_size=self.cnn_block.out_put_size['width'],
                    hidden_size=expert_hidden_size,
                    fc_output_size=expert_output_len
                    )
            )
        
        self.attention_block = AttentionBlock(
            num_features=self.cnn_block.out_put_size['features'],
            attention_value=1,
            height=self.cnn_block.out_put_size['height'],
            width=self.cnn_block.out_put_size['width']
                                        )
        
        self.wighted_sum = VeryBasicMLP(input_size=self.cnn_block.out_put_size['features']*expert_output_len, 
                                        output_size=final_mlp_output_len, 
                                        factor=final_mlp_factor)
        
        self.output_size = final_mlp_output_len
        
    def n_parameters(self) -> int: return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = x.unsqueeze(1)
        print(x)
        features = self.cnn_block(x)
        print(features)
        attention_values = self.attention_block(features)
        print(features)
        #print("features[:, 1, :, :] :", features[:, 1, :, :])
        #print("Attention.shape (batch_size, n_features, ,attention value) :", attention_values.shape)
        x = torch.stack([self.experts[i](features[:, i, :, :], attention_values[:, i, :]) for i in range(len(self.experts))], dim=1)
        #print("patata",x)
        x = x.flatten(start_dim=1)
        
        x = self.wighted_sum(x)
        return x
    
if __name__ == "__main__":
    height=19
    width=22
    # Crear instancia del modelo
    model = PatternAnalyzer(conv_structure=[1,4,8,16,32,32,32],
                            input_size=torch.Size((height, width)),
                            pool_depth=3,
                            expert_hidden_size=32,
                            expert_output_len=8,
                            final_mlp_factor=2,
                            final_mlp_output_len=10,
                            )
    input_tensor = torch.randn(2, height, width)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")