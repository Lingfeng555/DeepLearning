import torch
import torch.nn as nn
import torch.nn.functional as F

class UserDataMLP (nn.Module):
    
    layers: nn.ModuleList
    
    def __init__(self, input_size: int, output_size: int, factor : int):
        super(UserDataMLP, self).__init__()
        
        hidden_layers = [input_size]
        
        while input_size - factor > output_size:
            hidden_layers.append(input_size - factor)
            factor += factor
        hidden_layers.append(output_size)
        
        self.layers = nn.ModuleList([
            nn.Linear(hidden_layers[i],hidden_layers[i+1]) 
            for i in range(len(hidden_layers)-1)
            ])
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
    
# Ejemplo de prueba
if __name__ == '__main__':
    # Definición de parámetros
    input_size = 16    # Por ejemplo, 16 características de entrada
    output_size = 4    # Tamaño deseado de salida (no necesariamente se respeta en la arquitectura actual)
    factor = 2         # Factor de decremento

    # Crear datos de entrada de ejemplo (tensor de 16 elementos)
    user_data = torch.randn(input_size)
    
    # Instanciar el modelo con el tamaño de entrada obtenido de los datos
    model = UserDataMLP(input_size=user_data.size()[0], output_size=output_size, factor=factor)
    
    # Imprimir la arquitectura del modelo
    print("Arquitectura del modelo:")
    print(model)
    
    # Realizar un pase forward con los datos de entrada
    output = model(user_data)
    
    # Mostrar entrada y salida
    print("\nEntrada:")
    print(user_data)
    print("\nSalida:")
    print(output)
