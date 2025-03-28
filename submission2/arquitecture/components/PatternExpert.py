import torch
import torch.nn as nn
import torch.nn.functional as F
class PatternExpert(nn.Module):
    def __init__(self, input_size, hidden_size, fc_output_size):
        super(PatternExpert, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bi_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, fc_output_size)

    def forward(self, x, attention):
        
        #print("Patata",x)
        
        out, _ = self.lstm(x)
        out, _ = self.bi_lstm(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out * attention
    

if __name__ == "__main__":
    batch_size = 8
    seq_len = 10   # longitud de la secuencia
    n = 20         # dimensión del vector de entrada
    hidden_size = 32
    fc_output_size = 5

    # Crear datos de ejemplo
    x = torch.randn(batch_size, seq_len, n)
    modelo = PatternExpert(input_size=n, hidden_size=hidden_size, fc_output_size=fc_output_size)
    
    # Realizar la pasada hacia adelante
    salida = modelo(x, 1)
    print("Tamaño de la salida:", salida.shape)  # Debería ser (batch_size, fc_output_size)