import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=False
                            )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_output = lstm_out[:, -1, :]
        output = self.fc(final_output)
        return output

if __name__ == '__main__':
    batch_size = 32
    sequence_length = 10
    input_size = 20
    hidden_size = 50
    num_layers = 2
    output_size = 5

    model = LSTMNet(input_size, hidden_size, num_layers, output_size)
    sample_data = torch.randn(batch_size, sequence_length, input_size)
    print("Input shape:",sample_data.size())
    output = model(sample_data)
    
    print("Output shape:", output.shape)