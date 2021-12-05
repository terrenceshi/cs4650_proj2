import torch.nn as nn
import torch.nn.functional as F
import torch

class Net2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(Net2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True, nonlinearity='relu')

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        #h0 = h0.long()

        x = x.view(1, 1, -1)

        output, hn = self.rnn(x, h0)

        #print('\ntest:', output.shape)
        #print('test:', output)

        output = self.fc(output[:, -1, :])
        
        return output
