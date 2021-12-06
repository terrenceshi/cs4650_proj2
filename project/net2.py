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

    def forward(self, x, hidden):

        #h0 = h0.long()

        x = x.view(1, 1, -1)

        print('\ninput:', x.shape)
        print('x:',x)

        output, hn = self.rnn(x, hidden)

        #print('after rnn:',output)

        print('\ntest (after rnn):', output.shape)
        print('test:', output)

        output = self.fc(output[:, -1, :])

        print('\nafter linear:',output)

        #m = nn.Softmax(dim=1)
        #output = m(output)
        
        return output, hn

    def init_hidden(self, xLen):
        #return torch.zeros(self.n_layers, self.input_size, self.hidden_size)

        return torch.zeros(self.n_layers, xLen, self.hidden_size)
