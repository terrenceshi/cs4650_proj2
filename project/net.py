import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
		#basically copy pasted code from hw3
	
        #Input input: torch Tensor of shape (1,)
        #hidden: torch Tensor of shape (self.n_layers, 1, self.hidden_size)
        #Return output: torch Tensor of shape (1, self.output_size) 
        #and hidden: torch Tensor of shape (self.n_layers, 1, self.hidden_size)

        #print('test:', x.shape)
        #print('test:', x)

        x = x.view(1, 1, -1)

        output = self.encoder(x) #.view(1, 1, -1)


        #print('\ntest:',output[0].shape)
        #print('test:', output)

        output = torch.squeeze(output, dim = 0)

        output, hidden = self.gru(output, hidden)

        #print('\ntest:',torch.squeeze(output).shape)
        #print('test:', output)

        #output = self.decoder(output[0])
        output = self.decoder(torch.squeeze(output, dim = 0))
        
        #print('output shape:',output.size())
        #print('hidden shape:',hidden.size())

        output = self.softmax(output)
        
        return output, hidden


    def init_hidden(self, xLen):
        #return torch.zeros(self.n_layers, self.input_size, self.hidden_size)

        return torch.zeros(self.n_layers, xLen, self.hidden_size)