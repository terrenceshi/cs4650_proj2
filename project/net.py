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

    def forward(self, x, hidden):
		#basically copy pasted code from hw3
	
        #Input input: torch Tensor of shape (1,)
        #hidden: torch Tensor of shape (self.n_layers, 1, self.hidden_size)
        #Return output: torch Tensor of shape (1, self.output_size) 
        #and hidden: torch Tensor of shape (self.n_layers, 1, self.hidden_size)
        
        #print('input:',input.size()) #input is 1 number, has no dimensions
        #print('hidden:',hidden)
                #print('output shape:',output.size()) #is 100 numbers
        #print('output:',output)

        print('test:', x.shape)
        print('test:', x)

        output = self.encoder(x) #.view(1, 1, -1)

        #print('\ntest:',self.encoder(x).shape)
        #print('test:', self.encoder(x)[0])

        print('\ntest:',output.shape)
        print('test:', output)
        #print('test:', output[0][0][553983])

        output, hidden = self.gru(output, hidden)
        
        output = self.decoder(output[0])
        
        #print('output shape:',output.size())
        #print('hidden shape:',hidden.size())
        
        return output, hidden


    def init_hidden(self):
        return torch.zeros(self.n_layers, self.input_size, self.hidden_size)