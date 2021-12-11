import torch
from torch.nn import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyRNN(Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Linear = Linear(input_size + hidden_size, hidden_size)
        self.Softmax = LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = torch.cat((input, hidden), dim=1)
        hidden = self.Linear(input)
        output = self.Linear(input)
        output = self.Softmax(output)
        return output, hidden

    def initHidden(self):
        # Initialize hidden layer to 1 x batch size x hidden size
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)
