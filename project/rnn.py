import torch
from torch.nn import *
from matplotlib.pyplot import plot
from torch import optim
from torch.utils.data import Dataset, DataLoader
from dataloading import load_data, split_test_and_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionRNN(Module):
    def __init__(self, input_size, output_size, num_heads=1, embed_size=1, dropout=0.1, batch_size=1, hidden_size=128):
        super().__init__()
        #Store parameters
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        #initialize layers
        self.Attention = MultiheadAttention(embed_size, num_heads, dropout)
        self.RNN = RNN(input_size=input_size, hidden_size=hidden_size)
        self.Linear = Linear(hidden_size, output_size)
        self.Softmax = Softmax(dim=0)

    def forward(self, input, hidden, labels=None):
        #Attention layer
        if labels:
            input = self.Attention(input, labels, input)
        #run already embedded input through RNN
        output, new_hidden = self.RNN(input, hidden)
        #take output of last item in the sequence
        output = output[-1, :, :]
        #convert hidden size to output size
        output = self.Linear(output)
        #remove extra dimension left from taking last item in sequence
        output = output.squeeze()
        #Softmax linear outputs to get probability
        output = self.Softmax(output)
        #return output and new hidden state
        return output, new_hidden

    def initHidden(self):
        # Initialize hidden layer to 1 x batch size x hidden size
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)


def train_helper(model, sequence, label, optimizer, criterion):
    # for sequence, label in zip(train_x, train_y):
    hidden = model.initHidden()
    sequence = torch.tensor(sequence, dtype=torch.float)
    output, hidden = model(sequence, hidden)
    label = torch.tensor(label, dtype=int)
    loss = criterion(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss


def train(model, train_x, train_y, optimizer, criterion, epochs):
    train_x = train_x.reshape(-1, train_x.shape[1], model.batch_size, 1)
    train_y = train_y.reshape(-1, model.batch_size)
    losses = []
    for j in range(epochs):
        current_loss = 0
        plot_steps, print_steps = 20, 20
        n_iters = train_x.shape[0]
        for i, (sequence, label) in enumerate(zip(train_x, train_y)):
            label = [x-4 for x in label]
            output, loss = train_helper(model, sequence, label, optimizer, criterion)
            current_loss += loss
            if i % plot_steps == 0:
                losses.append(current_loss / plot_steps)
                current_loss = 0
                guess = torch.argmax(output, dim=1)
                label = torch.tensor(label)
                correct = torch.eq(guess, label)
                percent = sum(correct)/len(correct)
                correct = f"{percent} CORRECT"
                print(f"{i}, {i / n_iters * 100:.1f} {loss:.4f} {guess} {label} {correct}")
        # print(f'epoch: {i}, loss:{loss}')
    return losses


def predict(model, test_x):
    return model.predict(test_x)





