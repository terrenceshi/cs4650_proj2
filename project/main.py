from dataloading import load_data, split_test_and_train
from rnn import  AttentionRNN
from torch import optim
from torch.nn import NLLLoss
from matplotlib.pyplot import plot
import sys
from sklearn.naive_bayes import MultinomialNB


if __name__ == '__main__':
    path = 'project/Data/C50'
    df = load_data(path, 'padded')
    train_x, val_x, test_x, train_y, val_y, test_y = split_test_and_train(df)

    if sys.argv[1] == 'RNN':
        input_size = 1
        output_size = 50
        sequence_length = train_x.shape[1]
        batch_size = 16
        model = AttentionRNN(input_size=input_size, output_size=output_size, batch_size=batch_size)
        hidden = model.initHidden()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        criterion = NLLLoss()
        from rnn import train, predict

        losses = train(model, train_x, train_y, optimizer, criterion, 1)
        plot(losses.detach().numpy)
    elif sys.argv[1] == 'NB':
        model = MultinomialNB()
        from naivebayes import train, predict
        train(model, train_x, train_y)
    out = predict(model, test_x)
    print(out, test_y)
    # print([a == b for a, b in zip(out, test_y)])
    print(sum([a == b for a, b in zip(out, test_y)]))
