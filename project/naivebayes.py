import numpy as np
from sklearn.naive_bayes import MultinomialNB
import os
import pickle
import pandas as pd
import re


def load_data(filepath):
    if os.path.exists(os.getcwd() + '/loaded.pkl'):
        with open(os.getcwd() + '/loaded.pkl', 'rb') as file:
            df = pickle.load(file)
    else:
        df = pd.DataFrame(data=None, index=None, columns=['author', 'content'])
        for author in os.listdir(filepath):
            for title in os.listdir(filepath + f'/{author}'):
                file = filepath + f'/{author}/{title}'
                f = open(file, 'r')
                content = ''
                for line in f.readlines():
                    content += 'SOS ' + line[:-2] + f' {line[-2]}' + ' EOS '
                f.close()
                d = {'author': author, 'content': content}
                df = df.append(d, ignore_index=True)
        with open(os.getcwd() + '/loaded.pkl', 'wb') as file:
            pickle.dump(df, file)
    return df


def split_test_and_train(data, props=(0.8, 0.1, 0.1), shuffle=True):
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    x, y = data['content'], data['author']
    x = np.array(list(x))
    y = np.array(y, dtype=int)
    size = data.shape[0]
    train_index = round(size * props[0])
    val_index = train_index + round(size * props[1])
    train_x, val_x, test_x = x[:train_index, :], x[train_index:val_index, :], x[val_index:, :]
    train_y, val_y, test_y = y[:train_index], y[train_index:val_index], y[val_index:]
    return train_x, val_x, test_x, train_y, val_y, test_y


def build_embed(df):
    author_embedding = {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
    content_embedding = {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
    authors, words = 4, 4
    max_length = 0
    for i, row in df.iterrows():
        author, content = row
        if author not in author_embedding:
            author_embedding[author] = authors
            authors += 1
        length = 0
        for line in re.split(r'EOS', content):
            length += 2
            for word in (line + 'EOS').split(' '):
                if word not in content_embedding:
                    content_embedding[word] = words
                    words += 1
                length += 1
            if length > max_length:
                max_length = length
    return author_embedding, content_embedding, max_length, words


def padded_embed(df, max_len, author_embedding, content_embedding):
    for i, row in df.iterrows():
        author, content = row
        if author in author_embedding:
            embedded_author = author_embedding[author]
        else:
            embedded_author = author_embedding['UNK']
        embedded_text = []
        padded_text = [0] * max_len
        for line in re.split(r'EOS', content):
            for word in (line + 'EOS').split(' '):
                if word in content_embedding:
                    embedded_text.append(content_embedding[word])
                else:
                    embedded_text.append(content_embedding['UNK'])
        padded_text[:len(embedded_text)] = embedded_text
        df.iloc[i, 0] = embedded_author
        df.iloc[i, 1] = padded_text


def BOW_embed(df, vocab_size, author_embedding, content_embedding):
    for i, row in df.iterrows():
        author, content = row
        if author in author_embedding:
            embedded_author = author_embedding[author]
        else:
            embedded_author = author_embedding['UNK']
        embedded_text = [0] * vocab_size
        for line in re.split(r'EOS', content):
            for word in (line + 'EOS').split(' '):
                if word in content_embedding:
                    embedded_text[content_embedding[word]] += 1
                else:
                    embedded_text[content_embedding['UNK']] += 1
        df.iloc[i, 0] = embedded_author
        df.iloc[i, 1] = embedded_text


def train(model, train_x, train_y):
    model.fit(train_x, train_y)


def predict(model, test_x):
    return model.predict(test_x)


if os.path.exists(os.getcwd() + '/BOW_embed.pkl'):
    with open(os.getcwd() + '/BOW_embed.pkl', 'rb') as file:
        df = pickle.load(file)
else:
    df = load_data('Data/C50')
    author_embedding, content_embedding, max_length, vocab_size = build_embed(df)
    padded_embed(df, max_length, author_embedding, content_embedding)
    with open(os.getcwd() + '/padded_embed.pkl', 'wb') as file:
        pickle.dump(df, file)
    df = load_data('Data/C50')
    BOW_embed(df, vocab_size, author_embedding, content_embedding)
    with open(os.getcwd() + '/BOW_embed.pkl', 'wb') as file:
        pickle.dump(df, file)

train_x, val_x, test_x, train_y, val_y, test_y = split_test_and_train(df)
model = MultinomialNB()
train(model, train_x, train_y)
pred = predict(model, test_x)
print(pred)
print(test_y)
print(sum([a == b for a, b in zip(pred, test_y)]))



