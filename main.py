import pandas as pd
import pickle
import os

dfs = []
for i in range(3):
    dfs.append(pd.read_csv(f'articles{i+1}.csv'))

df = pd.concat(dfs, ignore_index=True)

file = open('all_the_news_csv.pkl', 'wb')
pickle.dump(df, file)

cwd = os.getcwd()
for i, row in df.iterrows():
    author = row['author']
    text = row['content']
    title = row['title']
    path = cwd + f'/{author}'
    if not os.path.exists(path):
        os.mkdir(path)
    file = open(path + f'/{title}.txt', 'w')
    file.write(text)
    file.close()


