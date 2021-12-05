
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

# Create Dataloader
class Custom_Dataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):

        input = self.data[index][1]

        target = self.data[index][0]

        return {
            'input': input,
            'target': target
        }


def create_data_loader(data, batchSize, shuffle):
    ds = Custom_Dataset(data = data)

    # batch_size 1 for this project
    return DataLoader(ds, batch_size=batchSize, shuffle=shuffle)

def split_train_val(data, props=[.8, .1, .1]):

    random.shuffle(data)

    length = len(data)

    firstSplit = int(length * props[0])

    secondSplit = int(length * props[1]) + firstSplit

    #print('firstSplit:',firstSplit)
    #print('secondSplit:',secondSplit)

    trainData = data[0 : firstSplit]
    valData = data[firstSplit : secondSplit]
    testData = data[secondSplit : length]

    return trainData, valData, testData


#good function if you want to put a space between author names. since author names are written FirstnameLastname in the folders
#used to need these functions but atm dont need them
def getUpperIndices(s):
    return [i for i, c in enumerate(s) if c.isupper()]

def addASpace(s):
    upIdx = getUpperIndices(s)[1]
    s = s[:upIdx] + ' ' + s[upIdx:] #add space between author's first and last name