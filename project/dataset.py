
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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

def split_train_val(data, props=[.8, .2]):
    assert round(sum(props), 2) == 1 and len(props) == 2
    # return values
    train_df, val_df = None, None

    ## YOUR CODE STARTS HERE (~6-10 lines of code)
    # hint: you can use df.iloc to slice into specific indexes
    size = df.shape[0]
    firstNum = int(size * props[0])
    train_df, val_df = df.iloc[0: firstNum], df.iloc[firstNum: size]
    ## YOUR CODE ENDS HERE ##

    return train_df, val_df