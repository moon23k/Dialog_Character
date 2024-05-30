import json, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence




class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = self.load_data(split)


    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        hist = []
        elem = self.data[idx]

        for uttr in elem['hist']:
            hist.extend(self.tokenizer.encode(uttr).ids)

        x = self.tokenizer.encode(elem['x']).ids
        y = self.tokenizer.encode(elem['y']).ids

        return torch.LongTensor(hist), torch.LongTensor(x), torch.LongTensor(y)




class Collator(object):
    
    def __init__(self, pad_id):
        self.padding_args = {'batch_first': True, 'padding_value': pad_id}        


    def __call__(self, batch):
        h_batch, x_batch, y_batch = zip(*batch)

        return {'hist': pad_sequence(h_batch, **self.padding_args),
                'x': pad_sequence(x_batch, **self.padding_args),
                'y': pad_sequence(y_batch, **self.padding_args)}




def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(tokenizer, split),
        batch_size=config.batch_size, 
        shuffle=True if split == 'train' else False,
        collate_fn=Collator(config.pad_id),
        num_workers=2
    )
    