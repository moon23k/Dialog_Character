import json, torch
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()

        self.character = None
        if config.mode == 'pretrain':
            self.character = config.character
            self.theshold = config.data_threshold

        self.data = self.load_data(split, self.character)


    def load_data(self, split, character=None):
        if self.mode == 'pretrain':
            f_name = f'data/{character}.json'
        else:
            f_name = f"data/{split}.json"    

        with open(f_name, 'r') as f:
            data = json.load(f)

        if self.mode == 'pretrain' and split == 'train':
            return data[:self.threshold]
        else:
            return data[self.threshold:]

        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        uttr = self.data[idx]['uttr']
        resp = self.data[idx]['resp']
        return uttrn, resp



def load_dataloader(config, split):
    return DataLoader(Dataset(config, split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      num_workers=2,
                      pin_memory=True)