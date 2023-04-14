import json, torch
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split=None):
        super().__init__()

        self.mode = config.mode
        self.character = None
        
        if self.mode == 'pretrain':
            self.model_type = config.model_type
            self.character = config.character
            self.threshold = config.data_threshold

        self.data = self.load_data(split, split)


    def load_data(self, split=None):
        if self.mode == 'pretrain':
            f_name = f'data/{self.character}.json'
        else:
            f_name = f"data/{split}.json"    

        with open(f_name, 'r') as f:
            data = json.load(f)

        if self.mode == 'pretrain' and split == 'train':
            data = data[:self.threshold]
        elif self.mode == 'pretrain' and split == 'valid':
            data = data[self.threshold:]

        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        if (self.mode == 'pretrain') and (self.model_type=='discriminator'):
            uttr = self.data[idx]['uttr']
            resp = self.data[idx]['resp']
            pred = self.data[idx]['pred']
            return uttr, resp, pred
        
        else:
            uttr = self.data[idx]['uttr']
            resp = self.data[idx]['resp']            
            return uttr, resp




def load_dataloader(config, split=None):
    
    #Conditions for Data Shuffling
    cond1 = config.mode == 'test'
    cond2 = config.mode == 'pretrain' and config.model_type == 'discriminator'

    return DataLoader(Dataset(config, split), 
                      batch_size=config.batch_size, 
                      shuffle=False if cond1 & cond2 else True,
                      num_workers=2,
                      pin_memory=True)