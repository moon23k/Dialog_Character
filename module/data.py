import json, torch
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()

        self.mode = config.mode
        self.model_type = config.model_type
        
        if self.mode in ['pretrain', 'generate']:
            self.character = config.character
            self.threshold = config.data_threshold

        self.data = self.load_data(split)


    def load_data(self):
        if self.mode in ['pretrain', 'generate']:
            f_name = f'data/{self.character}.json'
        else:
            f_name = f"data/{split}.json"    

        with open(f_name, 'r') as f:
            data = json.load(f)

        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        if self.mode == 'pretrain':
            uttr = self.data[idx]['uttr']
            resp = self.data[idx]['resp']
            pred = self.data[idx]['pred']
            return uttr, resp, pred
        
        else:
            uttr = self.data[idx]['uttr']
            resp = self.data[idx]['resp']            
            return uttr, resp




def load_dataloader(config, split):
    return DataLoader(
        Dataset(config, split), 
        batch_size=config.batch_size, 
        shuffle='train' in config.mode,
        num_workers=2,
        pin_memory=True
    )