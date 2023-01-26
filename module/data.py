import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.character = config.character
        self.data = self.load_data(self.character)


    def load_data(self, character):
        f_name = f'data/himym_{character}.json'
        with open(f_name, 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids']
        attention_mask = self.data[idx]['attention_mask']
        labels = self.data[idx]['labels']
        return input_ids, attention_mask, labels


def pad_batch(batch_list, pad_id):
    return pad_sequence(batch_list,
                        batch_first=True,
                        padding_value=pad_id)


def load_dataloader(config):
    global pad_id
    pad_id = config.pad_id    

    def collate_fn(batch):
        ids_batch, mask_batch, labels_batch = [], [], []

        for input_ids, attention_mask, labels in batch:
            ids_batch.append(torch.LongTensor(input_ids)) 
            mask_batch.append(torch.LongTensor(attention_mask))
            labels_batch.append(torch.LongTensor(labels))
        
        ids_batch = pad_batch(ids_batch, pad_id)
        mask_batch = pad_batch(mask_batch, pad_id)
        labels_batch = pad_batch(labels_batch, pad_id)

        return {'input_ids': ids_batch,
                'attention_mask': mask_batch,
                'labels': labels_batch}


    dataset = Dataset(config)
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      shuffle=False,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)