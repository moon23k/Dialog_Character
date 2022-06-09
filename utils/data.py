import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



def read_text(d_name, f_name, max_words=50):
    with open(f'data/{d_name}/ids/{f_name}', 'r', encoding='utf-8') as f:
        orig_data = f.readlines()

    #cut long sentences with max_words limitation
    data = []
    for seq in orig_data:
        _seq = list(map(int, seq.split()))

        if len(_seq) > max_words:
            _seq = _seq[:max_words - 1]
            _seq.append(3) #append eos token

        data.append(_seq)
    
    return data



class CustomDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src = src_data
        self.trg = trg_data
    
    def __len__(self):
        return len(self.trg)
    
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]



def _collate_fn(data_batch):    
    src_batch, trg_batch = [], []

    for batch in data_batch:
        src = torch.tensor(batch[0], dtype=torch.long)
        trg = torch.tensor(batch[1], dtype=torch.long)
        src_batch.append(src)
        trg_batch.append(trg)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=1)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=1)

    return src_batch, trg_batch



def get_dataloader(split, config):
    src = read_text(config.task, f"{split}.src")
    trg = read_text(config.task, f"{split}.trg")

    dataset = CustomDataset(src, trg)
    iterator = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=_collate_fn, num_workers=2)
    
    return iterator