import time
import argparse

import torch
import torch.nn as nn
import sentencepiece as spm

from utils.data import get_dataloader
from utils.train import eval_epoch
from utils.util import Config, epoch_time, set_seed, load_model



def run(config):
    chk_file = f"checkpoints/{config.model}_states.pt"
    test_dataloader = get_dataloader('test', config)

    #Load Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/vocab/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')

    #Load Model
    model = load_model(config)
    model_state = torch.load(f'checkpoints/{config.model}_states.pt', map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)
    model.eval()

    #do not apply label smoothing on test dataset
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)

    start_time = time.time()
    print('Test')
    test_loss = eval_epoch(model, test_dataloader, criterion, config)
    end_time = time.time()
    test_mins, test_secs = epoch_time(start_time, end_time)

    print(f"[ Test Loss: {test_loss} / Test BLEU Score: {test_bleu} / Time: {test_mins}min {test_secs}sec ]")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    args = parser.parse_args()

    assert args.model in ['valilla', 'light']

    set_seed()
    config = Config(args)
    run(config)