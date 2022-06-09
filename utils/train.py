import math
import torch
import torch.nn as nn




def create_src_mask(src, pad_idx=1):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    src_mask.to(src.device)
    return src_mask



def create_trg_mask(trg, pad_idx=1):
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
    trg_sub_mask = torch.tril(torch.ones((trg.size(-1), trg.size(-1)))).bool()

    trg_mask = trg_pad_mask & trg_sub_mask.to(trg.device)
    trg_mask.to(trg.device)
    return trg_mask




def train_epoch(model, dataloader, criterion, optimizer, config):
    model.train()
    epoch_loss = 0
    total_len = len(dataloader)

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        src, trg = batch[0].to(config.device), batch[1].to(config.device)

        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:].contiguous().view(-1)

        pred = model(src, trg_input)
        pred = pred.contiguous().view(-1, config.output_dim)

        loss = criterion(pred, trg_y)
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), max_norm=config.clip)
        optimizer.step()
        epoch_loss += loss.item()

        if (i + 1) % 1000 == 0:
            print(f"---- Step: {i+1}/{total_len} Train Loss: {round(loss, 3)}")

    return epoch_loss / total_len




def eval_epoch(model, dataloader, criterion, config):
    model.eval()
    epoch_loss = 0
    total_len = len(dataloader)

    
    for i, batch in enumerate(dataloader):
        src, trg = batch[0].to(config.device), batch[1].to(config.device)
        
        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:].contiguous().view(-1)
    
        with torch.no_grad():
            pred = model(src, trg_input)
        
        pred_dim = pred.shape[-1]
        pred = pred.contiguous().view(-1, pred_dim)

        loss = criterion(pred, trg_y)
        epoch_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"---- Step: {i+1}/{total_len} Eval Loss: {round(loss, 3)}")

    return epoch_loss / total_len