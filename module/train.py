import time, math, json, torch
from tqdm import tqdm
import torch.nn as nn
import torch.amp as amp
import torch.optim as optim
from collections import namedtuple



class Trainer:
    def __init__(self, config, 
                 g_model, d_model, 
                 g_tokenizer, d_tokenizer, 
                 train_dataloader, valid_dataloader):

        self.mode = config.mode
        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs
        self.device_type = config.device_type
        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate

        self.g_model = g_model
        self.d_model = d_model

        self.g_tokenizer = gen_tokenizer
        self.d_tokenizer = dis_tokenizer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.g_optimizer = optim.AdamW(params=self.generator.parameters(), lr=config.lr)
        self.d_optimizer = optim.AdamW(params=self.discriminator.parameters(), lr=config.lr)

        self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.gen_optimizer, 'min')
        self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.dis_optimizer, 'min')

        self.g_ckpt = config.g_ckpt
        self.d_ckpt = config.d_ckpt
        
        self.g_inputs = namedtuple('Generator_Inputs', ('ids', 'masks', 'labels'))
        self.d_inputs = namedtuple('Discriminator_Inputs', ('ids', 'masks', 'labels'))

        self.record_path = 'ckpt/gan_train.json'
        self.record_keys = ['epoch', 'g_train_loss', 'g_valid_loss',
                            'd_train_loss', 'd_valid_loss', 'g_lr', 'd_lr', 'train_time']


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        if isinstance(self, PreTrainer):
            print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
                  Valid Loss: {record_dict['valid_loss']:.3f}\n""".replace(' ' * 14, ''))
        elif isinstance(self, Trainer):
            print(f"""  >> Generator Train Loss: {record_dict['g_train_loss']:.3f} | \
                  Generator Valid Loss: {record_dict['g_valid_loss']:.3f}""".replace(' ' * 14, ''))            
            print(f"""  >> Discriminator Train Loss: {record_dict['dis_train_loss']:.3f} | \
                  Discriminator Valid Loss: {record_dict['d_valid_loss']:.3f}\n""".replace(' ' * 14, ''))


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"        


    def save_ckpt(self, epoch, ckpt, model, optimizer):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    ckpt)


    def update_inputs(self, batch):
        uttr, resp = batch[0], batch[1]

        #tokenize inputs for generator
        g_uttr_encodings = self.g_tokenizer(uttr).to(self.device)
        g_ids = g_uttr_encodings.input_ids
        g_masks = g_uttr_encodings.attention_mask
        g_labels = self.g_tokenizer(resp).input_ids

        #generate predictions
        preds = self.g_model.generate(input_ids=g_ids,
                                      attention_mask=g_masks, 
                                      max_new_tokens=self.max_tokens, 
                                      use_cache=True)
        #Decode generator predictions
        preds = self.g_tokenizer.batch_decode(preds, skip_special_tokens=True)


        #Tokenize inputs for discriminator
        d_inputs = resp + pred
        d_encodings = self.d_tokenizer(d_inputs, return_tensors='pt').to(self.device)
        
        d_ids = d_encodings.input_ids
        d_masks = d_encodings.attention_mask
        d_labels = torch.cat((torch.zeros(), torch.ones()), dim=0).to(self.device)
        d_indice = torch.randperm(d_ids.size(0))

        #Shuffle Discriminator inputs
        d_ids = d_ids[d_indice].to(self.device)
        d_masks = dis_masks[d_indice].to(self.device)
        d_labels = d_labels[d_indice].to(self.device)

        #Update inputs
        self.gen_inputs.input_ids = gen_ids
        self.gen_inputs.attention_mask = gen_masks
        self.gen_inputs.labels = gen_labels

        self.dis_inputs.input_ids = gen_ids
        self.dis_inputs.attention_mask = gen_masks
        self.dis_inputs.labels = gen_labels



    def get_losses(self):
        with torch.autocast(device_type=self.device_type, dtype=torch.float16):
            gen_loss = self.generator(input_ids=self.gen_inputs.ids, 
                                      attention_mask=self.gen_inputs.masks,
                                      labels=self.gen_inputs.labels).loss

            dis_loss = self.discriminator(input_ids=self.dis_inputs.ids, 
                                          attention_mask=self.dis_inputs.masks,
                                          labels=self.dis_inputs.labels).loss

        return (gen_loss + dis_loss.item()) * 0.5, dis_loss


    def train_epoch(self):
        gen_epoch_loss, dis_epoch_loss = 0, 0
        tot_len = len(self.train_dataloader)
        
        self.generator.eval() if self.mode == 'pretrain' else self.generator.train()
        self.discriminator.train()


        for idx, batch in tqdm(enumerate(self.train_dataloader)):
            self.update_inputs(batch)
            gen_loss, dis_loss = self.get_losses()

            gen_loss = gen_loss / self.iters_to_accumulate
            dis_loss = dis_loss / self.iters_to_accumulate

            self.scaler.scale(gen_loss).backward()
            self.scaler.scale(dis_loss).backward()
            
            if (idx + 1) % self.iters_to_accumulate == 0:
                #Gradient Clipping
                self.scaler.unscale_(self.gen_optimizer)
                self.scaler.unscale_(self.dis_optimizer)
                nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.clip)
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.gen_optimizer)
                self.scaler.step(self.dis_optimizer)
                
                self.scaler.update()
                self.gen_optimizer.zero_grad()
                self.dis_optimizer.zero_grad()

            gen_epoch_loss += gen_loss.item()
            dis_epoch_loss += dis_loss.item()
        
        gen_epoch_loss = round(gen_epoch_loss / tot_len, 3)
        dis_epoch_loss = round(dis_epoch_loss / tot_len, 3)
        return gen_epoch_loss, dis_epoch_loss
    


    def valid_epoch(self):
        gen_epoch_loss, dis_epoch_loss = 0, 0
        tot_len = len(self.valid_dataloader)

        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.valid_dataloader)):   
                self.update_inputs(batch)       
                gen_loss, dis_loss = self.get_losses()

                gen_epoch_loss += gen_loss.item()
                dis_epoch_loss += dis_loss.item()
    
        gen_epoch_loss = round(gen_epoch_loss / tot_len, 3)
        dis_epoch_loss = round(dis_epoch_loss / tot_len, 3)
        return gen_epoch_loss, dis_epoch_loss



    def train(self):
        gen_best_loss, dis_best_loss, records = float('inf'), float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.g_optimizer.param_groups[0]['lr'],
                           self.d_optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            g_curr_loss = record_dict['gen_valid_loss']
            d_curr_loss = record_dict['dis_valid_loss']
            self.g_scheduler.step(g_curr_loss)
            self.d_scheduler.step(d_curr_loss)

            #save best generator states
            if g_best_loss >= g_curr_loss:
                g_best_loss = g_curr_loss
                self.save_ckpt(epoch, self.g_ckpt, self.g_model, self.g_optimizer)

            #save best discriminator states
            if d_best_loss >= d_curr_loss:
                d_best_loss = d_curr_loss
                self.save_ckpt(epoch, self.d_ckpt, self.d_model, self.d_optimizer)

        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)        