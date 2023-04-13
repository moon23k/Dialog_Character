import time, json, torch
import torch.nn as nn
import torch.amp as amp
import torch.optim as optim



class TrainerBase:
    def __init__(self, config):
        
        self.mode = config.mode
        self.clip = config.clip
        self.device = config.device
        self.max_len = config.max_len
        self.n_epochs = config.n_epochs
        self.device_type = config.device_type
        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate

        self.early_stop = config.early_stop
        self.patience = config.patience


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"



    def tokenize(self, tokenizer, tokenizer_inputs):
        return tokenizer(tokenizer_inputs, 
                         padding=True, 
                         truncation=True, 
                         return_tensors='pt').to(self.device)



    def generate(self, uttr):        
        g_encodings = self.tokenize(self.g_tokenizer, uttr)

        with torch.autocast(device_type=self.device_type, dtype=torch.float16):
            preds = self.g_model.generate(input_ids=g_encodings.input_ids,
                                          attention_mask=g_encodings.attention_mask, 
                                          max_new_tokens=self.max_len, 
                                          use_cache=True)

        return self.g_tokenizer.batch_decode(preds, skip_special_tokens=True)



    def save_ckpt(self, epoch, ckpt, model, optimizer):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    ckpt)

        


class Trainer(TrainerBase):
    def __init__(self, config, g_model, d_model, g_tokenizer, 
                 d_tokenizer, train_dataloader, valid_dataloader):
        
        super(Trainer, self).__init__(config)

        self.g_model = g_model
        self.d_model = d_model

        self.g_tokenizer = g_tokenizer
        self.d_tokenizer = d_tokenizer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.g_optimizer = optim.AdamW(params=self.g_model.parameters(), lr=config.lr)
        self.d_optimizer = optim.AdamW(params=self.d_model.parameters(), lr=config.lr)

        self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min')
        self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer, 'min')

        self.g_ckpt = config.g_ckpt
        self.d_ckpt = config.d_ckpt

        self.record_path = 'ckpt/train.json'
        self.record_keys = ['epoch', 'g_train_loss', 'g_valid_loss',
                            'd_train_loss', 'd_valid_loss', 'g_lr', 'd_lr', 'epoch_time']



    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))

        print(f"""  >> Generator Train Loss: {record_dict['g_train_loss']:.3f} | \
              Generator Valid Loss: {record_dict['g_valid_loss']:.3f}\n""".replace(' ' * 14, ''))

        print(f"""  >> Discriminator Train Loss: {record_dict['d_train_loss']:.3f} | \
              Discriminator Valid Loss: {record_dict['d_valid_loss']:.3f}\n""".replace(' ' * 14, ''))



    def get_losses(self, batch):
        uttr, resp = batch[0], batch[1]
        batch_size = len(uttr)

        #Generate Predictions through Generator Model
        pred = self.generate(uttr)

        #Tokenize inputs for discriminator
        d_encodings = self.tokenize(self.d_tokenizer, preds + resp)
        d_ids = d_encodings.input_ids
        d_masks = d_encodings.attention_mask

        d_labels = torch.cat((torch.zeros(batch_size), torch.ones(batch_size)), dim=0).to(self.device)
        d_indice = torch.randperm(batch_size * 2)

        #Shuffle Discriminator inputs
        d_ids = d_ids[d_indice].to(self.device)
        d_masks = d_masks[d_indice].to(self.device)
        d_labels = d_labels[d_indice].to(self.device)

        with torch.autocast(device_type=self.device_type, dtype=torch.float16):
            d_outs = d_model(input_ids=d_ids, attention_mask=d_mask, labels=d_labels)
        

        #To be modified
        pos = d_outs.logit[d_labels == 1] > 0.5
        g_loss = -torch.log(pos.sum() / batch_size)

        return g_loss, d_outs.loss



    def train(self):
        records = []
        patience = self.patience
        g_best_loss, d_best_loss  = float('inf'), float('inf')

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


            #save best discriminator states
            if d_best_loss >= d_curr_loss:
                d_best_loss = d_curr_loss
                self.save_ckpt(epoch, self.d_ckpt, self.d_model, self.d_optimizer)


            #save best generator states
            if g_best_loss >= g_curr_loss:
                g_best_loss = g_curr_loss
                self.save_ckpt(epoch, self.g_ckpt, self.g_model, self.g_optimizer)

                #patience intialize
                if self.early_stop:
                    patience = self.patience
            
            else:
                if not self.early_stop:
                    continue
                patience -= 1
                if not patience:
                    print('\n--- Training Ealry Stopped ---')
                    break


        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)        
            


    def train_epoch(self):
        g_epoch_loss, d_epoch_loss = 0, 0
        tot_len = len(self.train_dataloader)
        
        self.g_model.train()
        self.d_model.train()


        for idx, batch in enumerate(self.train_dataloader):
            
            idx += 1
            g_loss, d_loss = self.get_losses(batch)
            
            g_loss = g_loss / self.iters_to_accumulate
            d_loss = d_loss / self.iters_to_accumulate

            self.scaler.scale(g_loss).backward()
            self.scaler.scale(d_loss).backward()
            

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):    
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

            gen_epoch_loss += g_loss.item()
            dis_epoch_loss += d_loss.item()
        
        g_epoch_loss = round(g_epoch_loss / tot_len, 3)
        d_epoch_loss = round(d_epoch_loss / tot_len, 3)
    
        return g_epoch_loss, d_epoch_loss
    


    def valid_epoch(self):
        g_epoch_loss, d_epoch_loss = 0, 0
        tot_len = len(self.valid_dataloader)

        self.g_model.eval()
        self.d_model.eval()
        
        with torch.no_grad():
            for batch in self.valid_dataloader:   
                self.update_inputs(batch)       
                g_loss, d_loss = self.get_losses()

                g_epoch_loss += g_loss.item()
                d_epoch_loss += d_loss.item()
    
        g_epoch_loss = round(g_epoch_loss / tot_len, 3)
        d_epoch_loss = round(d_epoch_loss / tot_len, 3)

        return g_epoch_loss, d_epoch_loss




class GenTrainer(TrainerBase):
    def __init__(self, config, model, tokenizer, train_dataloader, valid_dataloader):
        
        super(GenTrainer, self).__init__(config)

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        self.ckpt = config.ckpt
        self.record_path = "ckpt/generator_base.json"
        self.record_keys = ['epoch', 'train_loss', 'valid_loss', 'lr', 'train_time']
        


    def train(self):

        records = []
        best_loss = float('inf')
        patience = self.patience


        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()
            record_vals = [epoch, self.train_epoch(), self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
                #patience intialize
                if self.early_stop:
                    patience = self.patience
            
            else:
                if not self.early_stop:
                    continue
                patience -= 1
                if not patience:
                    print('\n--- Training Ealry Stopped ---')
                    break

        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)    


    def train_epoch(self):
        epoch_loss = 0
        tot_len = len(self.train_dataloader)
        self.model.trian()
        
        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            uttr, resp = batch[0], batch[1]
            uttr_encodings = self.tokenize('gen', uttr)
            resp_encodings = self.tokenizer('gen', resp)

            input_ids = uttr_encodings.input_ids.to(self.device)
            attention_mask = uttr_encodings.attention_mask.to(self.device)
            labels = resp_encodings.input_ids.to(self.device)

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels).loss
                loss = loss / self.iters_to_accumulate

            #Backward Loss
            self.scaler.scale(loss).backward()        

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()                

            epoch_loss += loss    

        epoch_loss = round(epoch_loss / tot_len, 3)
        return epoch_loss


    def valid_epoch(self):
        epoch_loss = 0
        self.model.eval()

        for batch in self.valid_dataloader:
            uttr, resp = batch[0], batch[1]
            uttr_encodings = self.tokenize('gen', uttr)
            resp_encodings = self.tokenizer('gen', resp)

            input_ids = uttr_encodings.input_ids.to(self.device)
            attention_mask = uttr_encodings.attention_mask.to(self.device)
            labels = resp_encodings.input_ids.to(self.device)


            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.model(input_ids=input_ids, 
                                  attention_mask=attention_mask,
                                  labels=labels).loss

            epoch_loss += loss

        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)
        return epoch_loss




class DisTrainer(TrainerBase):
    def __init__(self, config, g_model, d_model, g_tokenizer, 
                 d_tokenizer, train_dataloader, valid_dataloader):
        
        super(DisTrainer, self).__init__(config)

        self.g_model = g_model
        self.d_model = d_model

        self.g_tokenizer = g_tokenizer
        self.d_tokenizer = d_tokenizer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = optim.AdamW(params=self.d_model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        self.ckpt = config.d_ckpt
        self.record_path = 'ckpt/discriminator_base.json'
        self.record_keys = ['epoch', 'train_loss', 'valid_loss', 'lr', 'train_time']



    def train(self):
        records = []        
        best_loss = float('inf')
        patience = self.patience

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()
            record_vals = [epoch, self.train_epoch(), self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}

            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
                
                #patience intialize
                if self.early_stop:
                    patience = self.patience
            
            else:
                if not self.early_stop:
                    continue
                patience -= 1
                if not patience:
                    print('\n--- Training Ealry Stopped ---')
                    break

        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)    



    def get_loss(self, batch):
        uttr, resp = batch[0], batch[1]
        batch_size = len(uttr)

        #Generate Predictions through Generator Model
        pred = self.generate(uttr)

        #Tokenize inputs for discriminator
        d_encodings = self.tokenize(self.d_tokenizer, preds + resp)
        d_ids = d_encodings.input_ids
        d_masks = d_encodings.attention_mask

        d_labels = torch.cat((torch.zeros(batch_size), torch.ones(batch_size)), dim=0).to(self.device)
        d_indice = torch.randperm(batch_size * 2)

        #Shuffle Discriminator inputs
        d_ids = d_ids[d_indice].to(self.device)
        d_masks = d_masks[d_indice].to(self.device)
        d_labels = d_labels[d_indice].to(self.device)

        with torch.autocast(device_type=self.device_type, dtype=torch.float16):
            loss = d_model(input_ids=d_ids, attention_mask=d_mask, labels=d_labels).loss
        
        return loss


    def train_epoch(self):
        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        d_model.train()
        g_model.eval()        

        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            loss = self.get_loss(batch)
            loss = loss / self.iters_to_accumulate

            #Backward Loss
            self.scaler.scale(loss).backward()        

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.d_model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()                

            epoch_loss += loss    

        epoch_loss = round(epoch_loss / tot_len, 3)
        return epoch_loss
    

    def valid_epoch(self):
        epoch_loss = 0

        d_model.eval()
        g_model.eval()

        for batch in self.valid_dataloader:
            loss = self.get_loss(batch)
            epoch_loss += loss

        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)

        return epoch_loss