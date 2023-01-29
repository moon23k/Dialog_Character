import torch
from tqdm import tqdm



class Tester:
    def __init__(self, config, 
                 g_model, d_model, 
                 g_tokenizer, d_tokenizer
                 test_dataloader):
        
        self.g_model = g_model
        self.d_model = d_model
        
        self.g_tokenizer = g_tokenizer
        self.d_tokenizer = d_tokenizer
        
        self.device = config.device
        self.dataloader = test_dataloader



    def test(self):
        scores = 0

        self.g_model.eval()
        self.d_model.eval()

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.dataloader)):   
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
                d_encodings = self.d_tokenizer(preds, return_tensors='pt').to(self.device)
                logits = self.d_model(**d_encodings)
                scores += logits[logits > 0.5]


        print('Test Results')
        print(f"  >> Test Score: {scores:.2f}")