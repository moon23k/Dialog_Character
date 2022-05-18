import torch
import argparse
import sentencepiece as spm
from utils.util import Config, load_model
from utils.train import create_src_mask, create_trg_mask





def run(model, tokenizer, config, max_tokens=100):
    with torch.no_grad():
        print('Type "quit" to terminate Translation')
        while True:
            seq = input('\nUser Input sentence >> ')
            if seq == 'quit':
                print(' --- Terminate the Translation! ---')
                break

            #Convert tokens to ids with sentencepiece vocab
            src = tokenizer.EncodeAsIds(seq)

            #Convert ids to tensor
            src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(config.device)
            
            src_mask = create_src_mask(src)
            src = model.embedding(src)
            enc_out = model.encoder(src, src_mask)
            trg_indice = [tokenizer.bos_id()]

            for _ in range(max_tokens):
                trg_tensor = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0).to(config.device)
                trg_mask = create_trg_mask(trg_tensor)

                trg = model.embedding(trg_tensor)

                dec_out, _ = model.decoder(enc_out, trg, src_mask, trg_mask)
                out = model.fc_out(dec_out)

                pred_token = out.argmax(2)[:, -1].item()
                trg_indice.append(pred_token)

                if pred_token == tokenizer.eos_id():
                    break
                
            pred_seq = trg_indice[1:]
            pred_seq = tokenizer.Decode(pred_seq)

            print(f"Translated sentence >> {pred_seq}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-scheduler', default='constant', required=False)
    args = parser.parse_args()
    
    assert args.model in ['valilla', 'light']

    config = Config(args)
    config.device = torch.device('cpu')

    #Load Model
    model = load_model(config)
    model_state = torch.load()
    model.load_state_dict(model_state)
    model.eval()

    #Load Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/vocab/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')

    run(model. tokenizer, config)
