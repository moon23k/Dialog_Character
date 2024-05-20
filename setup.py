import os, re, json, yaml, argparse
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




def process_daily():
    result, corpus = [], []
    orig_data = load_dataset('daily_dialog')

    for split in ['train', 'validation', 'test']:
        for dial in orig_data[split]['dialog']:

            if max([len(d) for d in dial]) > 100 or len(dial) < 6:
                continue

            uttr_list = []
            for uttr in dial:
                _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
                _uttr = re.sub(r'([’])\s+', r'\1', _uttr).strip().lower()
                uttr_list.append(_uttr)


            corpus.extend(uttr_list)
            while len(uttr_list) >= 6:
                for i in range(1, len(uttr_list), 2):
                    elem = {}
                    elem['hist'] = uttr_list[:i-1] if i > 1 else []
                    elem['x'] = uttr_list[i-1]
                    elem['y'] = uttr_list[i]
                    result.append(elem)
                    if i > 6:
                        break

                uttr_list = uttr_list[1:]

    return result, corpus




def process_blended():

    result, corpus = [], []    
    orig_data = load_dataset('blended_skill_talk')

    for split in ['train', 'validation', 'test']:
        for elem in orig_data[split]:
            pre_fn = lambda s: s.lower().strip()

            uttr_list = []
            for uttr in elem['previous_utterance']:
                uttr_list.append(pre_fn(uttr))

            for uttr, resp in zip(elem['free_messages'], elem['guided_messages']):
                uttr_list.append(pre_fn(uttr))
                uttr_list.append(pre_fn(resp))

            if max([len(x) for x in uttr_list]) > 100 or len(uttr_list) < 14:
                continue

            corpus.extend(uttr_list)
            while len(uttr_list) >= 6:
                for i in range(1, len(uttr_list), 2):
                    elem = {}
                    elem['hist'] = uttr_list[:i-1] if i > 1 else []
                    elem['x'] = uttr_list[i-1]
                    elem['y'] = uttr_list[i]
                    result.append(elem)
                    if i > 6:
                        break

                uttr_list = uttr_list[1:]

    return result, corpus



def train_tokenizer(config, corpus):
    
    corpus_path = f'data/corpus.txt'
    with open(corpus_path, 'w') as f:
        f.write('\n'.join(corpus))    

    assert os.path.exists(corpus_path)
    assert os.path.exists('config.yaml')
    
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']


    tokenizer = Tokenizer(BPE(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_config['vocab_size'], 
        special_tokens=[
            vocab_config['pad_token'], 
            vocab_config['unk_token'],
            vocab_config['bos_token'],
            vocab_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save("data/tokenizer.json")





def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-5100], data_obj[-5100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')




def main(dataset):
    if dataset == 'daily':
        data, corpus = process_daily()
    elif dataset == 'blended':
        data, corpus = process_blended()
    else:
        daily_data, daily_corpus = process_daily()
        blend_data, blend_corpus = process_blended()
        data, corpus = daily_data + blend_data, daily_corpus + blend_corpus 

    train_tokenizer(corpus)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True)
    
    args = parser.parse_args()
    assert args.dataset in ['all', 'daily', 'blended']
    
    main(args.dataset)