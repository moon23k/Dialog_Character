import os
import json
from collections import defaultdict



def save_text(data, f_name):
    with open(f'{f_name}', 'w') as f:
        f.write('\n'.join(data))



def save_json(data, f_name):
    with open(f"{f_name}", 'w') as f:
        json.dump(data, f)



def concat_datasets(*args):
    concated = []

    for data in args:
        concated.extend(data)

    return concated


def divide_data(data, fixed_len=3000):
    div_1, div_2 = [], []
    div_term = len(data) // fixed_len
    
    for idx, seq in enumerate(data):
        if len(div_2) != fixed_len and idx % div_term == 0:
            div_2.append(seq)
        else:
            div_1.append(seq)
    
    return div_1, div_2



def split_dialogue(dataset, f_path=None, split=None):
    src, trg = [], []

    for dial in dataset:
        utters = dial['dialogue']
        n_utters = len(utters)

        if n_utters < 2:
            continue

        elif n_utters == 2:
            src.append(utters[0])
            trg.append(utters[1])

        #Incase of seq_len is even
        elif n_utters % 2 == 0:
            src.extend(utters[0::2])
            trg.extend(utters[1::2])

            src.extend(utters[1:-1:2])
            trg.extend(utters[2::2])
        
        #Incase of seq_len is odds
        elif n_utters % 2 == 1:
            src.extend(utters[0:-1:2])
            trg.extend(utters[1::2])
            
            src.extend(utters[1::2])
            trg.extend(utters[2::2])

    if f_path is not None:
        os.makedirs(f'{f_path}', exist_ok=True)
        save_text(f"{f_path}/{split}.src", src)
        save_text(f"{f_path}/{split}.trg", trg)        

    return src, trg



def process_daily(split, save=False):
    assert split in ['train', 'valid', 'test']

    orig = [json.loads(line) for line in open(f'dialogue/dailydialog/{split}.json', 'r')]
    processed = []

    for elem in orig:
        utters = elem['dialogue']
        dials = defaultdict(list)

        for utter in utters:
            dials['dialogue'].append(utter['text'])
        processed.append(dials)

    if save:
        os.makedirs('dialogue/daily', exist_ok=True)
        save_json(processed, f"dialogue/daily/{split}.json")        

    return processed




def process_persona(split, save=False):
    assert split in ['train', 'valid', 'test']

    with open(f"dialogue/Persona-Chat/personachat/{split}_self_original.txt") as f:
        orig = f.readlines()
    
    processed = []
    cur_idx, dict_idx = 0, 0
    dials = defaultdict(list)

    for line in orig:
        idx = line.strip().split(" ")[0]
        
        if cur_idx + 1 != int(idx):
            dict_idx +=1
        if '\t' in line:
            utters = line.strip()[len(idx):].strip()
            utters = utters.split('\t\t')[0]
            utters = utters.split('\t')
            dials[dict_idx].extend(utters)
        cur_idx = int(idx)


    for v in dials.values():
        temp_dials = defaultdict(list)
        temp_dials['dialogue'].extend(v)
        processed.append(temp_dials)

    if save:
        os.makedirs('dialogue/persona', exist_ok=True)
        save_json(processed, f"dialogue/persona/{split}.json")

    return processed




def process_empathetic(split, save=False):
    assert split in ['train', 'valid', 'test']

    with open(f'dialogue/empatheticdialogues/empatheticdialogues/{split}.csv', 'r') as f:
        orig = f.readlines()

    processed = []
    dials = defaultdict(list)

    for line in orig[1:]:
        elem = line.strip().split(',')
        dials[elem[0]].append(elem[5].replace('_comma_', ','))
    
    for v in dials.values():
        temp_dials = defaultdict(list)
        temp_dials['dialogue'].extend(v)
        processed.append(temp_dials)

    if save:
        os.makedirs('dialogue/empathetic', exist_ok=True)
        save_json(processed, f"dialogue/empathetic/{split}.json")

    return processed




def process_blended(split, save=False):
    assert split in ['train', 'valid', 'test']

    processed = []

    with open(f'dialogue/blended_skill_talk/{split}.json', 'r') as f:
        orig = json.load(f)

    for elem in orig:
        dials = defaultdict(list)

        dials['dialogue'].append(elem['free_turker_utterance'])
        dials['dialogue'].append(elem['guided_turker_utterance'])
        
        utters = elem['dialog']
        for utter in utters:
            dials['dialogue'].append(utter[-1])
        processed.append(dials)

    if save:
        os.makedirs('dialogue/blended', exist_ok=True)
        save_json(processed, f"dialogue/blended/{split}.json")

    return processed



def process_data(d_name):
    assert d_name in ['daily', 'persona', 'empathetic', 'blended']

    if d_name == 'daily':
        train = process_daily('train')
        valid = process_daily('valid')
        test = process_daily('test')

    elif d_name == 'persona':
        train = process_persona('train')
        valid = process_persona('valid')
        test = process_persona('test')

    elif d_name == 'empathetic':
        train = process_empathetic('train')
        valid = process_empathetic('valid')
        test = process_empathetic('test')

    elif d_name == 'blended':
        train = process_blended('train')
        valid = process_blended('valid')
        test = process_blended('test')
    

    train_src, train_trg = split_dialogue(train)
    valid_src, valid_trg = split_dialogue(valid)
    test_src, test_trg = split_dialogue(test)
    
    src = concat_datasets(train_src, valid_src, test_src)
    trg = concat_datasets(train_trg, valid_trg, test_trg)

    return src, trg



def main():
    daily_src, daily_trg = process_data('daily')
    persona_src, persona_trg = process_data('persona')
    empathetic_src, empathetic_trg = process_data('empathetic')
    blended_src, blended_trg = process_data('blended')

    src = concat_datasets(daily_src, persona_src, empathetic_src, blended_src)
    trg = concat_datasets(daily_trg, persona_trg, empathetic_trg, blended_trg)

    assert len(src) == len(trg)


    train_src, valid_src = divide_data(src)
    train_src, test_src = divide_data(train_src)

    train_trg, valid_trg = divide_data(trg)
    train_trg, test_trg = divide_data(train_trg)

    assert len(train_src) == len(train_trg)
    assert len(valid_src) == len(valid_trg)
    assert len(test_src) == len(test_trg)

    os.makedirs('dialogue/seq', exist_ok=True)
    save_text(train_src, 'dialogue/seq/train.src')
    save_text(valid_src, 'dialogue/seq/valid.src')
    save_text(test_src, 'dialogue/seq/test.src')

    save_text(train_trg, 'dialogue/seq/train.trg')
    save_text(valid_trg, 'dialogue/seq/valid.trg')
    save_text(test_trg, 'dialogue/seq/test.trg')



if __name__ == '__main__':
    main()
    files = next(os.walk('dialogue/seq'))[2]
    assert len(files) == 6