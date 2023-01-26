import os, re, json
from datasets import load_dataset
from transformers import BlenderbotSmallTokenizer



def tokenize_data(data_obj, tokenizer):
    tokenized_data = []
    for elem in data_obj:

        temp_dict = dict()
        encodings = tokenizer(elem['uttr'], truncation=True)

        temp_dict['input_ids'] = encodings.input_ids
        temp_dict['attention_mask'] = encodings.attention_mask
        temp_dict['labels'] = tokenizer.encode(elem['resp'], truncation=True)

        tokenized_data.append(temp_dict)
    
    return tokenized_data



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-6000], data_obj[-6000:-3000], data_obj[-3000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



def process_daily(orig_data, volumn=36000):
    volumn_cnt = 0
    uttr_list, resp_list, processed = [], [], []

    for dial in orig_data:
        dial_list = []
        dial_turns = len(dial)
        
        for uttr in dial:
            _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
            _uttr = re.sub(r'([’])\s+', r'\1', _uttr)
            dial_list.append(_uttr.strip().lower())
        
        if dial_turns < 2:
            continue

        elif dial_turns == 2:
            uttr_list.append(dial_list[0])
            resp_list.append(dial_list[1])
            continue  #To avoid duplicate on below condition

        #Incase of dial_turns is even
        elif dial_turns % 2 == 0:
            uttr_list.extend(dial_list[0::2])
            resp_list.extend(dial_list[1::2])

            uttr_list.extend(dial_list[1:-1:2])
            resp_list.extend(dial_list[2::2])
        
        #Incase of dial_turns is odds
        elif dial_turns % 2 == 1:
            uttr_list.extend(dial_list[0:-1:2])
            resp_list.extend(dial_list[1::2])
            
            uttr_list.extend(dial_list[1::2])
            resp_list.extend(dial_list[2::2])   

    assert len(uttr_list) == len(resp_list)
    for uttr, resp in zip(uttr_list, resp_list):
        temp_dict = dict()
        temp_dict['uttr'] = uttr
        temp_dict['resp'] = resp
        processed.append(temp_dict)

        #End Condition
        volumn_cnt += 1
        if volumn_cnt == volumn:
            break
    
    return processed




def main():
    tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    orig = load_dataset('daily_dialog', split='train')['dialog']
    processed = process_daily(orig)
    tokenized = tokenize_data(processed, tokenizer)
    save_data(tokenized)



if __name__ == '__main__':
    main()