import os, re, json, requests, argparse
from datasets import load_dataset
from bs4 import BeautifulSoup as bs



def select_data(orig_data, volumn=12000):
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


def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



def get_urls():
    urls = []
    base_url = 'https://transcripts.foreverdreaming.org/viewtopic.php?'
    main_url = 'https://transcripts.foreverdreaming.org/viewforum.php?f=177'
    post_urls = [main_url + f'&start={78 * i}' if i else main_url for i in range(3)]
    
    for url in post_urls:
        html = requests.get(url)
        soup = bs(html.text, "html.parser")
        soup.find_all('a', {'class': "topictitle"})        

        for elem in soup.find_all('a', {'class': "topictitle"})[1:]:
            season = elem.text[:2]
            if int(season) > 6:
                continue

            page_param = elem['href'].split('?')[1].split('&')[0]
            urls.append(base_url + page_param)

    return urls



def clean_script(script):
    clear_script = []
    for line in script:
        if not line or line.startswith('('): continue

        if line.startswith('[') or ':' not in line:
            clear_script.append(line)
            continue

        skip_char = False
        for char in ['narrator', 'son', 'daughter', '2030', 'voix', 'from']:
            if char in line.split(":")[0].lower():
                skip_char = True
                break
        
        if skip_char:
            continue

        if '(' in line:
            while True:
                start_idx = line.find('(')
                end_idx = line.find(')')
                line = line[:start_idx-1] + line[end_idx+1:]
                if '(' not in line:
                    break
                    
        if ':' in line and line.split(':')[1]:
            clear_script.append(line)

    return clear_script    



def split_script(script):
    plot_indice = [idx for idx, line in enumerate(script) if line.startswith('[') or ':' not in line]

    split_indice = []
    for i in range(len(plot_indice)-1):
        if plot_indice[i] - plot_indice[i-1] > 2:
            split_indice.append((plot_indice[i-1] + 1, plot_indice[i]))

    splited = []
    for i in range(len(split_indice)):
        dialog = script[split_indice[i][0]: split_indice[i][1]]
        splited.append({'dialogue': dialog})
    
    return splited    



def split_dialog(script, char):
    dialog = []
    prior_char, prior_uttr = '', ''

    for dial in script:
        for line in dial['dialogue']:
            curr_char = line.split(':')[0].lower().strip()
            curr_uttr = ''.join(line.split(':')[1:]).strip()
            
            if not prior_char:
                if curr_char == char:
                    continue

                prior_char = curr_char
                prior_uttr = curr_uttr
                continue 

            if prior_char != char and curr_char == char:        
                temp = dict()
                temp['uttr'] = prior_uttr.lower()
                temp['resp'] = curr_uttr.lower()

                dialog.append(temp)

            prior_char = curr_char
            prior_uttr = curr_uttr

    return dialog    



def setup_daily():
    orig = load_dataset('daily_dialog', split='train')['dialog']
    selected = select_data(orig)
    save_data(selected)



def setup_himym(character):

    data = []
    urls = get_urls()
    
    for url in urls:
        html = requests.get(url)
        soup = bs(html.text, "html.parser")
        orig = soup.select(".content")[0].text.split('\n')

        cleaned = clean_script(orig)
        splited = split_script(cleaned)
        dialog = split_dialog(splited, character)
        data.extend(dialog)

    with open(f'data/{character}.json', 'w') as f:
        json.dump(data, f)
    assert os.path.exists(f'data/{character}.json')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-char', required=False)
    
    args = parser.parse_args()
    assert args.mode in ['all', 'pretrain', 'train']

    if args.mode != 'pretrain':
        setup_daily()
    elif args.mode != 'train':
        assert args.char in ['ted', 'barney', 'marshall', 'lily', 'robin']
        setup_himym(args.char)
        