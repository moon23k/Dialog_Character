import os
from datasets import load_dataset


def split_data(data_obj):
	src, trg = [], []

	for d in data_obj:
		src.append(d['en'])
		trg.append(d['de'])

	return src, trg



def save_text(data, f_name):
	with open(f'translate/seq/{f_name}', 'w') as f:
		f.write('\n'.join(data))



def process_data(split):
	assert split in ['train', 'validation', 'test']

	data = load_dataset('wmt14', 'de-en', split=split)
	data = data['translation']

	if split == 'train':
		data = data[::10]
	elif split == 'validation':
		split = 'valid'

	src, trg = split_data(data)
	save_text(src, f"{split}.src")
	save_text(trg, f"{split}.trg")




if __name__ == '__main__':
	process_data('train')
	process_data('validation')
	process_data('test')
	
	files = next(os.walk('translate/seq'))[2]
	assert len(files) == 6