import os
from datasets import load_dataset


def split_data(data_obj):
	src, trg = [], []

	for d in data_obj:
		src.append(d['en'])
		trg.append(d['de'])

	return src, trg



def save_file(data_obj, f_name):
	with open(f'seq/{f_name}', 'w') as f:
		f.write('\n'.join(data_obj))



def run():
	train = load_dataset('wmt14', 'de-en', split='train')
	valid = load_dataset('wmt14', 'de-en', split='validation')
	test = load_dataset('wmt14', 'de-en', split='test')

	train = train['translation']
	valid = valid['translation']
	test = test['translation']

	#Downsize train dataset
	train = train[::10]

	#split data
	train_src, train_trg = split_data(train)
	valid_src, valid_trg = split_data(valid)
	test_src, test_trg = split_data(test)


	#save data obj to files
	save_file(train_src, 'train.src')
	save_file(train_trg, 'train.trg')

	save_file(valid_src, 'valid.src')
	save_file(valid_trg, 'valid.trg')

	save_file(test_src, 'test.src')
	save_file(test_trg, 'test.trg')



if __name__ == '__main__':
	print('Data Downloading and Processing Started!')
	run()
	files = next(os.walk('seq'))[2]
	assert len(files) == 6
	print('Data Downloading and Processing Completed!')