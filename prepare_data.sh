#!/bin/bash
mkdir -p data
cd data
mkdir -p seq tok ids vocab


splits=(train valid test)
langs=(en de)

#Download Data
echo "Downloading Dataset"
bash ../data_processing/download_data.sh
#python3 ../data_processing/download_data.py


#Pre tokenize with moses
echo "Pretokenize with moses"
python3 -m pip install -U sacremoses
for split in "${splits[@]}"; do
    for lang in "${langs[@]}"; do
        sacremoses -l ${lang} -j 8 tokenize < seq/${split}.${lang} > tok/${split}.${lang}
    done
done


#Get sentencepiece
echo "Downloading Sentencepiece"
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig
cd ../../


#Build Sentencepice Vocab and Model
echo "Building Vocab"
cat tok/* > concat.txt
bash ../data_processing/build_vocab.sh -i concat.txt -p vocab/spm
rm concat.txt


#Tokens to Ids
echo "Converting Tokens to Ids"
for split in "${splits[@]}"; do
    for lang in "${langs[@]}"; do
        spm_encode --model=vocab/spm.model --extra_options=bos:eos \
        --output_format=id < tok/${split}.${lang} > ids/${split}.${lang}
        echo " Converting Tokens to Ids on ${split}.${lang} has completed"
    done
done

rm -rf sentencepiece