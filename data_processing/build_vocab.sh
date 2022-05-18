#!/bin/bash

while getopts i:p: flag; do
    case "${flag}" in
        i) input=${OPTARG};;
        p) prefix=${OPTARG};;
    esac
done


function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

eval $(parse_yaml ../configs/vocab.yaml)


spm_train --input=$input --model_prefix=$prefix \
--vocab_size=$vocab_size --character_coverage=$coverage --model_type=$type \
--unk_id=$unk_id --unk_piece=$unk_piece --pad_id=$pad_id --pad_piece=$pad_piece \
--bos_id=$bos_id --bos_piece=$bos_piece --eos_id=$eos_id --eos_piece=$eos_piece