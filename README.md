# Light_Transformer

This repo covers two Transformer Architectures. One is Vanilla_Transformer and the other is Light_Transformer. Vanilla Transformer is simple implementation of famous paper "Attention is all you need" and Light Transformer is Lightened version of Vanilla Transformer. To Lighten Transformer Architecture, Parameter Sharing and Factorizing Techniques has applied.

Ultimate goal of the experiment is to Lighten Model with minimal performance degradation.
Comparisons were made for two Natural Language Generation Tasks(Neural Machine Translation and Single turn Dialogue Generation)

<br>
<br>

## Model Architecture

<div>
  <img src="https://user-images.githubusercontent.com/71929682/172880786-4974606f-ecf3-4aa4-907b-6f7ac2beda36.png" width=300 height=400>
  <img src="https://user-images.githubusercontent.com/71929682/172881156-edb1fc4b-b3c1-427f-9af7-df0ad4be79c6.png" width=500 height=400>
</div>

The Figures above represent Vanilla Transformer and Universal Transformer each.

Main Idea to construct Light Transformer has borrowed from Universal Transformer. As shown in the figure on the right above, Encoder and Decoder Block have recurrent structure. By using this method, weights are shared and recurrent bias can be used unlike the Vanilla Transformer.

In addition, Light Transformer also use Parameter Factorizing which proposed in TransformerXL paper.


<br>

### Number of Trainable Parameters

**Vanilla Transformer:** 48M

**Light Transformer:** 14M

<br>

<br>

## Training Setup

* Neural Machine Translation Dataset: downsized WMT14 (EN-DE)
* Single Turn Dialogue Generation Dataset: Processed Single-Turn Dialogue Dataset
* Tokenization: BPE
* Batch_size: 128
* Num of Epochs: 10
* Learning Rate: 1e-4
* scheduler: Noam




<br>


## Results
Evaluation
Speed

<br>

<br>

## Reference
Universal Transformer
Transformer XL
