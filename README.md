## Characteristic Dialogue Generation

> In this repository, we share a series of codes aimed at enhancing the potential for personalized responses, surpassing the limitations of dialogue generation deep learning models that tend to generate monotonous and generic answers.


<br><br>

## Models

**Generator**
> Generator Model takes source-language sequence as input and returns translated target-language sequence. In this experiment, generator is pretrained BlenderBot-small model. Generator already knows how to generate response according to the user utterance. But all responses the model generates is general and typical answers, without any characteristics. To model have the characterisic, GAN Style of Training will be used. Training details will be mentioned in training chapter in below  

<br>

**Discriminator**
> Unlike the generator, discriminator is a binary classification model. Discriminator takes response sequence and tell if it looks like user preset Character's dialog style or not. For this task, Discriminator initialized with pretrained BERT-Small model. This model will first be trained to distinguish pretrained BlenderBot-small model's response and the real response from the HIMYM script. After this pretraining, the model and pretrained model states will be used as a testing evaluation metric. In the actual Training Session, discriminator will be trained with generator to help generator to make better characterized sequence.


<br><br>

## Results
### TBD
<br><br>

## How to Use

<br><br>

## Reference

[SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)

<br>
