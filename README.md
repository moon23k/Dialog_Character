## Dialogue Generation via Sequence GAN


Dialogue Generation is a task which generating appropriate dialogue responses for given utterances.
Generalization of dialog generation models is more difficult than other Natural Language Generation Tasks because of the two main factors. First is that loss function-based learning induces the model to generate only general answers. Second is that there is no quantitative evaluation metrics to determine the appropriateness of generated sentences.
Of course, it is possible to overcome these limitations through learning based on large-scale data like GPT, but it is difficult to apply this method to general individuals.



<br><br>

## Model Structure

**Generator**
> Generator Model takes source-language sequence as input and returns translated target-language sequence. In this experiment, generator is pretrained BlenderBot-small model. Generator already knows how to generate response according to the user utterance. But all responses the model generates is general and typical answers, without any characteristics. To model have the characterisic, GAN Style of Training will be used. Training details will be mentioned in training chapter in below  

<br>

**Discriminator**
> Unlike the generator, discriminator is a binary classification model. Discriminator takes response sequence and tell if it looks like user preset Character's dialog style or not. For this task, Discriminator initialized with pretrained BERT-Small model. This model will first be trained to distinguish pretrained BlenderBot-small model's response and the real response from the HIMYM script. After this pretraining, the model and pretrained model states will be used as a testing evaluation metric. In the actual Training Session, discriminator will be trained with generator to help generator to make better characterized sequence.

<br><br>

## Training Configurations


<br><br>

## Training Process

The Training Process consists of two stages.

**Pretrain**
> We deal two models, each of Generator and Discriminator. But as we use pretrained Generator, only Discriminator get trained in this Pre-Training session. Discriminator learns to distinguish real and generator made response sequence. Discriminator will be trained on Binary Cross Entropy Loss.

<br>

**Train**
> On Training Session, both Generator and Discriminator get trained in a seqGAN style. 


<br><br>

## Results

<br><br>

## Reference

[SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)
