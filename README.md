## Character Dialogue Generation

People communicate with others through conversation. And one's personality is shown in the process of conversation. Personality makes the conversation much more enjoyable than the conversation without it. However, in the case of a dialog generation model which learns human conversation patterns based on large-scale data, such individuality does not exist. This repo presents a methodology that can give a personality to a model using seqGAN technique and HIMYM conversation dataset.

<br><br>

## Model

**Generator**
> Generator Model takes source-language sequence as input and returns translated target-language sequence. In this experiment, generator is pretrained BlenderBot-small model. Generator already knows how to generate response according to the user utterance. But all responses the model generates is general and typical answers, without any characteristics. To model have the characterisic, GAN Style of Training will be used. Training details will be mentioned in training chapter in below  

<br>

**Discriminator**
> Unlike the generator, discriminator is a binary classification model. Discriminator takes response sequence and tell if it looks like user preset Character's dialog style or not. For this task, Discriminator initialized with pretrained BERT-Small model. This model will first be trained to distinguish pretrained BlenderBot-small model's response and the real response from the HIMYM script. After this pretraining, the model and pretrained model states will be used as a testing evaluation metric. In the actual Training Session, discriminator will be trained with generator to help generator to make better characterized sequence.

<br><br>

## Data

**HIMYM**
> How I Met Your Mother is a famous sitcom. The sitcom has five main characters with strong personalities. Each character is Ted, Barney, Marshall, Lily, and Robin.

<br>

**Daily Mail**
> This dataset is well known dataset for dialogue generation task. Original dataset is designed for a multi-turn dialogues, but I split those into single-turns. This dataset will be used in the Training Session to make the generator generate Characterisic responses on the general daily dialigue situations.

<br><br>

## Training Process

**Pretrain**
> We deal two models, each of Generator and Discriminator. But as we use pretrained Generator, only Discriminator get trained in this Pre-Training session. Discriminator learns to distinguish real and generator made response sequence. Discriminator will be trained on Binary Cross Entropy Loss.

<br>

**Train**
> On Training Session, both Generator and Discriminator get trained in a seqGAN style. 


<br><br>

## Results

<br><br>

## Reference

<br>
