# DNN Speech Recognizer

This project is the third and final project of the Udacity Natural Language Processing (NLP) nanodegree and focuses on building a Deep Neural Network (DNN) based speech recognizer. The goal is to develop a system that can accurately recognize spoken words from audio data. In this notebook, we will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline.


## Project Overview
The DNN speech recognizer project involves the following key steps:

__Data Preprocessing__: The audio data is transformed into spectrograms, which are visual representations of the audio signals. These spectrograms serve as input features for the DNN model.

__Acoustic Modeling__: The DNN model is trained to learn the relationship between the spectrograms and the corresponding spoken words. This involves designing and training a DNN architecture using TensorFlow.

__Language Modeling__: Language models are used to improve the recognition accuracy by considering the probability of word sequences. This step involves creating and integrating a language model into the recognizer.

__Decoding__: The trained DNN model and language model are combined to perform decoding, where the system predicts the most likely word sequence given the input audio.


## Results

This project consist 5 model and the outcome of each is detailed below:

__Model 0: RNN__
> This model has the highest starting loss at 1 epoch. The starting loss rate is 844 and the reduction is quite slow and on a constant 779 up till epoch 18. The final loss rate is 799. So far, this model is the poorest.

__Model 1: RNN + TimeDistributed Dense__
>This model performed better than model 1. Two new layers were added to recurrent layer in `model_1` resulting in model two. We added batch normalization and a time distributed dense layer. The starting loss rate is at 303. There is also a constant decrease in the loss, at every epoch the loss decreases by any value from 1 to 5. The model shows there is no sign of over-fitting. The final loss rate is at 116.

__Model 2: CNN + RNN + TimeDistributed Dense__
>This model’s architecture has an additional level of complexity, by introducing a convolution layer. This model retained batch normalization and a time distributed dense layer as in `model_1`. Again this model performed better than the model before it. Over 20 epochs, There is a constant decrease in the loss value. Starting at 246 and ending at 74. `Model_1` performed better than this model (`model_2`) with a higher loss margin. Also, there are no signs of over-fitting.

__Model 3: Deeper RNN + TimeDistributed Dense__
>This model is a review of the single recurrent layer in `model_0`. This model utilizes a variable number `recur_layers` of recurrent layers. This model also retained batch normalization and a time distributed dense layer as in model_1. The starting loss value at `epoch_1` is 300. The loss value is on a constant decrease over 20 epochs. The final loss value at the 20th epoch is 116. This model has a loss margin of 184, but compared to `model_1`’s 187 loss margin, `model_1` performed better. There are also no signs of over-fitting.

__Model 4: Bidirectional RNN + TimeDistributed Dense__
> This architecture in `model_4` retained only the time distributed dense layer as in `model_1`. The architecture uses a single bidirectional RNN layer, before the Time Distributed dense layer.  There is a constant decrease in the loss value from the 1st epoch up till the 20th epoch. The loss values started at 280 and ends at 118. The loss margin is 162 which is lower than all the previous models. Also, there are no signs of over-fitting.

>Overall, the model with the highest loss margin has the best training performance and that is the second model which is a combination of convolutional layer, recurrent layer and time distributed dense layer. While `model_0` is the pooresr performing model with the lowest loss margin.

> Looking at the validation plot. The loss margins are quite close compared to the training loss margin, as the line overlap each other, However after examining the validation loss values, there are signs of overfitting, as the values flunctuate between higher and lower values. the validation loss of `model_0` is still the lowest (hence it still has a poor performance rate)
