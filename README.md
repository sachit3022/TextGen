# Writeup

## Introduction

I have closely followed the "Attention is All You Need" [^vaswani2017attention] and GPT [^GPT] papers to implement a transformer. The goal is to understand the inner workings of transformers to the extent that We try to improve performance on the Pig Latin task.

One of the core components of transformers is the attention mechanism. Each layer of the transformer comprises three such attention blocks: encoder self-attention, decoder self-attention, and cross-attention. Understanding these attention components is crucial to comprehend how information is transmitted from the source to the target. In the visualization section, We demonstrate that the current word of the encoder will have information only from the previous word in the cross-attention, and this will help cross attention module pick the next word given the current word.

In all NLP tasks, embeddings play a crucial role. We have observed this phenomenon in [HW-1](https://github.com/msu-deep-learning/homework-1-sachit3022/). Do we observe similar clusters for this task? Do vowels and consonants form clusters? Are "-" and "[EOS]" closer to each other? However, we don't observe such a phenomenon in this task. Wr hypothesize that the model's embedding space is large compared to the vocabulary dimension. It has the capacity to make all tokens orthogonal to each other. Maybe we can observe this phenomenon when we significantly reduce hidden dimensions.

Can we make any other modifications to improve the performance of the task? The task for Pig Latin relies on positions, as we can observe from the attention maps. So, instead of having cosine positional embeddings, can we train them to improve performance? We demonstrate that training positional embeddings increases performance on this task.

Finally, we perform an ablation study and show which hyperparameters are more influential.

## Preliminaries

**Loss Computation:** We measure the loss values per character. This makes the interpretation of loss simpler. Suppose the loss value is $-\log(\frac{1}{29}) = \log{29}$. This indicates on average model is predicting characters at random. We use this notion because the loss per word is often misleading when there is an imbalance in errors with respect to word length. We show that this is the case, where the model makes mistakes for longer words compared to smaller ones. This leads to a larger value than the simpler metric of average across words.

**Accuracy:** We compute how many words are predicted correctly, which provides a true metric. Even if the architecture or model is wrong, because train and validation sets are sampled from the same distribution, the loss will be very low but the model will output garbage.

## Interpretability of Transformers

For understanding the visualisation, we will use a smaller model, 2 encoders, 2 decoders and just 2 attention heads, we will keep the embeddings as 64.  We will study 3 different words which represents different type of translations in piglatin


(1) brown -> ownbray

(2) conditioning -> onditioningcay

(3) is -> isway

We use [^bertviz] to visualise the attention.

|   | brown  | conditioning  | is  | 
|---|---|---|---|
| Encoder Self attention  | ![encoder_brown](assets/image.png)  | ![encoder_conditioning](assets/image-2.png)  | ![encoder_is](assets/image-1.png)  |  
| Cross Attention  | ![cross_brown](assets/image-5.png)  | ![cross_conditioning](assets/image-3.png)  |![cross_is](assets/image-6.png)| 

### Explanation of the Attention Module

One of the heads in the attention mechanism has the specific job of retaining information from the previous token. This is essential for the decoder to predict the next word accurately. Furthermore, when the "[EOS]" token is reached, this head should contain information about the first character. If the two vectors are exactly the same, the dot product is large (close to 1).

This can be depicted visually as follows:

![Image showing the behavior of head 0](assets/head.001.jpeg)

The image illustrates the behavior of head 0. Encoder attention will only retain information about the previous token, making it easier for the decoder to predict the next one as the dot product with the same vectors is maximized.

![Image showing the behavior of head 1](assets/head.002.jpeg)

This image demonstrates the behavior of head 1. In this case, the encoder doesn't learn anything, or you can think of it as learning identity. The embeddings of "[SOS]" are learned such that they are closer to the first vowel (position + vowel embedding). This head aids in predicting the first character. For subsequent character predictions, head 0 will handle them.

Cross attentions are perfect in a sense that "[SOS]" attends to the first vowel, and then the next character will attend to the character that it will be predicting. If the word is the last word, it automatically points to the first word and continues generating until reaching the "[SOS]" matched token, then looks at `EOS`, and generates `ay`.


## visualisation of embeddings

![Embedding visualisation](assets/emb.png)
Expected a cluster around the vowels and consonants however no such pattern has emerged in learning embeddings, that implies it leart these embeddings differents and it memorises a,e,i,o,u differently.


## Abalation study

In our experimentation, we found that there isn't much tuning required for the architecture itself. Interestingly, we concluded that for this task, the transformer architecture is relatively easier to tune. Even the base model, with no tuning, achieves a validation accuracy of 91%. 

When we increase the model's capacity, the performance initially improves significantly, but eventually, it saturates because the training reaches perfect accuracy. However, a heavily tuned model Mamba performs slightly better than heavily tuned transformer.

In our ablation study, we aim to investigate the importance of different modules within the transformer architecture and how their significance changes with the scale of these modules.

In the following experiment, we consider the model with the below parameters as a base model. We only change one hyperparameter at one time 

| hyper parameter | val |
|---|---|
| hidden diamension | 64 |
| head size | 16 |
| num of encoder layers | 2 |
| num of decoder layers | 2 |
| batch size | 128 |
| learning rate | 3e-3 |
| number of epochs | 100 |


| Model |  val loss | train loss| val accuracy  | train accuracy|
|---|---|---|---|---|
| Base |0.043  | 0.050      |   0.909  |  0.957   |

Sample mistakes made by the Base model

| True value | Wrong prediction |
|---|---|
|acrossway | acrosssway|
|adieusway | adieussway|
|ankleway | ankleeway|
|oodgay-umouredhay|oodgay-mouray-odha|

Most identified mistakes involve starting with a vowel and not knowing the end, resulting in repeating the final character.


**Observation: Mistakes are due to inefficient encoding of postion**

Consider the task of Pig Latin transformation. To predict the next character, we need information from the previous character. Therefore, time plays a more important role than character embeddings.

The attention maps also support this conclusion. Now, if the positional embeddings are crucial, what if we remove them? Will there be a significant drop in accuracy?

There is indeed a substantial drop in accuracy when we drop the positional embeddings, as this task relies heavily on remembering the positional embeddings. Having them improves accuracy by 40%.

### Impact of Training Positional Embeddings

As we establish that positional encodings are important, does training them improve performance?

| Output Pos Embeddings | Validation Loss | Train Loss | Validation Accuracy | Train Accuracy |
|-----------------------|-----------------|------------|---------------------|----------------|
| Disable               | 0.232           | 0.114      | 0.576               | 0.858          |
| Train                 | 0.033           | 0.004      | 0.962               | 1.000          |

Training positional encodings significantly improves performance for this task because word shifts are based more on positional embeddings than on the embeddings of the words. This results in an improvement of 5.83%.


Lets look at the learning embeddings ( initialised with the embeddings from attention is all you need paper.)

![Positional Embedding visualisation](assets/pos_emb.png)
### Impact of Number of Heads

We will analyze the performance with the increase in head size:

| Head Size | Validation Loss | Train Loss | Validation Accuracy | Train Accuracy |
|-----------|-----------------|------------|---------------------|----------------|
| 4         | 0.070           | 0.049      | 0.853               | 0.970          |
| 8         | 0.059           | 0.034      | 0.912               | 0.971          |
| 16        | 0.043           | 0.050      | 0.909               | 0.957          |

The best performance we have found is with 8 heads, each with a dimension of 8, resulting in a hidden dimension of 64. We found that balancing the number of heads and the dimension of each head is crucial to improving performance.

### Impact of Hidden Dimension

| Dimension Size | Validation Loss | Train Loss | Validation Accuracy | Train Accuracy |
|----------------|-----------------|------------|---------------------|----------------|
| 32             | 0.106           | 0.137      | 0.779               | 0.923          |
| 64             | 0.043           | 0.050      | 0.909               | 0.957          |

Decreasing the dimension of the hidden size results in a drop in performance.

### Impact of Number of Layers

| Layers | Validation Loss | Train Loss | Validation Accuracy | Train Accuracy |
|--------|-----------------|------------|---------------------|----------------|
| 2      | 0.086           | 0.071      | 0.842               | 0.935          |
| 4      | 0.043           | 0.050      | 0.909               | 0.957          |
| 8      | 0.120           | 0.016      | 0.812               | 0.984          |

The encoder and decoder layers of 2 each give better performance. Increasing the layers starts to increase the training performance but deteriorates the validation performance.

### Engineering Improvements

**Accumulated Gradients:** To ensure stochastic gradient convergence to gradient descent, we train with accumulated gradients over 5 epochs to account for the violation of the assumption of randomly sampled samples in a single batch, as our single batch contains identical lengths.

## The Ultimate Beast Model

With 64 hidden dimensions, 8 attention heads, and 2 layers each for encoder and decoder, training the positional encoders. 

A few small modifications were made to improve performance, such as adjusting the scheduler. We observed that the ReduceLROnPlateau scheduler lags behind in training, as it trains one more epoch on high learning rate and the threshold is harder to tune, with increasing the threshold having unstudied effects on performance. Therefore, we opted for CosineAnnealing as a scheduler, as used by the GPT-3 paper, training for 100 epochs and with a weight decay of 0.01. The only difference from GPT is that we use accumulated gradients to account for variation in size.

The only change made from going from small to large is the number of layers, from 2 each to 4 each.

| Model                | Validation Loss | Train Loss | Validation Accuracy | Train Accuracy |
|----------------------|-----------------|------------|---------------------|----------------|
| Beast Small (143K)   | 0.025           | 0.004      | 0.970               | 1.000          |
| Beast Large (276K)   | 0.036           | 0.006      | 0.964               | 1.000          |

The Beast Small model makes only 3 mistakes:

| Latin            | True Pig Latin Value | Wrong Translation       |
|------------------|----------------------|-------------------------|
| pleasure-grounds| easureplay-oundsgray | easureplay-ouredsplay   |
| strengthening    | engtheningstray      | engtheneningstra        |
| thoughtfulness   | oughtfulnessthay     | oughtfulneststhay       |

### Training Progress

| Metric (40 steps = 1 epoch, 40 * 100 = 4K) | Validation | Training |
|--------------------------------------------|------------|----------|
| Loss                                       | ![train_loss](assets/train_loss.png) | ![val_loss](assets/val_loss.png) |
| Accuracy                                   | ![train_acc](assets/train_acc.png)   | ![val_acc](assets/val_acc.png)   |
| Learning Rate                              |            | ![lr](assets/lr.png)             |



Few more corrections to the code:
1. in pytorch lightning `scheduler` should be named to `lr_scheduler`
2. self.model.eval() in generate else the test cases after the training will still be in train mode.
3. accumulate gradients.


# Mamba 




## Introduction

Similar to transformer, We study various properties of Mamba architecture.

The core block of the Mamba architecture, or any RNN architecture, is the state transition matrix. In the experiments below, we study how the state transition matrix, parameterized by $A$ and $\nabla$, is responsible for making predictions.

One of the key takeaways from the experiments we perform on Mamba is that it behaves exactly like an RNN. The entire context information is stored in the "[EOS]" token, and the decoder starts retrieving the information from this token.

We then study how well P-SCAN can replace S-SCAN. P-SCAN takes exponentials and then computes the log, which has a huge impact if the values are larger in magnitude. This becomes a serious issue when the context length increases.

## Analysis of State Transition Matrix A

One of the important attributes to understand is the state transition matrix, A. We will analyze the properties of it and start with the base case as mentioned in the codebase. We aim to improve matrix A for this task.

We initialize the transition matrix to -I matrix, which means initializing A_log to 0. This implies that the state transition ($e^{A\nabla} = e^{-\nabla}$) is entirely controlled by $\nabla$. We call this setting "only $\nabla$". We try to observe what $\nabla$ will pick up.

We will visualize the $\nabla$ as well as the weighted average of character position, which character is getting more weight. As it will be weight decay, we observe a few things that make it very similar to RNNs.

### Deriving What Happens When A is -I

From the zeroth-order hold:
$$\begin{align*}
h_t &= e^{A\nabla_t}h_{t-1} + x'_t \\
    &= e^{-\nabla_t}h_{t-1} + x'_t \\
    &= e^{-\nabla_t}h_{t-1} + x'_t \\
    &= e^{-\nabla_t}x'_{t-1} + x'_t + e^{-\nabla_t -\nabla_{t-1} } x'_{t-1} \cdots \\
    &= \sum_t w_t x'_{t}
\end{align*}$$
where $w_t = e^{-\sum_t \nabla_t}$.

We will plot the $w_t$ for a few words and understand what is done by SSMs.

|   | Conditioning | easureplay-oundsgray | is  | 
|---|---|---|---|
| Weight given to each character (cumulative sum) | ![weight_c](assets/weight_c.png)  | ![weight_p](assets/weight_p.png)  | ![weight_i](assets/weight_i.png)  |  
| dt vector  | ![dt_vec_c](assets/dt_vec_c.png)  | ![dt_vec_p](assets/dt_vec_p.png)  |![dt_vec_i](assets/dt_vec_i.png)| 

As you can see, the weight stops at the "[EOS]" token and almost all of them have 0 weight after it. Therefore, we can assume that the SSMs add the entire context word to the EOS token and start decoding it for this task. However, this behavior might change for long-range arenas, as fixing A will not give us better performance. However, for this task, SSM behaves equivalently to RNNs.


One small advantage of RNN is adaptability here in case - it also stops at -

| Model |  val loss | train loss| val accuracy  | train accuracy|comments|
|---|---|---|---|---|---|
| $\nabla$ training and A = -I|0.168|   0.003       |  0.833       | 0.997   |images in the above table|
| A training and $\nabla$ as Pos embedding  | NA |    NA     |   NA      | NA |NA|
| A training and $\nabla$ as constant time step  | 0.157 |  0.002     |   0.839     | 0.997   ||
| A= -I and  $\nabla$ as Pos embedding  | NA |    NA     |   NA      | NA |NA|

NA - experiments didnot turn out to be important.

These set of experiments answer all the questions for us in SSMs.

We will start as usual, Tuning learning rate is very esential for getting the best performance from Mamba. We will tune the learning rate from 3e-4, however we found the learning rate is very small for mamba and we increase it to 3e-3 and found it to be ideal for out task. We use all the other parameters as default. And try to push the accuracy of the base model.


### Discretisation


State space models are very sensitive to the discretization method used. Therefore, we considered various discretizations and found that bilinear interpolation gives us the best performance.

| Model                                          | Validation Loss | Train Loss | Validation Accuracy | Train Accuracy |
|------------------------------------------------|-----------------|------------|---------------------|----------------|
| Approximate zeroth-order hold [^mamba]         | 0.237           | 0.003      | 0.654               | 0.996          |
| Exact zeroth-order hold (Equation 3) [^mamba]  | 0.252           | 0.004      | 0.638               | 0.998          |
| Bilinear interpolation [^S4]                   | 0.175           | 0.002      | 0.833               | 0.998          |


### P-scan vs S-scan

We will compare the time improvements achieved by implementing P-SCAN vs S-SCAN with `torch.compile`. We observe that P-SCAN performs significantly better, but there is a difference in performance when we run without compilation. Additionally, we are not sure about the behavior of `torch.compile`, but when we run with compilation, we see a significant boost in time when P-SCAN is used compared to S-SCAN.



![Comparing test time](assets/time.png)

**Precision of P-SCAN and S-SCAN**

As P-SCAN is a parallel implementation of S-SCAN, precision becomes crucial due to the logarithmic and exponential operations involved, especially when dealing with large numbers. Additionally, we approximate $\log{0} = -12$, which leads to non-exact solutions. We measure how P-SCAN and S-SCAN deviate. For smaller context lengths and logits distributed as $\mathcal{N}(0,I)$, the error grows with increasing context length as well as deviating from the normal behavior of logits.


![precision of Psacn and Sscan](assets/precision.png)



### Experimental details

Final comments on  Mamba architecture, Base case

| Model |  val loss | train loss| val accuracy  | train accuracy|
|---|---|---|---|---|
| Base | 0.237   | 0.003    |   0.654 |  0.996     |

| Model |  val loss | train loss| val accuracy  | train accuracy|
|---|---|---|---|---|
| Approx  zeroth order hold [^mamba] | 0.107   | 0.002    |   0.818  | 0.997  |
| Exact zeroth order hold (equation 3) [^mamba]|0.142 | 0.001     |     0.808|  0.998     |
| bi linear interpolation [^S4]| 0.119  | 0.002    |   0.873 | 0.998    |


From here we will perform our experiments with bi linear interpolation as the discreetisation step.

## Impact of hidden diamensions

Other minor experiments that didnot lead significant outcomes
(32,2) v_num: 153.000 val_loss: 0.153 val_acc: 0.844 train_loss: 0.011 train_acc: 0.996   
(64 1) val_loss: 0.167 val_acc: 0.867 train_loss: 0.004 train_acc: 0.997   

We will train the architecture with default the  

| hyper parameter | val |
|---|---|
| hidden diamension | 64 |
| kernel size | 16 |
| expansion factor | 2 |
| num of layers | 2 |
| batch size | 128 |
| learning rate | 3e-3 |
| Scheduler |Reduce on plateau|
| number of epochs | 100 |

On small change made to the architecture is we devide delta by $\sqrt{\textit{hidden size}}$ This bring stability to training ( credits: Ramin Akbari)


| Model |  val loss | train loss| val accuracy  | train accuracy|
|---|---|---|---|---|
| Mamba small ( 162 K  )| 0.144  | 0.002    |  0.870 | 0.998     |
| Mamba large ( 276 K ) | 0.015  | 0.000  |   0.983  |  0.998  |

More details on training

| Metric (40 steps = 1epoch 40*100 = 4K)   | validation  | training  |
|---|---|---|
|Loss |![train_loss](assets/mamba_train_loss.png)  | ![val_loss](assets/mamba_val_loss.png)|
|Accuracy|![train_acc](assets/mamba_train_acc.png) | ![val_acc](assets/mamb_val_acc.png)|
| Learning rate | | ![lr](assets/mamba_lr.png)|


### Submission

- Edited `README.md` file containing your answers to the conceptual questions, plots, and results with explanations.

[^sutskever2014sequence]: Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.

[^cho2014learning]: Kyunghyun Cho, Bart Van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.

[^vaswani2017attention]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.

[^S4]: [2111.00396\] Efficiently Modeling Long Sequences with Structured State Spaces (arxiv.org)](https://arxiv.org/abs/2111.00396)

[^heinsen2023scan]: [2311.06281\] Efficient Parallelization of a Ubiquitous Sequential Computation (arxiv.org)](https://arxiv.org/abs/2311.06281)

[^bertviz]: [Vig, J. (2019). A multiscale visualization of attention in the transformer model. arXiv preprint arXiv:1906.05714.](https://arxiv.org/abs/1906.05714)

[^GPT]: [ Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
