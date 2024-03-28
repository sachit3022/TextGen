[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/H9OQZxGY)
# CSE 849: Deep Learning: Sequence-to-Sequence

## Instructor: Vishnu Boddeti

### Instructions

- Submit solutions by directly editing this markdown file. Submissions in other formats will not be graded.
- **Submission:** Only submissions made through GitHub Classroom will be graded. To receive full credit, make sure to show all the steps of your derivations.
- **Integrity and Collaboration:** You are expected to work on the homework independently. You are not permitted to discuss them with anyone except the instructor. The homework that you hand in should be entirely your work. You may be asked to demonstrate how you got any results you report.
- **Clarifications:** If you have any questions, please look at Piazza first. Other students may have encountered the same problem, which has already been solved. If not, post your question there. We will respond as soon as possible.
- **Editor and Extensions**: It is best to edit this Markdown file using Visual Studio Code. For better support, you can also install the Markdown Preview Enhanced extension.

### Introduction

In this assignment, you will train two attention-based neural machine translation (NMT) models to translate words from English to Pig-Latin. Along the way, you will gain experience with several important concepts in NMT, including *attention,* *transformers*, and *state-space models*.

***Pig Latin:*** It is a simple transformation of English based on the following rules applied on a per-word basis:

- If the first letter of a word is a consonant, then the letter is moved to the end of the word, and the letters "ay" are added to the end. For instance, *team $\rightarrow$ eamtay*.
- If the first letter is a vowel, then the word is left unchanged, and the letters "way" are added to the end: *impress $\rightarrow$ impressway*.
- In addition, some consonant pairs, such as "sh", are treated as a block and are moved to the end of the string together: *shopping $\rightarrow$ oppingshay*.

To translate a whole sentence from English to Pig-Latin, we simply apply these rules to each word independently: *i went shopping $\rightarrow$ iway entway oppingshay*

We want to train a neural machine translation model to implicitly learn the rules of Pig-Latin from (English, Pig-Latin) word pairs. Since the translation to Pig Latin involves moving characters around in a string, we will use character-level neural networks for our model. Since English and Pig-Latin are very similar structurally, the translation task is almost a copy task; the model must remember each character in the input and recall the characters in a specific order to produce the output. This makes it an ideal task to understand the capacity of NMT models.

***Data:*** The data for this task consists of pairs of words $\{(s^{(i)}, t^{(i)})\}_{i=1}^N$ where the source $s^{(i)}$ is an English word, and the target $t^{(i)}$ is its translation in Pig-Latin. 

We provided a small and a large dataset to investigate the effect of dataset size on generalization ability. The small dataset comprises a subset of the unique words from the book “Sense and Sensibility,” by Jane Austen. The vocabulary consists of 29 tokens: the 26 standard alphabet letters (all lowercase), the dash symbol $-$, and two special tokens, start of sentence `<SOS>` and end of sentence `<EOS>`. The dataset contains 3198 unique (English, Pig-Latin) pairs in total; the first few examples are: *$\{$ (the, ethay), (family, amilyfay), (of, ofway), $\dots$ $\}$*

The second dataset is obtained from Peter Norvig's [natural language corpus](https://norvig.com/ngrams/). It contains the top 20,000 most used English words, which is combined with the previous data set to obtain 22,402 unique words. This dataset contains the same vocabulary as the previous dataset.

To simplify the processing of mini-batches of words, the word pairs are grouped based on the lengths of the source and target. Thus, in each mini-batch, the source words are all the same length, and the target words are all the same length. This simplifies the code, as we don’t have to worry about batches of variable-length sequences.

### Q1: Transformer Models (5pts)

***Encoder-Decoder NMT Setup:*** Translation is a sequence-to-sequence problem: in our case, both the input and output are sequences of characters. A common architecture used for seq-to-seq problems is the encoder-decoder model [^sutskever2014sequence], composed of two sequence models, as follows:

![img](assets/attention-rnn-8.svg)

**Scaled Dot-Product Attention:** In the lecture, we introduced Scaled Dot-product Attention used in the transformer models. The function $f$ is a dot product between the linearly transformed query and keys using weight matrices $\mathbf{W}_q$ and $\mathbf{W}_k$:
$$
\begin{aligned}
\tilde{\alpha}_i^{(t)} =& f(\mathbf{Q}_t,\mathbf{K}_i) = \frac{(\mathbf{W}_q\mathbf{Q}_t)^T(\mathbf{W}_k\mathbf{K}_i)}{\sqrt{d}} \\
\alpha_i^{(t)} =& softmax(\tilde{\alpha}^{(t)})_i \\
\mathbf{c}_t =& \sum_{i=1}^T \alpha_i^{(t)}\mathbf{W}_v\mathbf{v}_i
\end{aligned}
$$

where, $d$ is the dimension of the query, and $\mathbf{W}_v$ is the weight matrix corresponding to the values $\mathbf{v}_i$.

- Fill in the forward method in the `MultiHeadAttention` class in `models/transformer.py`. In this part, you will implement both standard "Scaled Dot Attention" for the encoder and "Causal Scaled Dot Attention" for the decoder. Use the `einsum` command to compute the dot product between the batched queries and the batched keys in the forward pass. The following functions are useful in implementing models like this. It is useful to get familiar with how they work. (click the links to jump to the documentation): [einsum - Einops](https://einops.rocks/api/einsum/), [rearrange - Einops](https://einops.rocks/api/rearrange/), [repeat - Einops](https://einops.rocks/api/repeat/) and [reduce - Einops](https://einops.rocks/api/reduce/). For the "Causal Scaled Dot Attention," we must mask the attention for the future steps. You must add `self.neg_inf` to some of the entries in the attention. You may find [torch.tril](https://pytorch.org/docs/stable/generated/torch.tril.html) handy for this part.

- We will now use `MultiHeadAttention` as the building blocks for a transformer [^vaswani2017attention] encoder. The encoder consists of three components (already provided):

    - Positional encoding: Without any additional modifications, self-attention is permutation-equivariant. To encode the position of each word, we add to its embedding a constant vector that depends on its position:
    $$
    \text{pth word embedding} = \text{input embedding} + \text{positional encoding(p)}
    $$
    We follow the same positional encoding methodology described in [^vaswani2017attention]. That is, we use sine and cosine functions:
    $$
    \begin{aligned}
            PE(pos, 2i) =& \sin \frac{pos}{10000^{2i/d_{model}}} \\
            PE(pos, 2i+1) =& \cos \frac{pos}{10000^{2i/d_{model}}} \\
        \end{aligned}
    $$
    - A `MultiHeadAttention` operation
    - An MLP

    Now, complete the forward method of TransformerEncoder. Most of the code is provided except for a few lines. Complete these lines.

- The decoder is similar to the encoder, except it utilizes the "Causal Scaled Dot Attention". The transformer solves the translation problem using layers of attention modules. In each layer, we first apply the "Causal Scaled Dot Attention" self-attention to the decoder inputs, followed by the "Scaled Dot Attention" attention module to the encoder annotations. The output of the attention layers is fed into a hidden layer using GELU activation. The final output of the last transformer layer is passed to `self.out` to compute the word prediction. We add residual connections between the attention and GELU layers to improve the optimization. Now, complete the forward method of `TransformerDecoder`. Again, most of the code is given to you - fill out the few missing lines.

- You will compare the model's performance with respect to the hidden and dataset sizes. Run the Transformer model using hidden size 32 versus 64 and using the small and large dataset (in total, 4 runs) as follows:

Change `hidden_size` in the config file to 32 and then run:

`python3 main.py --config exps/args_piglatin_small_transformer.txt`

`python3 main.py --config exps/args_piglatin_large_transformer.txt`

Change `hidden_size` in the config file to 64 and then run:

`python3 main.py --config exps/args_piglatin_small_transformer.txt`

`python3 main.py --config exps/args_piglatin_large_transformer.txt`

Run these experiments and report the effects of increasing hidden size and dataset size. In particular, how does the model's generalization change with model/dataset size? Are these results what you would expect? In your report, include plots comparing the training and validation losses for the four runs.

- You can modify the transformer configuration as you please, adding more transformer layers or increasing the number of attention heads, optimizer, learning rate scheduler, etc. Report the training and validation loss curves for your modification. What do you observe when training this modified transformer? Is it better than the default settings? Why?

**Deliverables:** Create a section in your report called Transformer. Include the following:

- Python files with code: `models/transformer.py`
- Training/validation plots you generated.
- Your response to the questions above.

### Q2: State-Space Models (5pts)

![](assets/fig1.svg)

A state-space model compresses the input sequence into a fixed-length vector. The predictor is conditioned on this vector to produce the character-by-character translation. The hidden state is typically initialized to $h^{enc}_0=\mathbf{0}$.

Input characters are passed through an embedding layer before they are fed into the state-space model; in our model, we learn a $29\times H$ embedding matrix, where each of the 29 characters in the vocabulary is assigned a $H$-dimensional embedding. At each time step, the state-space model outputs a vector of unnormalized log probabilities given by a linear transformation of the hidden state. When these probabilities are normalized, they define a distribution over the vocabulary, indicating the most probable characters for that time step. The model is trained via a cross-entropy loss between the predicted distribution and ground truth at each time step.

The output vocabulary distribution is conditioned on the previous hidden state and the output token in the previous time step. A common practice used to train NMT models is to feed in the ground-truth token from the previous time step to condition the decoder output in the current step, as shown in the previous question. We don't have access to the ground-truth output sequence at test time, so the state-space model must condition its output on the token it generated in the previous time step, as shown below.

![](assets/fig2.svg)

Specifically, we will implement the state-space model introduced in [[2312.00752\] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arxiv.org)](https://arxiv.org/abs/2312.00752). The Mamba architecture involves a Mamba Block (see below) that is repeated many times.

![img](assets/mamba-architecture.png)

The state-space model in Mamba is parameterized as follows:

$h'(t) = Ah(t) + Bx(t)$

$y(t) = Ch(t) + Dx(t)$​

- **Discretizing the SSM (1pt):** The continuous state-space model must be discretized before implementation. There are multiple choices for discretization, such as Zero-Order Hold, Forward Euler, Backward Euler, Bilinear Interpolation, etc. You will derive the Bilinear interpolation-based discretization. In the general case, for $x'(t)=f(x(t))$, the forward Euler discretizes as $x_{k+1}=x_{k} + \Delta f(x_k)$, and the backward Euler discretizes as $x_{k+1}=x_{k} + \Delta f(x_{k+1})$. The bilinear interpolation discretizes as $x_{k+1} = x_{k} + \Delta (\alpha f(x_k) + (1-\alpha)f(x_{k+1})) \text{ where } 0\leq\alpha\leq 1$. Setting $\alpha=0$ leads to backward Euler, $\alpha=1$ leads to forward Euler, and $\alpha=0.5$ was used in the [^S4] model. Discretize the state space model using this bilinear interpolation.

  $h_k = \bar{A}h_{k-1} + \bar{B}x_k$

  $y_k = \bar{C}h_k + \bar{D}x_k$

  Express the matrices $\bar{A}, \bar{B}, \bar{C}, \bar{D}$ in terms of $A, B, C, D, \Delta$. The Mamba implementation uses the zero-order hold and has already been implemented for you.

- **Parallel Scan (1pt):** As we saw in class, the update to the state space model can be implemented with a simple sequential scan, which has already been implemented for you. In this part, you will implement the parallel scan following the idea in [^heinsen2023scan]

- **Implementing Mamba (2 pts):** Now, you will implement the Mamba Block, from which the rest of the Mamba architecture is constructed. Most of the code has been written for you. Fill in the few missing lines. 

- You will train the Mamba model and compare its performance to the transformer in terms of validation loss, number of parameters, and latency.  Run the Mamba model using hidden size 32 versus 64 and using the small and large dataset (in total, 4 runs) as follows:

Change `hidden_size` in the config file to 32 and then run:

`python3 main.py --config exps/args_piglatin_small_transformer.text`

`python3 main.py --config exps/args_piglatin_large_transformer.text`

Change `hidden_size` in the config file to 64 and then run:

`python3 main.py --config exps/args_piglatin_small_mamba.py`

`python3 main.py --config exps/args_piglatin_large_mamba.py`

- **Bonus (2 pts):** You can modify the Mamba configuration by adding more SSM layers, initialization of $\bar{A}$, optimizer, learning rate scheduler, etc. Report the training and validation loss curves for your modification. What do you observe when training this modified transformer? Is it better than the default settings? Why?

### Submission

- Edited `README.md` file containing your answers to the conceptual questions, plots, and results with explanations.

[^sutskever2014sequence]: Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.
[^cho2014learning]: Kyunghyun Cho, Bart Van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.

[^vaswani2017attention]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.
[^S4]: [[2111.00396\] Efficiently Modeling Long Sequences with Structured State Spaces (arxiv.org)](https://arxiv.org/abs/2111.00396)
[^heinsen2023scan]: [[2311.06281\] Efficient Parallelization of a Ubiquitous Sequential Computation (arxiv.org)](https://arxiv.org/abs/2311.06281)