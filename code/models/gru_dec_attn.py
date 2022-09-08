# gru_enc_dec_attn.py

import torch
import torch.nn as nn
from .gru import MyGRUCell
from .attention import *

class GRUAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=None, attention_type='scaled_dot'):
        super(GRUAttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.gru = MyGRUCell(input_size=hidden_size*2, hidden_size=hidden_size)
        if attention_type == 'AdditiveAttention':
          self.attention = AdditiveAttention(hidden_size=hidden_size)
        elif attention_type == 'ScaledDotAttention':
          self.attention = ScaledDotAttention(hidden_size=hidden_size)

        self.out = nn.Linear(hidden_size, vocab_size)


    def forward(self, inputs, annotations, hidden_init):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
            hidden_init: The final hidden states from the encoder, across a batch. (batch_size x hidden_size)

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """

        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        hiddens = []
        attentions = []
        h_prev = hidden_init
        for i in range(seq_len):
            # You are free to follow the code template below, or do it a different way,
            # as long as the output is correct.

            # ------------
            # FILL THIS IN - START
            # ------------

            # embed_current = ...  # Get the current time step, across the whole batch
            # context, attention_weights = self.attention(...)  # batch_size x 1 x hidden_size
            # embed_and_context = ....  # batch_size x (2*hidden_size)
            # h_prev = self.gru(...)  # batch_size x hidden_size

            # ------------
            # FILL THIS IN - END
            # ------------
            hiddens.append(h_prev)
            attentions.append(attention_weights)

        hiddens = torch.stack(hiddens, dim=1) # batch_size x seq_len x hidden_size
        attentions = torch.cat(attentions, dim=2) # batch_size x seq_len x seq_len

        output = self.out(hiddens)  # batch_size x seq_len x vocab_size
        return output, attentions
