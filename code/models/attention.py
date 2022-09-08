# gru.py

import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()

        self.hidden_size = hidden_size

        # Create a two layer fully-connected network. Hint: Use nn.Sequential
        # hidden_size*2 --> hidden_size, ReLU, hidden_size --> 1

        # ------------
        # FILL THIS IN - START
        # ------------

        # self.attention_network = ...

        # ------------
        # FILL THIS IN - END
        # ------------

        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries, keys, values):
        """The forward pass of the additive attention mechanism.

        Arguments:
            queries: The current decoder hidden state. (batch_size x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x 1 x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The attention_weights must be a softmax weighting over the seq_len annotations.
        """
        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------

        # expanded_queries = ...
        # concat_inputs = ...
        # unnormalized_attention = self.attention_network(...)
        # attention_weights = self.softmax(...)
        # context = ...

        # ------------
        # FILL THIS IN - END
        # ------------
        return context, attention_weights

class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x k x seq_len)

            The output must be a softmax weighting over the seq_len annotations.
        """

        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------

        # if queries.dim() != 3:
        #     queries = ...
        # q = ...
        # k = ...
        # v = ...
        # unnormalized_attention = ...
        # attention_weights = ...
        # context = ...

        # ------------
        # FILL THIS IN - END
        # ------------
        return context, attention_weights.transpose(1,2)

class CausalScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CausalScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size
        self.neg_inf = -1e7

        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x k)

            The output must be a softmax weighting over the seq_len annotations.
        """

        # ------------
        # FILL THIS IN - START
        # ------------
        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.

        # if queries.dim() != 3:
        #     queries = ...
        # q = ...
        # k = ...
        # v = ...
        # unnormalized_attention = ....
        # mask = ...
        # attention_weights = ...
        # context = ...

        # ------------
        # FILL THIS IN - END
        # ------------
        return context, attention_weights.transpose(1,2)
