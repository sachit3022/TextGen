# transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

__all__ = ['Transformer']

class Transformer(nn.Module):
    def __init__(self, hidden_size, num_encoder_layers, num_decoder_layers, num_heads=1, dropout=0.2):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(hidden_size, num_encoder_layers, num_heads, dropout)
        self.decoder = TransformerDecoder(hidden_size, num_decoder_layers, num_heads, dropout)

    def forward(self, inputs):
        """Forward pass of the Transformer.
  
        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len x hidden_size)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """
        
        annotations = self.encoder(inputs)
        output = self.decoder(inputs, annotations)
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads=1, dropout=0.2):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        
        self.norm = nn.LayerNorm(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.self_attentions = nn.ModuleList([MultiHeadAttention(
            hidden_size, num_heads, 'scaled_dot_attention', dropout
            ) for i in range(self.num_layers)])

        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_size, hidden_size),
                                 ) for i in range(self.num_layers)])

        self.positional_encodings = create_positional_encodings(hidden_size)

    def forward(self, inputs):
        """Forward pass of the encoder Transformer.

        Arguments:
            inputs: Input embeddings across a batch for all time steps in the sequence. (batch_size x seq_len x hidden_size)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
        """

        batch_size, seq_len, hidden_size = inputs.size()
        # ------------
        # FILL THIS IN - START
        # ------------

        # Add positional embeddings from create_positional_encodings. (a'la https://arxiv.org/pdf/1706.03762.pdf, section 3.5)
        self.positional_encodings = ...
        encoded = ... # add the positional encodings to the inputs

        annotations = encoded

        for i in range(self.num_layers):
            
            # 1. normalization layer
            new_annotations = ...
            
            # 2. self attention layer
            new_annotations = ...  # batch_size x seq_len x hidden_size
            
            # skip connection with dropout
            annotations = annotations + self.dropout1(new_annotations)
            
            # 3. normalization layer
            new_annotations = ...
            
            # 4. feed forward layer
            new_annotations = ...
            
            # skip connection with dropout
            annotations = annotations + self.dropout2(new_annotations)
        # ------------
        # FILL THIS IN - END
        # ------------
        
        if self.norm:
            annotations = self.norm(annotations)

        # Transformer encoder does not have a last hidden layer.
        return annotations
    
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads=1, dropout=0.2):
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        
        self.norm = nn.LayerNorm(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.self_attentions = nn.ModuleList([MultiHeadAttention(
            hidden_size, num_heads, 'causal_scaled_dot_attention', dropout
            ) for i in range(self.num_layers)])

        self.encoder_attentions = nn.ModuleList([MultiHeadAttention(
            hidden_size, num_heads, 'scaled_dot_attention', dropout
                                 ) for i in range(self.num_layers)])
        
        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_size, hidden_size),
                                 ) for i in range(self.num_layers)])

    def forward(self, inputs, annotations):
        """Forward pass of the Transformer decoder.

        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """

        contexts = inputs
        for i in range(self.num_layers):
            # ------------
            # FILL THIS IN - START
            # ------------
            
            # 1. self attention between the inputs
            new_contexts = ... # batch_size x seq_len x hidden_size
            
            # 2. skip connection with dropout
            contexts = contexts + self.dropout1(new_contexts)
            
            # 3. normalization layer
            contexts = ...
            
            # 4. attention to the encoder annotations
            new_contexts = ...  # batch_size x seq_len x hidden_size
            
            # 5. skip connection with dropout
            contexts = contexts + self.dropout2(new_contexts)
            
            # 6. normalization layer
            contexts = ...
            
            # 7. feed forward layer
            new_contexts = ...
            
            # 8. skip connection with dropout
            contexts = contexts + self.dropout3(new_contexts)
            
            # 9. normalization layer
            contexts = ...
            
            # ------------
            # FILL THIS IN - END
            # ------------

        if self.norm is not None:
            contexts = self.norm(contexts)

        return contexts

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_type, dropout=None):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.attention_type = attention_type
        self.neg_inf  = float("Inf")
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, queries, keys, values):
        
        if queries.dim() != 3:
            #why would this be possible?    
            queries = queries.unsqueeze(1)
            keys = keys.unsqueeze(1)
            values = values.unsqueeze(1)
            raise ValueError("Input shape is not correct")
        
        q = self.Q(queries) 
        q = q.view(q.shape[0],self.num_heads*self.hidden_size,self.head_size ).transpose(0,1)
        k = self.K(keys)
        k = q.view(k.shape[0],self.num_heads*self.hidden_size,self.head_size ).transpose(0,1)
        v = self.V(values)
        v = v.view(v.shape[0],self.num_heads*self.hidden_size,self.head_size ).transpose(0,1)

        
        scaling_factor = torch.rsqrt(torch.tensor(q.shape[-1], dtype= torch.float, device=queries.device))
        
        attention = scaling_factor * torch.bmm(q,k.transpose(1,2))
        
        if self.attention_type == 'causal_scaled_dot_attention':
            mask = torch.tril(attention)
            attention[mask==0] = -1*self.neg_inf
        
        attention =F.softmax(attention,dim=-1)
        
        if self.dropout is not None:
            attention = self.dropout(attention)
        
        context = ... #write this part
        context = ... # reorder dimensions to b x q x d
        return context
    
def create_positional_encodings(hidden_size, max_seq_len=1000):
    """Creates positional encodings for the inputs.

    Arguments:
        max_seq_len: a number larger than the maximum string length we expect to encounter during training

    Returns:
        pos_encodings: (max_seq_len, hidden_dim) Positional encodings for a sequence with length max_seq_len.
    """
    pos_indices = torch.arange(max_seq_len)[..., None]
    dim_indices = torch.arange(hidden_size//2)[None, ...]
    exponents = (2*dim_indices).float()/(hidden_size)
    trig_args = pos_indices / (10000**exponents)
    sin_terms = torch.sin(trig_args)
    cos_terms = torch.cos(trig_args)

    pos_encodings = torch.zeros((max_seq_len, hidden_size))
    pos_encodings[:, 0::2] = sin_terms
    pos_encodings[:, 1::2] = cos_terms

    return pos_encodings