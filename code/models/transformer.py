# transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, repeat

__all__ = ['Transformer']

class Transformer(nn.Module):
    def __init__(self, hidden_size, num_encoder_layers, num_decoder_layers, num_heads=1, dropout=0.2):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.positional_encodings = nn.Parameter(create_positional_encodings(hidden_size))
        self.encoder = TransformerEncoder(hidden_size, num_encoder_layers, num_heads, dropout, positional_encodings = self.positional_encodings)
        self.decoder = TransformerDecoder(hidden_size, num_decoder_layers, num_heads, dropout, positional_encodings = self.positional_encodings)

    def forward(self, inputs,outputs=None):
        """Forward pass of the Transformer.
  
        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len x hidden_size)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """
        
        enc_attn,annotations = self.encoder(inputs)
        if outputs is None:
            outputs = inputs
        dec_attn,output = self.decoder(inputs, annotations)
    
        return (enc_attn,dec_attn), output
    
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads=1, dropout=0.2,positional_encodings = None):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        
        self.norm = nn.LayerNorm(hidden_size)
        self.norm1 = nn.ModuleList([ nn.LayerNorm(hidden_size)  for _ in range(self.num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden_size)  for _ in range(self.num_layers)])
        
        self.dropout1 = nn.ModuleList([ nn.Dropout(dropout) for _ in range(self.num_layers)])
        self.dropout2 =  nn.ModuleList([  nn.Dropout(dropout) for _ in range(self.num_layers)])
        
        self.self_attentions = nn.ModuleList([MultiHeadAttention(
            hidden_size, num_heads, 'scaled_dot_attention', dropout
            ) for i in range(self.num_layers)])

        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_size, hidden_size),
                                 ) for i in range(self.num_layers)])
        if positional_encodings is not None:
            self.positional_encodings = positional_encodings
        else:
            self.register_buffer('positional_encodings',create_positional_encodings(hidden_size))


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
        
        encoded = inputs + self.positional_encodings[:seq_len,:] # add the positional encodings to the inputs

        annotations = encoded

        #combine them
        attn =[]
        for i in range(self.num_layers):
            
            # 1. normalization layer
            new_annotations = self.norm1[i](annotations)
            
            # 2. self attention layer
            self_attention,new_annotations = self.self_attentions[i](new_annotations,new_annotations,new_annotations)  # batch_size x seq_len x hidden_size
            
            # skip connection with dropout
            annotations = annotations + self.dropout1[i](new_annotations)
            
            # 3. normalization layer
            new_annotations = self.norm2[i](annotations)
            
            # 4. feed forward layer
            new_annotations = self.attention_mlps[i](new_annotations)
            
            # skip connection with dropout
            annotations = annotations + self.dropout2[i](new_annotations)
            attn.append(self_attention)
        

            
        # ------------
        # FILL THIS IN - END
        # ------------
        
        if self.norm:
            annotations = self.norm(annotations)

        # Transformer encoder does not have a last hidden layer.
        return attn,annotations
    
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads=1, dropout=0.2,positional_encodings=None):
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        
        self.norm = nn.LayerNorm(hidden_size)
        self.norm1 = nn.ModuleList([  nn.LayerNorm(hidden_size) for _ in range(num_layers) ] )
        self.norm2 = nn.ModuleList([  nn.LayerNorm(hidden_size) for _ in range(num_layers) ] )
        self.norm3 =  nn.ModuleList([  nn.LayerNorm(hidden_size) for _ in range(num_layers) ] )
        
        self.dropout1 = nn.ModuleList( [nn.Dropout(dropout) for _ in range(num_layers) ] ) 
        self.dropout2 =  nn.ModuleList( [nn.Dropout(dropout) for _ in range(num_layers) ] )
        self.dropout3 =  nn.ModuleList( [nn.Dropout(dropout) for _ in range(num_layers) ] ) 
        
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
        
        if positional_encodings is not None:
            self.positional_encodings = positional_encodings
        else:
            self.register_buffer('positional_encodings',create_positional_encodings(hidden_size))


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
        batch_size, seq_len, hidden_size = inputs.size()
        
        contexts = inputs +  self.positional_encodings[:seq_len,:] # add the positional encodings to the inputs

        self_attn,cross_attn=[],[]

        for i in range(self.num_layers):
            # ------------
            # FILL THIS IN - START
            # ------------
            
            # 1. self attention between the inputs
            self_attention, new_contexts = self.self_attentions[i](contexts,contexts,contexts) # batch_size x seq_len x hidden_size
            
            # 2. skip connection with dropout
            contexts = contexts + self.dropout1[i](new_contexts)
            
            # 3. normalization layer
            contexts = self.norm1[i](contexts)
        
            # 4. attention to the encoder annotations
            cross_attention,new_contexts = self.encoder_attentions[i](contexts,annotations,annotations)  # batch_size x seq_len x hidden_size
            
            # 5. skip connection with dropout
            contexts = contexts + self.dropout2[i](new_contexts)
            
            # 6. normalization layer
            contexts = self.norm2[i](contexts)
            
            # 7. feed forward layer
            new_contexts = self.attention_mlps[i](contexts)
            
            # 8. skip connection with dropout
            contexts = contexts + self.dropout3[i](new_contexts)
            
            # 9. normalization layer
            contexts = self.norm3[i](contexts)

            self_attn.append(self_attention)
            cross_attn.append(cross_attention)
            
            # ------------
            # FILL THIS IN - END
            # ------------

        
        if self.norm is not None:
            contexts = self.norm(contexts)

        return (self_attn,cross_attn),contexts

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_type, dropout=None):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.attention_type = attention_type
        self.neg_inf  = -float("Inf")
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, queries, keys, values):

        q_batch_size, q_context_len, q_hidden_dim = queries.shape
        kv_batch_size, kv_context_len, kv_hidden_dim = keys.shape
        assert q_batch_size == kv_batch_size
        
        

        q = self.Q(queries)
        q = q.transpose(0,1).contiguous().view(q_context_len,q_batch_size*self.num_heads,self.head_size).transpose(0,1).contiguous()
        

        k = self.K(keys)
        k = k.transpose(0,1).contiguous().view(kv_context_len,kv_batch_size*self.num_heads,self.head_size).transpose(0,1).contiguous() 

        v = self.V(values)
        v = v.transpose(0,1).contiguous().view(kv_context_len,kv_batch_size*self.num_heads,self.head_size).transpose(0,1).contiguous()

        
        scaling_factor = torch.rsqrt(torch.tensor(q.shape[-1], dtype= torch.float, device=queries.device))

        attention = scaling_factor * torch.bmm(q,k.transpose(-2,-1)) # B X C X C 
        
        if self.attention_type == 'causal_scaled_dot_attention':
            mask = torch.tril(attention)
            attention[mask==0] = self.neg_inf
        
        attention =F.softmax(attention,dim=-1)  # B X C X C 

        
        if self.dropout is not None:
            attention = self.dropout(attention) 
            
        context = torch.bmm(attention,v) # (B H) X C X C  *  (B H) X C X d -> (B H) X C X d  
        context =context.transpose(1,0).contiguous().view(q_context_len,q_batch_size,q_hidden_dim).transpose(1,0) # reorder dimensions to b x q x d
        context = self.O(context)
        _,c1,c2 = attention.shape
        return attention.view(q_batch_size,-1,c1,c2),context
    
def create_positional_encodings(hidden_size, max_seq_len=100):
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
    #requires grad is false
    return pos_encodings