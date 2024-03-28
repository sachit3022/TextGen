# mamba.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum

__all__ = ['Mamba']

def complex_log(input, eps=1e-12):
    eps = input.new_tensor(eps)
    real = input.abs().maximum(eps).log()
    imag = (input < 0).to(input.dtype) * torch.pi
    return torch.complex(real, imag)

class Mamba(nn.Module):
    def __init__(self, hidden_size, kernel_size, expansion_factor, dt_rank, num_layers):
        super(Mamba, self).__init__()
        # full Mamba model
        self.layers = nn.ModuleList([ResidualBlock(hidden_size, kernel_size, expansion_factor, dt_rank) for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_size)
    
    def forward(self, inputs):
        """Forward pass of S4.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """
        
        for layer in self.layers:
            inputs = layer(inputs)
        inputs = self.norm(inputs)
        
        return inputs
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, expansion_factor, dt_rank):
        super(ResidualBlock, self).__init__()
        
        # The main Mamba block.
        self.ssm = MambaBlock(hidden_size, kernel_size, expansion_factor, dt_rank)
        self.norm = RMSNorm(hidden_size)
        
    def forward(self, inputs):
        """Forward pass of MambaBlock.
        """        
        inputs = self.ssm(self.norm(inputs)) + inputs
        return inputs

class MambaBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, expansion_factor, dt_rank):
        super(MambaBlock, self).__init__()
        
        # the core input selective SSM model in Mamba
        
        self.hidden_size = hidden_size
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(hidden_size / 16)
        else:
            self.dt_rank = dt_rank
        
        # expanded hidden size
        self.expanded_hidden_size = hidden_size * expansion_factor
        
        # linear layer to expand hidden state
        self.expanded_hidden_state = nn.Linear(hidden_size, self.expanded_hidden_size * 2)
        
        # linear layer to compress back to the original hidden state
        self.output_state = nn.Linear(hidden_size * expansion_factor, hidden_size)
        
        # linear layer to predict dt_rank, B and C
        self.param_linear = nn.Linear(self.expanded_hidden_size, self.dt_rank + hidden_size * 2,bias = False)
        
        # linear layer to project dt_rank to dt
        self.dt_linear = nn.Linear(self.dt_rank, self.expanded_hidden_size)
        
        # 1D convolutional layer
        self.conv1d = CausalConv1d(self.expanded_hidden_size, kernel_size)

        # we initialize the state space model parameters A and D
        A = repeat(torch.ones((hidden_size)), "n -> d n", d=self.expanded_hidden_size)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log.requires_grad = True

        self.D = nn.Parameter(torch.ones((self.expanded_hidden_size)))
        self.D.requires_grad = True
        #self.A_log._no_weight_decay = True
        #self.D._no_weight_decay = True
        

    def sscan(self, x, delta, A, B, C, D):
        
        b, l, d = x.size()
        n = A.size(1)
        #A = torch.ones(A.shape,requires_grad=False, device=x.device, dtype=x.dtype)
        deltaA = einsum(delta, A, 'b l d, d n -> b l d n')
        deltaAExp = torch.exp(deltaA)
        
       
        deltaBx = einsum(delta, B, x, 'b l d, b l n, b l d -> b l d n')
        #deltaBx = einsum((deltaAExp  -1 )/deltaA  ,olddeltaBx, 'b l d n,b l d n -> b l d n')

        h = torch.zeros(b, d, n, device=x.device, dtype=x.dtype)
        
        # forward pass of the state space model        
        ylist = []
        for i in range(l):
            h = deltaAExp[:, i] * h + deltaBx[:, i]
            ylist.append(einsum(h, C[:,i], 'b d n, b n -> b d'))
        
        y = torch.stack(ylist, dim=1)
        y = y + D * x
        
        return y
    
    def pscan(self, x, delta, A, B, C, D):
        b, l, d = x.size()
        n = A.size(1)
        deltaA = einsum(delta, A, 'b l d, d n -> b l d n')
        deltaAExp = torch.exp(deltaA)
        
       
        deltaBx = einsum(delta, B, x, 'b l d, b l n, b l d -> b l d n')
        #deltaBx = einsum((deltaAExp  -1 )/deltaA  ,olddeltaBx, 'b l d n,b l d n -> b l d n')

        #A_diag,B -> bldn
        a_t_star= torch.cumsum(deltaA,dim=1)
        x_0 = torch.ones(b, 1, d, n, device=x.device, dtype=x.dtype)*1e-12
        y = (torch.cumsum(torch.cat((complex_log(x_0),complex_log(deltaBx)-a_t_star),dim=1).exp(),dim=1)[:,1:].log() + a_t_star).exp().real
        y = einsum(y, C, 'b l d n, b l n -> b l d')
        y = y + D * x
        return y



    
    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        dbc = self.param_linear(x) # linear layer to predict delta, B and C
        delta, B, C = dbc.split([self.dt_rank, self.hidden_size, self.hidden_size], dim=-1)
        delta = F.softplus(self.dt_linear(delta)) # want delta to be non-negative
        
        if hasattr(self, "pscan"):
            y = self.pscan(x, delta, A, B, C, D)
        else:
            y = self.sscan(x, delta, A, B, C, D)
        
        return y


    def forward(self, x):
        """Forward pass of SSM.

        Arguments:
            x: Input signal across a batch for all time steps in the sequence. (batch_size x seq_len x input_size)

        Returns:
            output: The output computed for each step of the input sequence. (batch_size x seq_len x output_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """
        # ------------
        # FILL THIS IN - START
        # ------------
        
        x_and_skip = self.expanded_hidden_state(x) # batch_size x seq_len x expanded_hidden_size * 2
        x, skip = torch.split(x_and_skip, self.expanded_hidden_size, dim=2)
        
        x = self.conv1d(x) # 1D convolution
        x = F.silu(x) # SILU activation
    
        y =  self.ssm(x) * F.silu(skip) # gating mechanism
        
        output = self.output_state(y)
        
        # ------------
        # FILL THIS IN - END
        # ------------
        
        return output

class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1)
        self.conv1d = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            padding=self.padding)
        
    def forward(self, x):
        
        b, l, d = x.size()
        
        # x is going to be batch_size x seq_len x dim, we reshape it to batch_size x dim x seq_len
        x = rearrange(x, 'b l d -> b d l')
        
        # run 1d convolution
        x = self.conv1d(x)
        
        # keep only the first l values from the start
        x = x[:, :, 0:l]
        
        # reshape it back to batch_size x seq_len x dim
        x = rearrange(x, 'b d l -> b l d')
        
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output



if __name__ == '__main__':
    
    b = 16
    l = 32
    d = 64
    n = 64
    
    x = torch.randn(b, l, d)
    A = F.softmax(torch.randn(d, n),dim=-1)
    B = torch.randn(b, l, d)
    C = torch.randn(b, l, d)
    D = torch.randn(b, l, d)
    delta = torch.randn(x.size())
    
    mamba = MambaBlock(d, 4, 2, "auto")
    #y = mamba(x)
    y1 = mamba.sscan(x, delta, A, B, C, D)
    y2 = mamba.pscan(x, delta, A, B, C, D)
    print((y1-y2).abs().max())
    print(torch.allclose(y1, y2, atol=1e-4))