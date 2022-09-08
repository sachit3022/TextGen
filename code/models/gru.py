# gru.py

import torch
import torch.nn as nn

class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------

        # self.Wr = nn.Linear(...)
        # ...
        # ...
        # ...

        # ------------
        # FILL THIS IN - END
        # ------------


    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """

        # ------------
        # FILL THIS IN - START
        # ------------

        # z = ...
        # r = ...
        # g = ...
        # h_new = ...

        # ------------
        # FILL THIS IN - END
        # ------------
        return h_new
