"""
Model definition.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from icecream import ic


EOS = '<EOS>'
SOS = '<SOS>'
SOS_INDEX = 0
EOS_INDEX = 1
MAX_LENGTH = 20


# model
#
class Encoder(nn.Module):
    """Encoder module."""
    def __init__(self, input_size, hidden_size=200, n_layers=2, dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        y = self.embedding(x)
        y, (hn, cn) = self.lstm(y)
        return y, (hn, cn)


class Decoder(nn.Module):
    """Decoder module."""

    def __init__(self, output_size, hidden_size=200, n_layers=2, dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, enc_output, enc_hidden):
        batch_size = enc_output.size(0)
        # hidden_size = enc_output.size(2)
        dec_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_INDEX)
        dec_hidden = enc_hidden

        dec_outputs = []
        for i in range(MAX_LENGTH):
            # forward pass for next token
            dec_output, dec_hidden = self.forward_step(dec_input, dec_hidden)
            dec_outputs.append(dec_output)

            # prepare for next iteration
            _, top_index = dec_output.topk(1)
            dec_input = top_index.squeeze(-1).detach()

        dec_outputs = torch.cat(dec_outputs, dim=1)
        dec_outputs = F.log_softmax(dec_outputs, dim=-1)
        return dec_outputs, dec_hidden

    def forward_step(self, input, hidden):
        # predict next token
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden
