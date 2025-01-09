"""
Main script.
"""
import torch
from icecream import ic

from model import Encoder, Decoder
from data import load_data, pad_batch

# data
#
dataset, commands_vocab, actions_vocab = load_data('data/tasks.txt')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=pad_batch)

# model
#
commands_vocab_size = len(commands_vocab)
actions_vocab_size = len(actions_vocab)
n_layers = 2
encoder = Encoder(commands_vocab_size, n_layers=n_layers)
decoder = Decoder(actions_vocab_size, n_layers=n_layers)
ic(encoder)
ic(decoder)
ic('---')

# train
#
n_epochs = 1
for epoch in range(n_epochs):
    ic(epoch)
    for i, (x_seq, y_seq) in enumerate(data_loader):
        ic(i, x_seq.shape, y_seq.shape)
        # TODO
