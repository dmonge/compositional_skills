"""
Main script.
"""
import torch
from icecream import ic

from model import Encoder, Decoder, EOS_INDEX
from data import load_data, pad_batch

# data
#
dataset, commands_vocab, actions_vocab = load_data('data/tasks.txt')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=pad_batch)
max_length = dataset.get_max_length_actions()
print(max_length)
# model
#
commands_vocab_size = len(commands_vocab)
actions_vocab_size = len(actions_vocab)
n_layers = 2
encoder = Encoder(commands_vocab_size, n_layers=n_layers)
decoder = Decoder(actions_vocab_size, n_layers=n_layers, max_seq_length=max_length)
ic(encoder)
ic(decoder)
ic('---')

# train
#
n_epochs = 10
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
for epoch in range(n_epochs):
    ic(epoch)
    for i, (x_seq_train, y_seq_train) in enumerate(data_loader):
        optimizer.zero_grad()

        enc_out, enc_hidden = encoder(x_seq_train)
        y_seq_pred, _ = decoder(enc_out, enc_hidden)

        loss = criterion(
            y_seq_pred.view(-1, actions_vocab_size),
            y_seq_train.view(-1)
        )
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            ic(epoch, i, loss.item())
