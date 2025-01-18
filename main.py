"""
Main script.
"""
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from icecream import ic
from torchmetrics import Accuracy

from model import Encoder, Decoder
from data import load_data, pad_batch

# data
#
dataset, commands_vocab, actions_vocab = load_data('data/tasks.txt')
n = len(dataset)
dataset_train, dataset_val, dataset_test = random_split(dataset, [int(n * .8), int(n * .1), int(n * .1)])
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, collate_fn=pad_batch)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=True, collate_fn=pad_batch)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True, collate_fn=pad_batch)

# model
#
commands_vocab_size = len(commands_vocab)
actions_vocab_size = len(actions_vocab)
n_layers = 2
max_length = dataset.get_max_length_actions()
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
train_losses = []
val_losses = []
for epoch in range(n_epochs):
    ic(epoch)
    # training
    encoder.train()
    decoder.train()
    _acc_loss = 0
    _n = 0
    for i, (x_seq_train, y_seq_train) in enumerate(data_loader_train):
        optimizer.zero_grad()

        enc_out, enc_hidden = encoder(x_seq_train)
        y_seq_pred, _ = decoder(enc_out, enc_hidden)

        loss = criterion(
            y_seq_pred.view(-1, actions_vocab_size),
            y_seq_train.view(-1)
        )
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            ic(epoch, i, loss.item())

        _acc_loss += loss.item()
        _n += 1
    train_losses.append(_acc_loss / _n)

    # validation
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        _acc_loss = 0
        _n = 0
        for i, (x_seq_val, y_seq_val) in enumerate(data_loader_val):
            enc_out, enc_hidden = encoder(x_seq_val)
            y_seq_pred, _ = decoder(enc_out, enc_hidden)

            loss = criterion(
                y_seq_pred.view(-1, actions_vocab_size),
                y_seq_val.view(-1)
            )
            _acc_loss += loss.item()
            _n += 1
        val_losses.append(_acc_loss / _n)

plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# evaluation
#
accuracy = Accuracy(task='multiclass', num_classes=actions_vocab_size, average=None)
encoder.eval()
decoder.eval()
with torch.no_grad():
    for i, (x_seq_test, y_seq_test) in enumerate(data_loader_test):
        enc_out, enc_hidden = encoder(x_seq_test)
        y_seq_pred, _ = decoder(enc_out, enc_hidden)

        acc = accuracy(y_seq_pred.argmax(dim=-1), y_seq_test)

print(f'Accuracy: {accuracy.compute()}')
print(actions_vocab.index2word)
print(f'Accuracy avg.: {accuracy.compute().mean().item()}')
accuracy.plot()
plt.show()
