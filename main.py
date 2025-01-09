"""
Main script.
"""
import torch
from icecream import ic

from model import Encoder, Decoder, SOS, EOS, SOS_INDEX, EOS_INDEX


# data
#
commands = []
actions = []

class Language:

    def __init__(self):
        self.index2word = [SOS, EOS]
        self.word2index = {SOS: SOS_INDEX, EOS: EOS_INDEX}

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            index = len(self.index2word)
            self.word2index[word] = index
            self.index2word.append(word)

    def encode(self, words):
        return list(map(lambda w: self.word2index[w], words))

    def decode(self, indices):
        return list(map(lambda i: self.index2word[i], indices))

    def __len__(self):
        return len(self.index2word)


commands_language = Language()
actions_language = Language()
with open('data/tasks.txt') as file:
    for line in file.readlines():
        in_part, out_part = line.split(" OUT: ")
        in_sequence = in_part.replace("IN: ", "").strip().split()
        out_sequence = out_part.strip().split()

        commands_language.add_sentence(in_sequence)
        actions_language.add_sentence(out_sequence)

        commands.append(commands_language.encode(in_sequence + [EOS]))
        actions.append(actions_language.encode([SOS] + out_sequence + [EOS]))


# model
#
in_vocab_size = len(commands_language)
out_vocab_size = len(actions_language)
n_layers = 2
ic(in_vocab_size)
ic(out_vocab_size)

encoder = Encoder(in_vocab_size, n_layers=n_layers)
decoder = Decoder(out_vocab_size, n_layers=n_layers)
ic(encoder)
ic(decoder)
ic('---')


# forward pass
#
ic('data')
x = torch.tensor(commands[-1]).unsqueeze(0)  # batch_size, seq_length
y_true = torch.tensor(actions[-1]).unsqueeze(0)  # batch_size, seq_length
ic(x, x.shape)
ic(y_true, y_true.shape)

ic('encoder')
enc_out, enc_hidden = encoder(x)
ic(enc_out.shape)
ic(enc_hidden[-1].shape)
ic(enc_hidden[0].shape)

ic('decoder')
y_pred, dec_hidden = decoder(enc_out, enc_hidden)
ic(y_pred.shape)
ic(dec_hidden[-1].shape)
ic(dec_hidden[0].shape)
