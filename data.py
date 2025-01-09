"""
Data loading and processing.
"""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from model import SOS, EOS, SOS_INDEX, EOS_INDEX


class Vocabulary:
    """Vocabulary."""
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


def load_data(data_path: str):
    """Creates a dataset and the vocabularies for commands and actions."""
    commands = []
    actions = []
    commands_vocab = Vocabulary()
    actions_vocab = Vocabulary()
    with open(data_path) as file:
        for line in file.readlines():
            in_part, out_part = line.split(" OUT: ")
            in_sequence = in_part.replace("IN: ", "").strip().split()
            out_sequence = out_part.strip().split()

            commands_vocab.add_sentence(in_sequence)
            actions_vocab.add_sentence(out_sequence)

            commands.append(commands_vocab.encode(in_sequence + [EOS]))
            actions.append(actions_vocab.encode([SOS] + out_sequence + [EOS]))

    dataset = CompositionalDataset(commands, actions)
    return dataset, commands_vocab, actions_vocab


class CompositionalDataset(Dataset):
    """Dataset for compositional tasks."""
    def __init__(self, commands, actions):
        self.commands = commands
        self.actions = actions

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        return torch.IntTensor(self.commands[idx]), torch.IntTensor(self.actions[idx])


def pad_batch(batch: list):
    """Pads a batch of variable length sequences."""
    commands, actions = zip(*batch)
    padded_commands = pad_sequence(commands, batch_first=True, padding_value=EOS_INDEX)
    padded_actions = pad_sequence(actions, batch_first=True, padding_value=EOS_INDEX)
    return padded_commands, padded_actions
