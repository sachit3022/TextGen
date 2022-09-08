# pig_latin.py

import math
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

__all__ = ['PigLatinData']

class PigLatinData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.filename = args.data_file
        source_lines, target_lines = self.read_pairs()

        # Filter lines
        source_lines = self.filter_lines(source_lines)
        target_lines = self.filter_lines(target_lines)

        all_characters = set(''.join(source_lines)) | set(
            ''.join(target_lines))

        # Create a dictionary mapping each character to a unique index
        self.char_to_index = {char: index for (
            index, char) in enumerate(sorted(list(all_characters)))}

        # Add start and end tokens to the dictionary
        self.start_token = len(self.char_to_index)
        self.end_token = len(self.char_to_index) + 1
        self.char_to_index['SOS'] = self.start_token
        self.char_to_index['EOS'] = self.end_token

        # Create the inverse mapping, from indexes to characters (used to decode the model's predictions)
        self.index_to_char = {index: char for (
            char, index) in self.char_to_index.items()}

        # Store the final size of the vocabulary
        self.vocab_size = len(self.char_to_index)
        self.idx_dict = {'char_to_index': self.char_to_index,
                         'index_to_char': self.index_to_char,
                         'start_token': self.start_token,
                         'end_token': self.end_token}
        line_pairs = list(set(zip(source_lines, target_lines)))  # Python 3

        self.val_dict = self.create_dict(line_pairs[math.floor(0.8 * len(line_pairs))+1:])
        self.train_dict = self.create_dict(line_pairs[0:math.floor(0.8 * len(line_pairs))])

    def filter_lines(self, lines):
        """Filters lines to consist of only alphabetic characters or dashes "-".
        """
        return [line for line in lines if self.all_alpha_or_dash(line)]

    def all_alpha_or_dash(self, s):
        """Helper function to check whether a string is alphabetic, allowing dashes '-'.
        """
        return all(c.isalpha() or c == '-' for c in s)

    def read_lines(self):
        """Read a file and split it into lines.
        """
        lines = open(self.filename).read().strip().lower().split('\n')
        return lines

    def read_pairs(self):
        """Reads lines that consist of two words, separated by a space.
        Returns:
            source_words: A list of the first word in each line of the file.
            target_words: A list of the second word in each line of the file.
        """
        lines = self.read_lines()
        source_words, target_words = [], []
        for line in lines:
            line = line.strip()
            if line:
                source, target = line.split()
                source_words.append(source)
                target_words.append(target)
        return source_words, target_words

    def create_dict(self, pairs):
        """Creates a mapping { (source_length, target_length): [list of (source, target) pairs]
        This is used to make batches: each batch consists of two parallel tensors, one containing
        all source indexes and the other containing all corresponding target indexes.
        Within a batch, all the source words are the same length, and all the target words are
        the same length.
        """
        unique_pairs = list(
            set(pairs))  # Find all unique (source, target) pairs

        d = defaultdict(list)
        for (s, t) in unique_pairs:
            d[(len(s), len(t))].append((s, t))

        return d

    def get_batch_data(self, data_dict, batch_size):
        for key in data_dict:
            input_strings, target_strings = zip(*data_dict[key])
            input_tensors = [torch.LongTensor(self.string_to_index_list(
                s, self.char_to_index, self.end_token)) for s in input_strings]
            target_tensors = [torch.LongTensor(self.string_to_index_list(
                s, self.char_to_index, self.end_token)) for s in target_strings]
            num_tensors = len(input_tensors)
            num_batches = int(math.ceil(num_tensors / float(batch_size)))
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size

                inputs = torch.stack(input_tensors[start:end])
                targets = torch.stack(target_tensors[start:end])
                yield inputs, targets

    def string_to_index_list(self, s, char_to_index, end_token):
        """Converts a sentence into a list of indexes (for each character).
        """
        return [char_to_index[char] for char in s] + [end_token]  # Adds the end token to each index list

    def train_dataloader(self):
        return self.get_batch_data(self.train_dict, self.args.batch_size)

    def val_dataloader(self):
        return self.get_batch_data(self.val_dict, self.args.batch_size)

    def test_dataloader(self):
        return self.get_batch_data(self.train_dict, 1)
