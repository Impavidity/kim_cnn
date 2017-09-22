from tqdm import tqdm
import array
import torch
import six
import numpy as np

from argparse import ArgumentParser



def convert(fname):
    save_file = '{}.pt'.format(fname)
    itos, vectors, dim = [], array.array('d'), None

    with open(fname, 'r') as f:
        lines = [line for line in f]
    print("Loading vectors from {}".format(fname))

    for line in tqdm(lines, total=len(lines)):
        entries = line.strip().split('\t')
        word, entries = entries[0], entries[1:]
        if dim is None:
            dim = len(entries)
        vectors.extend(float(x) for x in entries)
        itos.append(word)

    stoi = {word: i for i, word in enumerate(itos)}
    vectors = torch.Tensor(vectors).view(-1, dim)
    print('saving vectors to', save_file)
    torch.save((stoi, vectors, dim), save_file)


if __name__ == '__main__':
    parser = ArgumentParser(description='create word embedding')
    parser.add_argument('--input', type=str, required=True)

    args = parser.parse_args()
    convert(args.input)
