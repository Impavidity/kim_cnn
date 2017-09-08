import torch 
import torch.optim as optim
import torch.nn as nn
import time
import os
import numpy as np

from torchtext import data
from torchtext import datasets
from args import get_args
from model import KimCNN

# Set default configuration in : args.py
args = get_args()
# Set random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")

# Set up the data for training
# SST-1
if args.dataset == 'SST-1':
    TEXT = data.Field()
    LABEL = data.Field(sequential=False)
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=False, train_subtrees=True,
        filter_pred=lambda  ex: ex.label != 'neutral' # Need to check this parameters later
    )

TEXT.build_vocab(train, vectors='glove.6B.300d')
LABEL.build_vocab(train)

#print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=args.batch_size, device=args.gpu
)

#batch = next(iter(train_iter))
#print(batch.text)
#print(batch.label)

config = args
config.target_class = len(LABEL.vocab)
config.words_num = len(TEXT.vocab)
config.embed_num = len(TEXT.vocab)


print(config)

print("len(LABEL.target_class)", len(LABEL.vocab))

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = KimCNN(config)
    model.embed.weight.data = TEXT.vocab.vectors


while True:
    for batch_idx, batch in enumerate(train_iter):
        pass
    print("epoch")

















