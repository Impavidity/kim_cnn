import os
import sys
import numpy as np
import torch

from torchtext import data
from args import get_args

from SST1 import SST1Dataset
from utils import clean_str_sst
import random

args = get_args()
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")
np.random.seed(args.seed)
random.seed(args.seed)

if not args.trained_model:
    print("Error: You need to provide a option 'trained_model' to load the model")
    sys.exit(1)

if args.dataset == 'SST-1':
    TEXT = data.Field(batch_first=True, tokenize=clean_str_sst)
    LABEL = data.Field(sequential=False)
    HEAD_TEXT = data.Field(batch_first=True)
    HEAD_POS_TAG = data.Field(batch_first=True)
    HEAD_DEP_TAG = data.Field(batch_first=True)
    WORD_POS_TAG = data.Field(batch_first=True)
    WORD_DEP_TAG = data.Field(batch_first=True)
    WORD_SENTIMENT = data.Field(batch_first=True, tokenize=clean_str_sst)
    train, dev, test = SST1Dataset.splits(TEXT, LABEL, HEAD_TEXT, HEAD_POS_TAG, HEAD_DEP_TAG, WORD_POS_TAG,
                                          WORD_DEP_TAG, WORD_SENTIMENT)

TEXT.build_vocab(train, min_freq=2)
LABEL.build_vocab(train)
WORD_POS_TAG.build_vocab(train)
WORD_DEP_TAG.build_vocab(train)
HEAD_TEXT.build_vocab(train, min_freq=2)
HEAD_POS_TAG.build_vocab(train)
HEAD_DEP_TAG.build_vocab(train)
WORD_SENTIMENT.build_vocab(train, min_freq=2)


train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=False)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

config = args
config.target_class = len(LABEL.vocab)
config.words_num = len(TEXT.vocab)
config.embed_num = len(TEXT.vocab)
config.pos_vocab = len(WORD_POS_TAG.vocab)
config.dep_vocab = len(WORD_DEP_TAG.vocab)
config.sentiment_num = len(WORD_SENTIMENT.vocab)

print("Label dict:", LABEL.vocab.itos)

if args.cuda:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = torch.load(args.trained_model, map_location=lambda storage,location: storage)


def predict(dataset_iter, dataset, dataset_name):
    print("Dataset: {}".format(dataset_name))
    model.eval()
    dataset_iter.init_epoch()

    n_correct = 0
    for data_batch_idx, data_batch in enumerate(dataset_iter):
        scores = model(data_batch)
        n_correct += (torch.max(scores, 1)[1].view(data_batch.label.size()).data == data_batch.label.data).sum()

    print("no. correct {} out of {}".format(n_correct, len(dataset)))
    accuracy = 100. * n_correct / len(dataset)
    print("{} accuracy: {:8.6f}%".format(dataset_name, accuracy))

# Run the model on the dev set
predict(dataset_iter=dev_iter, dataset=dev, dataset_name="valid")

# Run the model on the test set
predict(dataset_iter=test_iter, dataset=test, dataset_name="test")