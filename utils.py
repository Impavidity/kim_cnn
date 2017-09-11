import re
import torch
import numpy as np

dep_tags = ['csubj', 'aux', 'acl:relcl', 'mark', 'expl', 'amod', 'acl', 'parataxis', 'compound',
            'advmod', 'nmod:poss', 'cc:preconj', 'det', 'case', 'ROOT', 'punct', 'nmod:npmod',
            'nsubjpass', 'det:predet', 'advcl', 'root', 'dep', 'mwe', 'xcomp', 'nmod', 'cop',
            'cc', 'nsubj', 'csubjpass', 'appos', 'conj', 'nummod', 'discourse', 'auxpass', 'ccomp',
            'nmod:tmod', 'iobj', 'compound:prt', 'dobj', 'neg', 'NO_DEP']

pos_tags = ['RBS', "''", 'VB', '#', '.', 'WP$', 'SYM', 'LS', 'WDT', 'NNP', 'TO', 'CD', 'NNPS',
            'NN', 'MD', 'RBR', 'JJS', 'VBN', 'VBP', '``', 'WRB', 'JJR', 'VBD', 'FW', 'RB', 'NNS',
            'POS', ',', 'PDT', 'UH', 'VBG', '$', 'PRP$', 'VBZ', 'PRP', ':', 'WP', 'IN', 'CC', 'DT',
            'JJ', 'RP', 'EX', 'NO_POS']

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.lower().strip().split()


def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.lower().strip().split()

def one_hot(tag, index):
    tag_one_hot = np.zeros(len(tag), dtype=float)
    tag_one_hot[index] = 1.0
    return tag_one_hot


def create_lookup():
    pos_pt = './data/pos.sst-1.pt'
    pos_dim = len(pos_tags)
    stoi = {word: i for i, word in enumerate(pos_tags)}
    vectors = [one_hot(pos_tags, pos_tags.index(tag)) for tag in pos_tags]
    vectors = torch.Tensor(vectors).view(-1, pos_dim)
    print('saving vectors to', pos_pt)
    torch.save((stoi, vectors, pos_dim), pos_pt)

    dep_pt = './data/dep.sst-1.pt'
    dep_dim = len(dep_tags)
    stoi = {word: i for i, word in enumerate(dep_tags)}
    vectors = [one_hot(dep_tags, dep_tags.index(tag)) for tag in dep_tags]
    vectors = torch.Tensor(vectors).view(-1, dep_dim)
    print('saving vectors to', dep_pt)
    torch.save((stoi, vectors, dep_dim), dep_pt)