import re
from argparse import ArgumentParser

def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.lower().strip().split()


parser = ArgumentParser(description='pre-process the file')
parser.add_argument('--input', type=str)
args = parser.parse_args()

with open(args.input) as f, open(args.input + '.tsv', 'w') as g :
    for line in f:
        cleaned_str = clean_str_sst(line)

        if len(cleaned_str) > 1:
            g.write(cleaned_str[0] + "\t" + ' '.join(cleaned_str[1:]) + "\n")
