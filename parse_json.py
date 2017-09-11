import json
from argparse import ArgumentParser
import os

def get_dep_pos(string):
  this_pos = []
  json_dict = json.loads(string)

  for token in json_dict['tokens']:
    this_pos.append(token['pos'])

  len_dep = len(this_pos)
  this_dep = [None] * len_dep
  headwords = [None] * len_dep
  this_head_index = [None] * len_dep

  for dep in json_dict['basicDependencies']:
    index, dependency = dep['dependent'], dep['dep']
    this_dep[index - 1] = dependency
    headwords[index - 1] = dep['governorGloss']
    this_head_index[index - 1] = dep['governor']

  this_head_dep = []
  this_head_pos = []

  for head_index in this_head_index:
    if head_index - 1 < 0:
      this_head_dep.append('NO_DEP')
      this_head_pos.append('NO_POS')
    else:
      this_head_dep.append(this_dep[head_index - 1])
      this_head_pos.append(this_pos[head_index - 1])

  return ' '.join(headwords) + '\t' + ' '.join(this_head_pos) + '\t' + ' '.join(this_head_dep) + '\t' + \
         ' '.join(this_pos) + '\t' + ' '.join(this_dep)


if __name__ == '__main__':
    parser = ArgumentParser(description="Parse jsons")
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    input = os.path.splitext(args.input)[0]
    with open(input + '.tsv') as f, open(input + '.deps.json') as g, open(input + '.concat', 'w') as h:
        for line1, line2 in zip(f, g):
            line1 = line1.strip()
            line2 = line2.strip()

            h.write('{}\t{}'.format(line1, get_dep_pos(line2)) + '\n')