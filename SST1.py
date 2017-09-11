from torchtext import data
import os

class SST1Dataset(data.TabularDataset):
    dirname = 'data'
    @classmethod
    def splits(cls, text_field, label_field, head_text_field, head_pos_field, head_dep_field, word_pos_field,
               word_dep_field, train='phrases.train.tsv', validation='dev.tsv', test='test.tsv'):
        prefix_name = 'stsa.fine.'
        path = './data'
        return super(SST1Dataset, cls).splits(
            os.path.join(path, prefix_name), train, validation, test,
            format='TSV', fields=[('label', label_field), ('text', text_field), ('head_text', head_text_field),
                                  ('head_pos', head_pos_field), ('head_dep', head_dep_field),
                                  ('word_pos', word_pos_field), ('word_dep', word_dep_field)]
        )