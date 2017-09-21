import torch
import torch.nn as nn

import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, config):
        super(KimCNN, self).__init__()
        output_channel = config.output_channel
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        embed_num = config.embed_num
        embed_dim = config.embed_dim
        self.mode = config.mode
        Ks = 3 # There are three conv net here

        vocab_pos_size = config.pos_vocab
        vocab_dep_size = config.dep_vocab
        pos_size = 44
        dep_size = 41

        if self.mode == 'linguistic_multichannel':
            input_channel = 4
        elif self.mode == 'multichannel' or self.mode == 'linguistic_static' or self.mode == 'linguistic_nonstatic':
            input_channel = 2
        elif self.mode == 'linguistic_head':
            input_channel = 1
        else:
            input_channel = 1

        if 'linguistic_nonstatic' in self.mode or 'linguistic_static' in self.mode:
            words_dim += pos_size + dep_size
        elif 'linguistic_head' in self.mode:
            words_dim += (pos_size + dep_size + words_dim)


        self.embed = nn.Embedding(words_num, words_dim)
        self.static_embed = nn.Embedding(embed_num, embed_dim)
        self.non_static_embed = nn.Embedding(embed_num, embed_dim)
        self.static_embed.weight.requires_grad = False

        self.static_pos_embed = nn.Embedding(vocab_pos_size, pos_size)
        self.static_dep_embed = nn.Embedding(vocab_dep_size, dep_size)
        self.static_pos_embed.weight.requires_grad = False
        self.static_dep_embed.weight.requires_grad = False

        self.non_static_pos_embed = nn.Embedding(vocab_pos_size, pos_size)
        self.non_static_dep_embed = nn.Embedding(vocab_dep_size, dep_size)

        self.static_sentiment_embed = nn.Embedding(words_num, 2)
        self.static_sentiment_embed.weight.requires_grad = False
        self.non_static_sentiment_embed = nn.Embedding(words_num, 2)

        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2,0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3,0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (7, words_dim), padding=(6,0))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(Ks * output_channel, target_class)


    def forward(self, x):
        x_text = x.text
        x_pos = x.word_pos
        x_dep = x.word_dep
        head_x = x.head_text
        head_pos = x.head_pos
        head_dep = x.head_dep

        if self.mode == 'rand':
            word_input = self.embed(x_text) # (batch, sent_len, embed_dim)
            x = word_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'static':
            static_input = self.static_embed(x_text)
            x = static_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'non-static':
            non_static_input = self.static_embed(x_text)
            x = non_static_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'multichannel':
            non_static_input = self.non_static_embed(x_text)
            static_input = self.static_embed(x_text)
            x = torch.stack([non_static_input, static_input], dim=1) # (batch, channel_input=2, sent_len, embed_dim)
        elif self.mode == 'linguistic_static':
            word_static_input = self.static_embed(x_text)
            word_static_pos_input = self.static_pos_embed(x_pos)
            word_static_dep_input = self.static_dep_embed(x_dep)
            word_channel = torch.cat([word_static_input, word_static_pos_input, word_static_dep_input], 2)
            head_static_input = self.static_embed(head_x)
            head_static_pos_input = self.static_pos_embed(head_pos)
            head_static_dep_input = self.static_dep_embed(head_dep)
            head_channel = torch.cat([head_static_input, head_static_pos_input, head_static_dep_input], 2)
            x = torch.stack([head_channel, word_channel], dim=1)
        elif self.mode == 'linguistic_nonstatic':
            word_non_static_input = self.non_static_embed(x_text)
            word_non_static_pos_input = self.non_static_pos_embed(x_pos)
            word_non_static_dep_input = self.non_static_dep_embed(x_dep)
            word_channel = torch.cat([word_non_static_input, word_non_static_pos_input, word_non_static_dep_input], 2)
            head_non_static_input = self.non_static_embed(head_x)
            head_non_static_pos_input = self.non_static_pos_embed(head_pos)
            head_non_static_dep_input = self.non_static_dep_embed(head_dep)
            head_channel = torch.cat([head_non_static_input, head_non_static_pos_input, head_non_static_dep_input], 2)
            x = torch.stack([head_channel, word_channel], dim=1)
        elif self.mode == 'linguistic_multichannel':
            word_static_input = self.static_embed(x_text)
            word_static_pos_input = self.static_pos_embed(x_pos)
            word_static_dep_input = self.static_dep_embed(x_dep)
            word_channel_static = torch.cat([word_static_input, word_static_pos_input, word_static_dep_input], 2)

            head_static_input = self.static_embed(head_x)
            head_static_pos_input = self.static_pos_embed(head_pos)
            head_static_dep_input = self.static_dep_embed(head_dep)
            head_channel_static = torch.cat([head_static_input, head_static_pos_input, head_static_dep_input], 2)

            word_non_static_input = self.non_static_embed(x_text)
            word_non_static_pos_input = self.non_static_pos_embed(x_pos)
            word_non_static_dep_input = self.non_static_dep_embed(x_dep)
            word_channel_dynamic = torch.cat([word_non_static_input, word_non_static_pos_input,
                                                word_non_static_dep_input], 2)
            head_non_static_input = self.non_static_embed(head_x)
            head_non_static_pos_input = self.non_static_pos_embed(head_pos)
            head_non_static_dep_input = self.non_static_dep_embed(head_dep)
            head_channel_dynamic = torch.cat([head_non_static_input, head_non_static_pos_input,
                                                head_non_static_dep_input], 2)
            x = torch.stack([word_channel_dynamic, word_channel_static, head_channel_dynamic,head_channel_static], dim=1)
        elif self.mode == 'linguistic_head':
            word_static_input = self.non_static_embed(x_text)
            word_static_pos_input = self.static_pos_embed(x_pos)
            word_static_dep_input = self.static_dep_embed(x_dep)
            word_head_input = self.non_static_embed(head_x)
            x = torch.cat([word_static_input, word_static_pos_input, word_static_dep_input, word_head_input], 2)
        elif self.mode == 'nonstatic_plus_feats':
            non_static_input = self.static_embed(x_text)
            non_static_sentiment =  self.non_static_sentiment_embed(x_text)
            x = torch.cat([non_static_input, non_static_sentiment], dim=2).unsqueeze(1)
        else:
            print("Unsupported Mode")
            exit()
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        # (batch, channel_output) * Ks
        x = torch.cat(x, 1) # (batch, channel_output * Ks)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit
