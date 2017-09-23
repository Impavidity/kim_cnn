import torch
import torch.nn as nn

import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, config):
        super(KimCNN, self).__init__()
        output_channel1 = config.output_channel1
        output_channel2 = config.output_channel2
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        embed_num = config.embed_num
        embed_dim = config.embed_dim
        self.mode = config.mode
        Ks = 3 # There are three conv net here
        if config.mode == 'multichannel':
            input_channel = 2
        else:
            input_channel = 1
        self.embed = nn.Embedding(words_num, words_dim)
        self.static_embed = nn.Embedding(embed_num, embed_dim)
        self.non_static_embed = nn.Embedding(embed_num, embed_dim)
        self.static_embed.weight.requires_grad = False

        self.conv11 = nn.Conv2d(input_channel, output_channel1, (3, words_dim), padding=(2,0))
        self.conv12 = nn.Conv2d(input_channel, output_channel1, (4, words_dim), padding=(3,0))
        self.conv13 = nn.Conv2d(input_channel, output_channel1, (5, words_dim), padding=(4,0))

        self.conv21 = nn.Conv2d(input_channel, output_channel2, (3, words_dim), padding=(2,0))
        self.conv22 = nn.Conv2d(input_channel, output_channel2, (4, words_dim), padding=(3,0))
        self.conv23 = nn.Conv2d(input_channel, output_channel2, (5, words_dim), padding=(4,0))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(Ks * output_channel2, target_class)


    def forward(self, x):
        x = x.text
        if self.mode == 'rand':
            word_input = self.embed(x) # (batch, sent_len, embed_dim)
            x = word_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'static':
            static_input = self.static_embed(x)
            x = static_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'non-static':
            non_static_input = self.static_embed(x)
            x = non_static_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'multichannel':
            non_static_input = self.non_static_embed(x)
            static_input = self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1) # (batch, channel_input=2, sent_len, embed_dim)
        else:
            print("Unsupported Mode")
            exit()
        x = [F.relu(self.conv11(x)).squeeze(3), F.relu(self.conv12(x)).squeeze(3), F.relu(self.conv13(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        # (batch, channel_output) * Ks
        x = torch.cat(x, 1) # (batch, channel_output * Ks)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit
