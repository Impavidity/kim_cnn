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
        self.embed = nn.Embedding(config.embed_num, config.embed_dim)


    def forward(self, x):
        return 0