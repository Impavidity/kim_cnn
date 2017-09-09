import os

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Kim CNN")
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0) # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--mode', type=str, default='multichannel')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2324)
    parser.add_argument('--dataset', type=str, default='SST-1')
    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--dev_every', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--save_path', type=str, default='saves')
    parser.add_argument('--output_channel', type=int, default=100)
    parser.add_argument('--words_dim', type=int, default=300)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch_dacay', type=int, default=15)
    parser.add_argument('--vector_cache', type=str, default="data/word2vec.sst-1.pt")
    parser.add_argument('--trained_model', type=str, default="")


    args = parser.parse_args()
    return args
