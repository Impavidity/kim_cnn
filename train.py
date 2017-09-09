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
    TEXT = data.Field(batch_first=True, lower=True)
    LABEL = data.Field(sequential=False)
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=True,
        filter_pred=lambda  ex: ex.label != 'neutral' # Need to check this parameters later
    )

if args.dataset == 'TREC':
    TEXT = data.Field(batch_first=True, lower=True)
    LABEL = data.Field(sequential=False)
    train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)



TEXT.build_vocab(train, vectors='glove.6B.300d')
LABEL.build_vocab(train)

#print('len(TEXT.vocab)', len(TEXT.vocab))
#print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

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


#print(config)

print("len(LABEL.target_class)", len(LABEL.vocab))

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = KimCNN(config)
    model.embed.weight.data = TEXT.vocab.vectors
    if args.cuda:
        model.cuda()
        print("Shift model to GPU")


parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)
criterion = nn.CrossEntropyLoss()
early_stop = False
best_dev_acc = 0
iterations = 0
iters_not_improved = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
print(header)

for epoch in range(args.epochs):
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, best_dev_acc))
        break

    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        # Batch size : (Sentence Length, Batch_size)
        iterations += 1
        model.train(); optimizer.zero_grad()
        #print("Text Size:", batch.text.size())
        #print("Label Size:", batch.label.size())
        scores = model(batch)
        n_correct += (torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(scores, batch.label)
        loss.backward()

        optimizer.step()


        # Evaluate performance on validation set
        if iterations % args.dev_every == 0:
            # switch model into evalutaion mode
            model.eval(); val_iter.init_epoch()
            n_dev_correct = 0
            dev_losses = []
            for dev_batch_idx, dev_batch in enumerate(val_iter):
                scores = model(dev_batch)
                n_dev_correct += (torch.max(scores, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(scores, dev_batch.label)
                dev_losses.append(dev_loss.data[0])
            dev_acc = 100. * n_dev_correct / len(val)
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0],
                                          sum(dev_losses) / len(dev_losses), train_acc, dev_acc))

            # Update validation results
            if dev_acc > best_dev_acc:
                iters_not_improved = 0
                best_dev_acc = dev_acc
                snapshot_path = os.path.join(args.save_path, 'best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 0:
            # print progress message
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                                      n_correct / n_total * 100, ' ' * 12))




















