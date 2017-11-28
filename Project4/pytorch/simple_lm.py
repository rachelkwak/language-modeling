"""Simplest possible neural language model:
    use word w_i to predict word w_(i + 1)
"""
import os
import pickle
from time import clock
from math import exp

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


MAX_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_DIM = 32
USE_UNLABELED = False
VOCAB_SIZE = __FIXME__


def make_batches(data, batch_size):
    batches = []
    batch = []
    for _, sent in data:
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []

        sent = Variable(torch.LongTensor(sent))
        batch.append(sent)

    if batch:
        batches.append(batch)

    return batches


class SimpleLM(nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        super(SimpleLM, self).__init__()

        self.embed = nn.Embedding(self.vocab_size,
                                  self.hidden_dim)

        self.hid = nn.Linear(self.hidden_dim,
                             self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim,
                             self.vocab_size,
                             bias=False)

        self.loss = nn.CrossEntropyLoss(size_average=False)

    def forward(self, batch):

        ctx_ix = torch.cat([sent[:-1] for sent in batch])
        out_ix = torch.cat([sent[1:] for sent in batch])

        emb = self.embed(ctx_ix)
        hid = self.hid(emb)
        hid = nn.functional.tanh(hid)
        out = self.out(hid)

        return self.loss(out, out_ix)


if __name__ == '__main__':

    torch.manual_seed(1)

    with open(os.path.join('processed', 'train_ix.pkl'), 'rb') as f:
        train_ix = pickle.load(f)

    if USE_UNLABELED:
        __FIXME__

    with open(os.path.join('processed', 'valid_ix.pkl'), 'rb') as f:
        valid_ix = pickle.load(f)

    lm = SimpleLM(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    train_batches = make_batches(train_ix, batch_size=BATCH_SIZE)
    valid_batches = make_batches(valid_ix, batch_size=BATCH_SIZE)

    n_train_words = sum(len(sent) for _, sent in train_ix)
    n_valid_words = sum(len(sent) for _, sent in valid_ix)

    opt = optim.Adam(lm.parameters())

    for it in range(MAX_EPOCHS):
        tic = clock()

        # iterate over all training batches, accumulate loss.
        train_loss = 0
        for batch in train_batches:

            opt.zero_grad()
            loss = lm(batch)
            train_loss += loss.data[0]
            loss.backward()
            opt.step()

        # iterate over all validation batches, accumulate loss.
        valid_loss = 0
        for batch in valid_batches:
            loss = lm(batch)
            valid_loss += loss.data[0]

        toc = clock()

        print(("Epoch {:3d} took {:3.1f}s. "
               "Train perplexity: {:8.3f} "
               "Valid perplexity: {:8.3f}").format(
            it,
            toc - tic,
            exp(train_loss / n_train_words),
            exp(valid_loss / n_valid_words)
            ))

    # FIXME: make sure to update filenames when implementing ngram models
    fn = "pt_embeds_baseline_lm"
    if USE_UNLABELED:
        fn += "_unlabeled"

    print("Saving embeddings to {}".format(fn))
    torch.save(lm.embed.weight, fn)
