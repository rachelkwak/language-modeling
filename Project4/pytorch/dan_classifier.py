"""Simple deep averaging net classifier"""
import os
import pickle
from time import clock

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


MAX_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_DIM = 32
VOCAB_SIZE = __FIXME__


def make_batches(data, batch_size):
    batches = []
    batch = []
    for y, sent in data:
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []

        sent = Variable(torch.LongTensor(sent))
        batch.append((y, sent))

    if batch:
        batches.append(batch)

    return batches


class DANClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        super(DANClassifier, self).__init__()

        self.embed = nn.Embedding(self.vocab_size,
                                  self.hidden_dim)

        self.hid = nn.Linear(self.hidden_dim,
                             self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim, 1)

    def _predict(self, batch, train=True):

        sent_embeds = []
        for _, sent in batch:

            sent_embed = self.embed(sent)
            sent_embed = sent_embed.mean(dim=0)
            sent_embeds.append(sent_embed)

        sent_embeds = torch.stack(sent_embeds)

        # computes b + W * sent_embed behind the scenes
        hid = self.hid(sent_embeds)
        hid = nn.functional.tanh(hid)
        y_scores = self.out(hid)
        y_probas = nn.functional.sigmoid(y_scores)
        return y_probas


    def forward(self, sents, train=True):
        """Computes the batch loss"""
        probas = self._predict(sents, train).squeeze()

        # we make a pytorch vector out of the true ys
        y_true = Variable(torch.Tensor([y for y, _ in sents]))

        total_loss = nn.functional.binary_cross_entropy(probas,
                                                        y_true,
                                                        size_average=False)

        return total_loss

    def num_correct(self, sents, train=True):
        probas = self._predict(sents, train=False)
        probas = [p.data[0] for p in probas]
        y_true = [y for y, _ in sents]

        correct = 0

        # FIXME: count the number of correct predictions here

        return correct


if __name__ == '__main__':

    torch.manual_seed(1)

    with open(os.path.join('processed', 'train_ix.pkl'), 'rb') as f:
        train_ix = pickle.load(f)

    with open(os.path.join('processed', 'valid_ix.pkl'), 'rb') as f:
        valid_ix = pickle.load(f)

    clf = DANClassifier(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)

    train_batches = make_batches(train_ix, batch_size=BATCH_SIZE)
    valid_batches = make_batches(valid_ix, batch_size=BATCH_SIZE)

    # opt = optim.Adadelta(clf.parameters(), rho=0.95)
    opt = optim.Adam(clf.parameters())

    for it in range(MAX_EPOCHS):
        tic = clock()

        # iterate over all training batches, accumulate loss.
        train_loss = 0
        for batch in train_batches:
            opt.zero_grad()
            loss = clf(batch)
            train_loss += loss.data[0]
            loss.backward()
            opt.step()

        # iterate over all validation batches, accumulate # correct pred.
        valid_acc = 0
        for batch in valid_batches:
            valid_acc += clf.num_correct(batch)

        valid_acc /= len(valid_ix)

        toc = clock()

        print(("Epoch {:3d} took {:3.1f}s. "
               "Train loss: {:8.5f} "
               "Valid accuracy: {:8.2f}").format(
            it,
            toc - tic,
            train_loss / len(train_ix),
            valid_acc * 100
            ))
