import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
import statistics
import nltk
import random
import collections
import time

class Dictionary(object):
    def __init__(self, dataset, size):
        ## init vocab ##
        self.word2idx = {'<pad>':0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = ['<pad>', '<sos>', '<eos>', '<unk>']

        self.build_dict(dataset, size)

    def __call__(self, word):
        return self.word2idx.get(word, 3) # if word does not exist in vocab then return unk idx

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def build_dict(self, dataset, dict_size):
        """Tokenizes a text file."""
        total_words = (word for sent in dataset for word in sent)
        word_freq = collections.Counter(total_words) # count the number of each word
        vocab = sorted(word_freq.keys(), key=lambda word: (-word_freq[word], word)) # sort by frequency
        vocab = vocab[:dict_size]
        for word in vocab:
            self.add_word(word)

    def __len__(self):
        return len(self.idx2word)

def brown_dataset(min=5, max=30):
    nltk.download('brown')

    # get sentences with the length between min and max
    # convert all words into lower-case
    all_seq = [[token.lower() for token in seq] for seq in nltk.corpus.brown.sents() if min <= len(seq) <= max]

    random.shuffle(all_seq) # shuffle
    return all_seq

class Corpus(object):
    def __init__(self, dataset, device, dict_size=20000, train_ratio=0.95):
        train_size = int(len(dataset) * train_ratio)
        valid_size = len(dataset) - train_size
        self.device = device
        self.dictionary = Dictionary(dataset, dict_size)
        self.train = dataset[:train_size]
        self.valid = dataset[:valid_size]

    def indexing(self, dat):
        src_idxes = []
        tgt_idxes = []
        for sent in dat:
            src_idx = [self.dictionary('<sos>')] + [self.dictionary(word) for word in sent]
            tgt_idx = [self.dictionary(word) for word in sent] + [self.dictionary('<eos>')]
            src_idxes.append(torch.tensor(src_idx).type(torch.int64))
            tgt_idxes.append(torch.tensor(tgt_idx).type(torch.int64))

        src_idxes = rnn.pad_sequence(src_idxes, batch_first=True).to(self.device) # shape = [B, L]
        tgt_idxes = rnn.pad_sequence(tgt_idxes, batch_first=True).to(self.device).view(-1) # flatten shape = [B * L]

        return src_idxes, tgt_idxes

    def batch_iter(self, batch_size, isTrain=True):
        dat = self.train if isTrain else self.valid
        if isTrain:
            random.shuffle(dat)

        for i in range(len(dat) // batch_size):
            batch = dat[i * batch_size: (i+1) * batch_size]
            src, tgt = self.indexing(batch)
            yield {'src': src, 'tgt': tgt}

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load dataset
dataset = brown_dataset()
corpus = Corpus(dataset, device)

# Define network
class RNNModel(nn.Module):
    def __init__(self, ntoken, hidden_size, nlayers, dropout=0.1):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(ntoken, hidden_size, padding_idx=0)
        self.rnn = nn.LSTM(hidden_size, hidden_size, nlayers, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, ntoken)
        self.sm = nn.LogSoftmax(dim=1)

        self.ntoken = ntoken
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def forward(self, input, hidden):
        emb = self.embeddings(input) # emb = (batch, length, dim)
        output, hidden = self.rnn(emb, hidden) # output = (batch. length. dim)
        output = self.drop(output)
        output = self.output_layer(output)
        output = output.view(-1, self.ntoken) # output = (batch * length, dim)

        return self.sm(output), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()) # to set init tensor with the same torch.dtype and torch.device
        return (weight.new_zeros(self.nlayers, bsz, self.hidden_size),
                weight.new_zeros(self.nlayers, bsz, self.hidden_size))


# Hyperparameters
batch_size = 40
hidden_size = 256
dropout = 0.2
max_epoch = 30

# build model
ntokens = len(corpus.dictionary)
model = RNNModel(ntokens, hidden_size, 2, dropout).to(device)

# set loss func and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.NLLLoss(ignore_index=0, reduction='mean')

########################################################################
#   Training code
########################################################################

# accuracy
def cal_acc(scores, target):
    pred = scores.max(-1)[1]
    non_pad = target.ne(0)
    num_correct = pred.eq(target).masked_select(non_pad).sum().item()
    num_non_pad = non_pad.sum().item()
    return 100 * (num_correct / num_non_pad)

# train func.
def train():
    model.train() # Turn on training mode which enables dropout.
    mean_loss = []
    mean_acc = []
    start_time = time.time()

    for batch in corpus.batch_iter(batch_size):
        hidden = model.init_hidden(batch_size)
        target = batch['tgt']
        optimizer.zero_grad()
        output, hidden = model(batch['src'], hidden)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        mean_loss.append(loss.item())
        mean_acc.append(cal_acc(output, target))

    total_time = time.time() - start_time
    mean_acc = statistics.mean(mean_acc)
    mean_loss = statistics.mean(mean_loss)

    return mean_loss, total_time, mean_acc

# evaluation func.
def evaluate():
    model.eval() # Turn off dropout
    mean_loss = []
    mean_acc = []

    for batch in corpus.batch_iter(batch_size, isTrain=False):
        with torch.no_grad():
            hidden = model.init_hidden(batch_size)
            target = batch['tgt']
            output, hidden = model(batch['src'], hidden)
            loss = criterion(output, target)
            mean_loss.append(loss.item())
            mean_acc.append(cal_acc(output, target))

    mean_acc = statistics.mean(mean_acc)
    mean_loss = statistics.mean(mean_loss)

    return mean_loss, mean_acc

isTrain=False # Flag variable

if isTrain: # set False if you don't need to train model
    start_time = time.time()

    for epoch in range(1, max_epoch+1):
        loss, epoch_time, accuracy = train()
        print('epoch {:4d} | times {:3.3f} |  loss: {:3.3f} | accuracy: {:3.2f}'.format(epoch+1, epoch_time, loss, accuracy))

        if epoch % 10 == 0:
            loss, accuracy = evaluate()
            print('=' * 60)
            print('Evaluation | loss: {:3.3f} | accuracy: {:3.2f}'.format(epoch+1, epoch_time, loss, accuracy))
            print('=' * 60)

    with open('model.pt', 'wb') as f:
        print('save model at: ./model.pt')
        torch.save(model, f)

def pred_sent_prob(sent):
    # import numpy as np
    # model.eval()
    # with torch.no_grad():
    #     src, tgt = corpus.indexing(sent)
    #     hidden = model.init_hidden(bsz=1)
    #     output, hidden = model(src, hidden)
    #     word_prob = output[np.arange(len(tgt)), tgt].tolist()
    #     sent_prob = np.sum(word_prob)
    #     return sent_prob

#    import numpy as np
    model.eval()
    sent_prob = 0
    with torch.no_grad():

        idx_src, idx_tgt = corpus.indexing(sent)
        hidden = model.init_hidden(1)
        output, hidden = model(idx_src, hidden)

        for i,embedded in enumerate(output) :
            sent_prob+= embedded[idx_tgt][i]

        return sent_prob

# load saved model
with open('./model.pt', 'rb') as f:
    print('load model from: ./model.pt')
    model = torch.load(f).to(device)

    print('log prob of [the cat bark .]: {:3.3f}'.format(pred_sent_prob([['the', 'dog', 'bark', '.']])))
    print('log prob of [the cat bark .]: {:3.3f}'.format(pred_sent_prob([['the', 'cat', 'bark', '.']])))

    print('log prob of [boy am a i .]: {:3.3f}'.format(pred_sent_prob([['boy', 'am', 'a', 'i', '.']])))
    print('log prob of [i am a boy .]: {:3.3f}'.format(pred_sent_prob([['i', 'am', 'a', 'boy', '.']])))

