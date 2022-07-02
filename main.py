import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

# TODO:
# - implement on device
# - implement micro & macro F1 score
# - answer theoretical questions


class SimpleSequenceTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, class_size):
        super(SimpleSequenceTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.class_size = class_size
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            bidirectional=True)
        self.hidden_to_ner = nn.Linear(2 * self.hidden_dim, self.class_size)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, sentence):
        output, _ = self.lstm(sentence)
        labels = self.hidden_to_ner(output)
        scores = F.softmax(labels, dim=1)
        return scores

    def train(self, epochs, data, learning_rate=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for x, y in data:
                outputs = self.forward(x)  # forward pass
                optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

                # obtain the loss function
                loss = self.loss_function(outputs, y)

                loss.backward()  # calculates the loss of the loss function

                optimizer.step()  # improve from loss, i.e backprop
            print("Iteration:", epoch, "Loss:", loss.item())

    def predict(self, data):
        for d in data:
            outputs = self.lstm.forward(d[0])
        return outputs


class CustomDataset(Dataset):
    def __init__(self, file, plain_text=False, embeddings_file='./embeddings/glove.6B.50d.txt', embedding_size=50):
        self.mapping_index = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6,
                              'I-PER': 7, 'O': 8}
        self.embedding_size = embedding_size
        self.file = file
        self.vocab = self._generate_vocab(embeddings_file)
        f = open(self.file)
        labels = []
        sentence = []
        data = []
        for line in f:
            splitted = line.split('\t')
            if splitted[0].startswith('-DOCSTART') or splitted[0] == "\n" or len(splitted[0]) == 0:
                if len(sentence) != 0:
                    data.append([sentence, labels])
                    sentence = []
                    labels = []
            else:
                sentence.append(splitted[0].lower())
                labels.append(splitted[-1].removesuffix('\n'))
        if not plain_text:
            embedded_data = []
            for d in data:
                mapped_labels = []
                for l in d[1]:
                    mapped_labels.append(self.mapping_index[l])
                embedded_data.append(
                    [self._generate_embeddings(d[0], self.vocab),
                     self._one_hot_encoding(mapped_labels, len(self.mapping_index))])
            data = embedded_data
        self.data = data

    def _one_hot_encoding(self, data_labels, class_size):
        targets = torch.zeros(len(data_labels), class_size)
        for i, label in enumerate(data_labels):
            targets[i, label] = 1
        return torch.Tensor(targets)

    def _generate_embeddings(self, sentence, vocab):
        embeddings = torch.zeros(len(sentence), self.embedding_size)
        for i, s in enumerate(sentence):
            try:
                embeddings[i] = torch.Tensor(vocab[s])
            except:
                # if not out of vocab word
                embeddings[i] = torch.zeros(self.embedding_size)

        return embeddings

    def _generate_vocab(self, embeddings_file):
        vocab = {}
        f = open(embeddings_file, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            vocab[word] = coefs
        f.close()
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_sent, seq_label = self.data[idx]
        return seq_sent, seq_label


def main():
    dev_file = './data/dev.conll'  # path to training data
    test_file = './data/test.conll'  # path to validation data
    train_file = './data/train.conll'  # path to test data

    train_dataset = CustomDataset(file=train_file)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    dev_dataset = CustomDataset(file=dev_file)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True)
    test_dataset = CustomDataset(file=test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    seq_tagger = SimpleSequenceTagger(input_dim=50, hidden_dim=100, num_layers=1, class_size=9)

    seq_tagger.train(epochs=20, data=train_dataloader, learning_rate=0.01)


if __name__ == "__main__":
    main()
