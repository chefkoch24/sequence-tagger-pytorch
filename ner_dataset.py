from torch.utils.data import Dataset
import numpy as np
import torch


class NERDataset(Dataset):
    def __init__(self, file, embeddings_file='./embeddings/glove.6B.50d.txt', embedding_size=50,
                 mapping_index={'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6,
                                'I-PER': 7, 'O': 8}):
        self.mapping_index = mapping_index  # class mapping index
        self.embedding_size = embedding_size
        self.file = file
        self.vocab = self._generate_vocab(embeddings_file)
        f = open(self.file)
        labels = []
        sentence = []
        plain_original = []
        data = []
        for line in f:
            splitted = line.split('\t')
            if splitted[0].startswith('-DOCSTART') or splitted[0] == "\n" or len(splitted[0]) == 0:
                if len(sentence) != 0:
                    data.append([sentence, labels, plain_original])
                    sentence = []
                    labels = []
                    plain_original = []
            else:
                sentence.append(splitted[0].lower())
                plain_original.append(splitted[0].lower())
                labels.append(splitted[-1].removesuffix('\n'))
            # embed the data
        embedded_data = []
        for d in data:
            mapped_labels = []
            for l in d[1]:
                mapped_labels.append(self.mapping_index[l])
            embedded_data.append(
                [self._generate_embeddings(d[0], self.vocab),
                 self._one_hot_encoding(mapped_labels, len(self.mapping_index)), d[2]])
        self.data = embedded_data

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
                # embeddings to zero if word is not in vocab
                embeddings[i] = torch.zeros(self.embedding_size)
                #embeddings[i] = torch.rand(self.embedding_size)

        return embeddings

    def _generate_vocab(self, embeddings_file):
        # generating vocab by loading embeddings
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
        seq_sent, seq_label, plain_txt = self.data[idx]
        return seq_sent, seq_label, plain_txt
