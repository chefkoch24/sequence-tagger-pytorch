import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np


class SimpleSequenceTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, class_size):
        super(SimpleSequenceTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.class_size = class_size
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            bidirectional=True)
        # hidden layer for mapping to the number of classes
        self.hidden_to_ner = nn.Linear(2 * self.hidden_dim, self.class_size)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, sentence):
        output, _ = self.lstm(sentence)
        labels = self.hidden_to_ner(output)
        labels = F.softmax(labels, dim=2)
        return labels

    def train(self, data, learning_rate=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for x, y, plain_txt in data:
            outputs = self.forward(x)  # forward pass to get outputs of lstm
            optimizer.zero_grad()  # calculate the gradient, manually setting to 0
            # obtain the loss function
            loss = self.loss_function(outputs, y)
            loss.backward()  # calculates the loss of the loss function
            optimizer.step()
        return loss.item()

    def evaluate(self, data):
        # initalize statistics: confusion matrix for statistics and calculation of F1 scores, word statistics where model is successful and fails
        confusion_matrix = np.zeros((self.class_size, self.class_size))
        word_statistics_success, word_statistics_fail = {}, {}
        context_statistics = {}
        for x, y, plain_text in data:
            outputs = self.forward(x)
            # map labels back from one hot encodings for the whole sequence
            pred_labels = self._get_labels(outputs)
            real_labels = self._get_labels(y)
            # generate statistics incl. plain text for easier human analysis
            for i in range(len(pred_labels)):
                confusion_matrix[real_labels[i], pred_labels[i]] += 1
                if real_labels[i] == pred_labels[i]:
                    word_statistics_success[plain_text[i][0]] = word_statistics_success.setdefault(plain_text[i][0],0) + 1
                else:
                    word_statistics_fail[plain_text[i][0]] = word_statistics_fail.setdefault(plain_text[i][0], 0) + 1
                    context = ""
                    if i != 0:
                        context += str(plain_text[i - 1][0] + " " + plain_text[i][0] + " ")
                    if i != len(pred_labels) - 1:
                        context += plain_text[i + 1][0]
                    context_statistics[context] = context_statistics.setdefault(context, 0) + 1
        # sorting to have most common errors at the top
        word_statistics_fail = sorted(word_statistics_fail.items(), key=lambda x: x[1], reverse=True)
        word_statistics_success = sorted(word_statistics_success.items(), key=lambda x: x[1], reverse=True)
        context_statistics = sorted(context_statistics.items(), key=lambda x: x[1], reverse=True)
        f1_scores = self._calculate_f1(confusion_matrix)
        return {'f1_scores': f1_scores,
                'word_statistics': {'success': word_statistics_success, 'fail': word_statistics_fail},
                'confusion_matrix': confusion_matrix, 'context_statistics': context_statistics}

    def _calculate_f1(self, confusion_matrix):
        # calculation of the f1 scores
        classes = len(confusion_matrix)
        curr_class = 0
        precison, recall = [], []
        sum_tp, sum_fn, sum_fp = 0, 0, 0
        for c in confusion_matrix:
            tp = c[curr_class]
            sum_tp += tp
            # calculate recall
            sum_of_all_elements_in_class = np.sum(c) # sum of row in confusion matrix
            sum_fn += sum_of_all_elements_in_class
            if sum_of_all_elements_in_class > 0:
                r = tp / sum_of_all_elements_in_class  # calculate recall per class that already includes the tp
            else:
                r = 0
            recall.append(r)
            # calculate precision
            sum_of_all_classified_elements_class = np.sum(confusion_matrix[:, curr_class]) # sum of column in confusion matrix
            sum_fp += sum_of_all_classified_elements_class
            if sum_of_all_classified_elements_class > 0:
                p = tp / sum_of_all_classified_elements_class  # calculate precison per class that already includes the tp
            else:
                p = 0
            precison.append(p)
            curr_class += 1  # set to next class
        # macro recall
        macro_recall = sum(recall) / classes
        macro_precison = sum(precison) / classes
        macro_f1 = (2 * macro_precison * macro_recall) / (macro_precison + macro_recall)
        # micro recall
        micro_recall = sum_tp / sum_fn
        micro_precison = sum_tp / sum_fp
        micro_f1 = (2 * micro_precison * micro_recall) / (micro_precison + micro_recall)
        return {'macro': macro_f1, 'micro': micro_f1}

    # map the labels back from encoding
    def _get_labels(self, encoded_labels):
        seq_labels = []
        for el in encoded_labels[0]:
            arr = el.cpu().detach().numpy()
            seq_labels.append(np.argmax(arr))
        return np.array(seq_labels)

    def predict(self, data):
        for d in data:
            outputs = self.lstm.forward(d[0])
        return outputs
