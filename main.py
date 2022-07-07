from torch.utils.data import DataLoader
from model import SimpleSequenceTagger
from ner_dataset import NERDataset
import numpy as np


# TODO: answer theoretical questions

# Note: I implemented the whole project not on a gpu because my laptop hasn't one
def main():
    dev_file = './data/dev.conll'  # path to training data
    test_file = './data/test.conll'  # path to validation data
    train_file = './data/train.conll'  # path to test data
    num_epochs = 20
    train_dataset = NERDataset(file=train_file)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    dev_dataset = NERDataset(file=dev_file)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True)
    test_dataset = NERDataset(file=test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    seq_tagger = SimpleSequenceTagger(input_dim=50, hidden_dim=100, num_layers=1, class_size=9)
    for epoch in range(num_epochs):
        loss = seq_tagger.train(data=train_dataloader, learning_rate=0.01)
        print("Iteration:", epoch, "Loss:", loss)
        metrics = seq_tagger.evaluate(dev_dataloader)
        print('DEV-Data','macro', metrics['f1_scores']['macro'], 'micro', metrics['f1_scores']['micro'])

    # evaluate on test data
    metrics = seq_tagger.evaluate(test_dataloader)
    print('TEST-Data','macro', metrics['f1_scores']['macro'], 'micro', metrics['f1_scores']['micro'])
    # print further evaluations
    print('TEST-Data', 'wrong_predicted_words',  metrics['word_statistics']['fail'])
    print('TEST-Data', 'context_statistics', metrics['context_statistics'])
    print('TEST-Data', 'confusion_matrix', metrics['confusion_matrix'])


if __name__ == "__main__":
    main()
