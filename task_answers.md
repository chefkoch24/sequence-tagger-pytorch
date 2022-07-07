1) The results for macro and micro F1-score are different. The macro F1 score is much lower than the micro F1 score.
After training of 20 epochs I got the following F1 scores on the test data:
macro-F1=0.5725304003418387, micro-F1=0.913556584472919
The difference in the calculation is that the macro F1 score is the unweighted mean of all F1 scores for every class. 
The micro F1 score is the global F1 score by the sum over all classes.
For this task, I would use the macro F1 score. Because the distribution of classes is heavily imbalanced for the provided data. In general most of the tokens in the data are not named entities. By using the  
macro F1 score, all classes are handled with equal importance in the evaluation.

2) My model fails most often for tokens that have ambiguities in the classes. 
As you can see in the data excerpt of the five most often wrong predicted tokens those tokens exist with ambiguous classes in the dataset.
For example, the token "new" occurs in named entities like "new york" but is also a common adjective with class 0. 
The same happens for tokens like "world", which occurs as part of named entities e.g. "World Trade Organisation (WTO)" and even in the class Organisation but also in "World Cup" as class Miscellaneous, and also as a usual noun with class 0.
Also, the three other tokens have the problem they have different class labels.
Another minor error of my model occurs by out of vocabulary words like "bre-x".

First five wrong predicted tokens: [('new', 74), ('world', 68), ('of', 36), ('york', 31), ('national', 25)]

4) According to the survey from Yadav and Bethard [1] state-of-art results use more advanced model architectures.
One improvement is adding a CRF layer on top of the LSTM, which increased the performance of those models. 
Another mentioned enhancement is a character level prediction and the combination of word and character level architectures. 
This type of model is also implemented in the work of Lample et al. [2]. In the same study, Lample et al. mentioned that embeddings are crucial for the success of the model. They used pre-trained embeddings from Collobert et al. [3].
For the minor problem of OOV words, Wang et al. [4] proposed a model that used the context (span) of the word and added an information optimization layer. 

[1] https://arxiv.org/pdf/1910.11470.pdf
[2] https://arxiv.org/pdf/1603.01360.pdf
[3] https://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf
[4] https://arxiv.org/pdf/2204.04391.pdf