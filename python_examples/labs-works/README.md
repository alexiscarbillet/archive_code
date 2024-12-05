# labs-works
labs works for the social media data analysis courses taken in itmo university.

The goals are to predict if the tweets are positives, neutrals or negatives and to determine what are the topic of the tweet (Google, Tweeter, Apple or Microsoft).

In order to realize this sentiment analysis, the tweet are converted in interger by using the pca method.
Then, 10 different machine learning methods are used and compared.

Two studies have been done, one using the time of the tweets and another without using it.

This work is available in pdf, jupiter notebook and in python script.

According to the results obtained during the test step, the best classifier for the sentiment prediction is the kNN 3 neighbors with a F1 score of 0.4659178378204406 with the time column in input and it is the kNN 7 neighbors with a F1 score of 0.45955869019351836 without the time column in input.

About the topic prediction, with the time column, the best classifier is the tree with a F1 score of 0.9900990099009901 and without the time column, the best classifier is still the tree but with a F1 score of 0.3173798098399649.

The time column is very important for the topic prediction. In fact, most tweets about the same topic have been collected in short amount of time. This explains the important influence of this column on the topic prediction.
