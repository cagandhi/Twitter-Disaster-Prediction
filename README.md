# Twitter-Disaster-Prediction

A Bidirectional LSTM model to classify whether a given tweet talks about a real disaster or not. This was my project in "CSC 522: Automated Learning and Data Analysis" course at NC State University.

## Dataset
The data can be found on the Kaggle challenge Data Page here: [Real or Not? Kaggle](https://www.kaggle.com/c/nlp-getting-started/data/).

## Methodology

### Text Pre-processing:

We used standard text pre-processing techniques of lowering the text, lemmatization, removing stop words and punctuations. However, we also saw that since we have text of the tweets in our data, people embed links and even emojis in the text. Thus, we also performed the added step of removing html links and emojis from the tweet text.

### Approach 1: Traditional ML Algorithms

We used two techniques in Bag of Words model - CountVectorization and TF-IDF Vectorization. The first technique simply generates a vector of the length of the vocabulary with the entry for corresponding word being the count of the word in the tweet. The second technique generates a vector of the length of the vocabulary with the entry for corresponding word being the TF-IDF score of the word in the tweet. TF-IDF score captures the relative importance of the word in all the tweets. A high TF-IDF score signifies that the word appears frequently in the tweets but does not appear extremely frequently in all the tweets such as stop words. The features created were passed as input to traditional ML algorithms such as SVM, Logistic Regression, Multinomial Naive Bayes, etc. 

### Approach 2: GloVe Embedding + LSTM model

We performed simple text-preprocessing steps without removing stop words since they would disrupt the semantic of the tweets. We used a Twitter pre-trained GloVe embedding layer and represented each word as a 100-dimensional vector. The maximum length of the tweet was limited to 32 words. Tweets with more than 32 words were truncated at the end and those with less words were padded with zero vectors. These vectors were then passed to a Bidirectional LSTM layer which consisted of 32 LSTM cells. The output from these cells were passed through 4 Dense layers and the final layer output the probability of a tweet being real.

### Approach 3: GloVe Embedding + LSTM model + weighted probability score

The approach is the same as approach 2 till the probability step. We took into account the percentage of tweets that were real disaster tweets for each keyword associated with the tweet in the training data. We calculated the probability of being a real disaster tweet for the test data using the LSTM network with pre-trained GloVe embeddings. Then for each tweet that had an associated keyword, we took the average of the probability calculated by the LSTM network and the probability obtained by dividing the percentage of tweets that were real for the keyword.

More details about the approaches used along with the results achieved can be found in the [Presentation](Disaster_Prediction_Presentation.pdf) and the [Project Report](Disaster_Prediction_Report.pdf) in the repository.
