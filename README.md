# COS 427 Term Project Final Report

Team members: Christopher Gifford

## Project Description

This project aimed to generate three multi-class classifier algorithms using common text analytics processing techniques in Python and compare their results. The model is trained and tested on medical article abstracts from the free-to-use PubMed repository. The model with the highest accuracy will be used to classify unknown articles, once again utilizing only the abstract of the article.

## Materials and Methods

PubMed is a free-to-use, widely accessible medical research article search engine containing more than 33 million articles from more than 7,000 journals including full-text for more than 4 million of these articles. For the purposes of this program, only the abstracts of the articles were used in training and testing the models. PubMed allows for the search of any medical-related topic. In particular, to reduce the volume of articles and number of classes, we were interested in classifying only the following four topics. Included are the quantity of articles for each topic as well (as of 11/10/21):

* disease,lyme - 3085
* knee osteoarthritis - 16106
* acute rheumatic arthritis - 1738
* abnormalities, cardiovascular - 56138
  
In total, we accumulated roughly 77,000 articles for this program.

The algorithms utilized are the following: Complement Naive Bayes, Support Vector Classification and Logistic Regression. Each of these algorithms can be used for multi-class classification which all fit our use-case well.

Complement Naive Bayes (CNB) was used in place of Multinomial Naive Bayes (MNB) as CNB had much better performance on unbalanced classes, in particular for "disease, lyme" and "acute rheumatic arthritis" given that they had a much lower amount of articles to train from.

Each algorithm was coupled with a TF-IDF-enabled Bag-of-Words implemenation due to its ease of use, high manageability and faster performance over Word2Vec. This technique also showed no clear accuracy deficit as opposed to Word2Vec for this application. With the use of TF-IDF and BoW, I was able to leverage the normalization and tokenization techniques of sklearn in Python, avoiding using NLTK.
I configured the sklearn normalizer to eliminate stop words, special characters and punctuation, lowercase all words, tokenize by word, but stemming and lemmatization was not used and did not affect overall accuracy or performance. In fact, NLTK was much slower in data pre-processing and was omitted from the final program.

## Experimental Validation and Results

Roughly 57,000 articles were used in training each algorithm, and roughly 20,000 articles were used in testing (75:25 training vs. testing ratio). Then, accuracy, precision, recall and F1 score where calculated from the results.

The following is a table comparing the results for each algorithm:
| Algorithm | Accuracy | Precision | Recall | F1 Score |
| ------------- | ----- | ----- | ----- | ------|
| Complement NB | 0.975 | 0.977 | 0.798 | 0.841 |
| SVC           | 0.986 | 0.909 | 0.809 | 0.85 |
| Logistic Regression | 0.984 | 0.984 | 0.894 | 0.932 |

Although SVC obtained a slightly higher accuracy score, Complement Naive Bayes and Logistic Regression trained and tested within 20 sec and 30 sec respectively whereas SVC trained and tested within 50 minutes.

## Conclusion

Due to the long training and testing times of the SVC algorithm, and the lower recall and F1 score of Complement Naive Bayes, I determined Logistic Regression is the optimal model for use in this application. Nonetheless, each algorithm in terms of accuracy performed incredibly well and if training or testing performance is not of much concern, any one of these three algorithms can be used.

## Discussion

As stated previously, the algorithms each performed very well and due to this accuracy cannot be improved by great margins, although some improvements can be made.

Firstly, more data could be collected and trained on from different sources. It is possible that articles within PubMed have a bias of some kind. This brings us to one of the limitations of this process where it may not perform well for articles not in PubMed. Using difference sources of medical articals may alleviate this issue.

Secondly, we could substitute other classification models such as Stocastic Gradiant Descent, Random Forest and others and test their accuracies against the current algorithms.

## Outlook

This program once again proves the high accuracy for these predictive, classifier algorithms when classifying documents. As stated above, other methods may be used to produce similar or better results. This program is not currently deployed for any use case, but there are no restrictions for doing so sometime in the future. When deployed, periodic updates could be made to the data to ensure it is using the most up-to-date data possible.
