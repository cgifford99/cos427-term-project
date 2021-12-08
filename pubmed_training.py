import csv
import math
import os
import sys
import time

import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from csv_util import read_file

IMPORT_FILE_ROWS = ['class', 'article']
IMPORT_FILE = os.path.join(os.path.split(
    os.path.abspath(sys.argv[0]))[0], 'pubmed_import.csv')


def read_file(filename, encoding='utf-16'):
    csv_file = open(filename, 'r', encoding=encoding)
    return csv.DictReader(csv_file)

def split_dataset_by_class(dataset, class_list, split_ratio=0.75):
    class_separated_ds = []
    for ds_class in class_list:
        class_separated_ds.append(
            [dict_elem for dict_elem in dataset if dict_elem.get('class') == ds_class])

    training = {'data': [], 'target': []}
    for class_ds_dict in class_separated_ds:
        for class_ds_idx in range(math.ceil(len(class_ds_dict)*split_ratio)):
            training['data'].append(
                class_ds_dict[class_ds_idx].get('abstract'))
            training['target'].append(class_ds_dict[class_ds_idx].get('class'))

    testing = {'data': [], 'target': []}
    for class_ds_dict in class_separated_ds:
        for class_ds_idx in range(math.ceil(len(class_ds_dict)*split_ratio), len(class_ds_dict)):
            testing['data'].append(class_ds_dict[class_ds_idx].get('abstract'))
            testing['target'].append(class_ds_dict[class_ds_idx].get('class'))

    return training, testing


def train_and_predict(model_name, training, testing):
    tf_idf_vectorizer = TfidfVectorizer(input='content', encoding='utf-8', strip_accents='unicode',
                                        lowercase=True, analyzer='word', stop_words='english', use_idf=True, smooth_idf=True)

    model = None
    if model_name == 'nb':
        model = make_pipeline(tf_idf_vectorizer, ComplementNB())
    elif model_name == 'svc':
        model = make_pipeline(tf_idf_vectorizer, SVC())
    elif model_name == 'logreg':
        model = make_pipeline(tf_idf_vectorizer, LogisticRegression(
            solver='lbfgs', max_iter=400))
    else:
        raise NotImplementedError('Invalid model name')
    model.fit(training.get('data'),
              training.get('target'))

    print('Training finished. Evaluating model...')

    predicted_categories_training = model.predict(training.get('data'))
    class_list = numpy.unique(predicted_categories_training)
    training_accuracy = accuracy_score(training.get('target'),
                                       predicted_categories_training)
    print(
        f'Accuracy (seen data), n={len(training.get("target"))}: {str(round(training_accuracy*100, 3))}%')

    predicted_categories_testing = model.predict(testing.get('data'))
    testing_accuracy = accuracy_score(testing.get('target'),
                                      predicted_categories_testing)
    precision, recall, f1score, _ = precision_recall_fscore_support(testing.get(
        'target'), predicted_categories_testing, average=None, labels=class_list, zero_division=1)
    print(
        f'Accuracy (unseen data), n={len(testing.get("target"))}: {str(round(testing_accuracy*100, 3))}%')

    print('Precision: ' + print_list_with_class(precision, class_list))
    print('Recall: ' + print_list_with_class(recall, class_list))
    print('F1 Score: ' + print_list_with_class(f1score, class_list))

    prec_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        testing.get('target'), predicted_categories_testing, average='macro')
    print('Precision (Averaged): ' + str(round(prec_avg, 3)))
    print('Recall (Averaged): ' + str(round(recall_avg, 3)))
    print('F1 Score (Averaged): ' + str(round(f1_avg, 3)))


def print_list_with_class(data_list, class_list):
    newline = '\n    '  # fix for newlines unsupported in f-string in python
    return f'{newline}{newline.join([class_list[idx] + ": " + str(round(data_list[idx],3)) for idx in range(len(data_list))])}'


if __name__ == '__main__':
    class_list = []
    pm_article_data = []
    for article_element in read_file(IMPORT_FILE):
        raw_text = article_element.get('abstract')
        pm_class = article_element.get('class')
        if pm_class not in class_list:
            class_list.append(pm_class)
        pm_article_data.append({'abstract': raw_text, 'class': pm_class})

    training, testing = split_dataset_by_class(pm_article_data, class_list)

    print('\nTraining and evaluating Naive Bayes model')
    nb_st_time = time.time()
    train_and_predict('nb', training, testing)
    nb_total_time = time.time() - nb_st_time
    nb_converted_time = time.strftime("%H:%M:%S", time.gmtime(nb_total_time))
    print("--- Time taken (Naive Bayes): %s ---\n" % nb_converted_time)

    print('\nTraining and evaluating Support Vector Classification model')
    svc_st_time = time.time()
    train_and_predict('svc', training, testing)
    svc_total_time = time.time() - svc_st_time
    svc_converted_time = time.strftime("%H:%M:%S", time.gmtime(svc_total_time))
    print("--- Time taken (SVC): %s ---\n" % svc_converted_time)

    print('\nTraining and evaluating Logistic Regression model')
    lr_st_time = time.time()
    train_and_predict('logreg', training, testing)
    lr_total_time = time.time() - lr_st_time
    lr_converted_time = time.strftime("%H:%M:%S", time.gmtime(lr_total_time))
    print("--- Time taken (Logistic Regression): %s ---\n" % lr_converted_time)
