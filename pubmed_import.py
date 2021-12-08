import csv
import os
import sys
import time
import unicodedata

from Bio import Entrez

import csv_util

# Entrex maximum retmax from their documentation. Do not change.
ENTREZ_MAX_RETMAX_SEARCH = int(1e5)
ENTREZ_MAX_RETMAX_FETCH = int(1e4)

# Customizable article limit for each class
MAX_ARTICLE_LIMIT = 1000000

DATA_IMPORT_FILENAME = os.path.join(os.path.split(
    os.path.abspath(sys.argv[0]))[0], 'pubmed_import.csv')


def search_ids(pm_class):
    query = f'({pm_class}[MeSH Terms]) AND (("2010"[Date - Publication] : "3000"[Date - Publication])) AND (fha[Filter])'
    id_list = []
    prev_result_len = ENTREZ_MAX_RETMAX_SEARCH
    for retstart in range(0, MAX_ARTICLE_LIMIT, ENTREZ_MAX_RETMAX_SEARCH):
        if prev_result_len < ENTREZ_MAX_RETMAX_SEARCH:
            break
        search_handle = Entrez.esearch(db='pubmed',
                                       sort='relevance',
                                       retmode='xml',
                                       retmax=str(MAX_ARTICLE_LIMIT),
                                       retstart=str(retstart),
                                       term=query)
        search_results = Entrez.read(search_handle)['IdList']
        prev_result_len = len(search_results)
        id_list.extend(search_results)
    return id_list


def fetch(id_list):
    ids = ','.join(id_list)
    article_list = []
    prev_result_len = ENTREZ_MAX_RETMAX_FETCH
    for retstart in range(0, MAX_ARTICLE_LIMIT, ENTREZ_MAX_RETMAX_FETCH):
        if prev_result_len < ENTREZ_MAX_RETMAX_FETCH:
            break
        fetch_handle = Entrez.efetch(db='pubmed',
                                     retmode='xml',
                                     retmax=str(MAX_ARTICLE_LIMIT),
                                     retstart=str(retstart),
                                     id=ids)
        fetch_results = Entrez.read(fetch_handle)['PubmedArticle']
        prev_result_len = len(fetch_results)
        article_list.extend(fetch_results)
    return article_list


def fetch_and_collect(pm_class):
    id_list = search_ids(pm_class)
    fetch_results = fetch(id_list)
    article_list = []
    for article in fetch_results:
        medline_citation_article_dict = article['MedlineCitation']['Article']
        abstract = ''
        if medline_citation_article_dict.get('Abstract'):
            abstract = ' '.join(
                medline_citation_article_dict['Abstract']['AbstractText']).strip()
            if not abstract:
                continue
        else:
            continue
        abstract = unicodedata.normalize("NFKD", abstract)
        article_list.append({'class': pm_class, 'abstract': abstract})
    return article_list


def write_dict_batch_to_file(filename, fieldnames, batch, encoding='utf-16'):
    csv_file = open(filename, 'w+',
                              newline='', encoding=encoding)
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    for elem in batch:
        csv_writer.writerow(elem)
    csv_file.close()


if __name__ == '__main__':
    Entrez.email = 'christopher.gifford@maine.edu'

    class_list = ['disease, lyme', 'knee osteoarthritis',
                  'acute rheumatic arthritis', 'abnormalities, cardiovascular']

    all_articles = []
    st_time = time.time()
    for pm_class in class_list:
        print('Fetching results for class: ' + pm_class)
        all_articles.extend(fetch_and_collect(pm_class))

    write_dict_batch_to_file(
        DATA_IMPORT_FILENAME, ['class', 'abstract'], all_articles)

    total_time = time.time() - st_time
    convertedTime = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print("--- Time taken: %s ---\n" % convertedTime)
