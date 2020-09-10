## imports
import pandas as pd
import math
import pickle
import os
import joblib

## importing stopwords
from nltk.corpus import stopwords


## dictionaries needed to calculate tf-df for all ages
tf_dict = []
count_dict = []
idf_dict = []
tfidf_dict = []
word_dict = []
tfidf_vector = []


length = 0
# need the path for the data file 
file_name = 'D:\\college\\Digital_System\\Book_Recommender_System\\data\\books.csv'

stop = stopwords.words('english')

def save_dicts():
    get_tfidf(file_name)
    save_word_dict()
    save_tfidf_dict()
    # save_tfidf_vectors()

def get_tfidf(file_name):
    global tf_dict
    global count_dict
    global idf_dict
    global tfidf_dict
    global word_dict
    global tfidf_vector
    global length

    data = pd.read_csv(file_name)
    sample = data['Plot_Summary']
    sample = clean_data(sample)
    # print(data)
    length = len(data)

    for summ in sample:
        tf_dict.append(find_tf(summ))

    # print(tf_dict[0])
    count_dict = find_count()
    idf_dict = find_idf()
    tfidf_dict = [find_tf_idf(sample) for sample in tf_dict]

    word_dict = sorted(count_dict.keys())

    tfidf_vector = [find_tfidf_vector(rev) for rev in tfidf_dict]

def clean_data(data_to_clean):
    data_to_clean = data_to_clean.apply(lambda x: x.lower())
    data_to_clean = data_to_clean.apply(lambda x: [item for item in x.split() if item not in stop])

    return data_to_clean

def save_word_dict():
    '''
        save the word dictionary generated to a text file
    '''
    global word_dict
    with open('Alg/all_word_dict.txt', 'w', encoding='utf8') as f:
        for item in word_dict:
            f.write("%s\n" % item)

def save_tfidf_dict():
    '''
        save the tf's generated to a csv file
    '''
    global tfidf_dict
    with open("Alg/tfidf_dict_all.pkl", 'wb') as handle:
                    pickle.dump(tfidf_dict, handle)


def save_tfidf_vectors():
    '''
        save the tfidf vectors generated
    '''
    global tfidf_vector
    with open("Alg/tfidf_vectors_all.pkl", 'wb') as handle:
                    pickle.dump(tfidf_vector, handle)

## code from 
# https://streamsql.io/blog/tf-idf-from-scratch
# original comments are kept

def find_tf(sample):

    """ Returns a tf dictionary for each review whose keys are all
    the unique words in the review and whose values are their
    corresponding tf.
    """
    # Counts the number of times the word appears in review
    new_dict = {}
    for word in sample:
        if word in new_dict:
            new_dict[word] += 1
        else:
            new_dict[word] = 1
    # Computes tf for each word
    for word in new_dict:
        new_dict[word] = new_dict[word] / len(sample)
    return new_dict

def find_count():
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    global count_dict
    count_dict = {}
    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    for sample in tf_dict:
        for word in sample:
            if word in count_dict:
                count_dict[word] += 1
            else:
                count_dict[word] = 1
    return count_dict

def find_idf():
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    global length
    global count_dict

    idfDict = {}
    for word in count_dict:
        idfDict[word] = math.log(length / count_dict[word])
    return idfDict

def find_tf_idf(tfdict):
    """ Returns a dictionary whose keys are all the unique words in the
    review and whose values are their corresponding tfidf.
    """
    global idf_dict
    tfidfdict = {}
    #For each word in the review, we multiply its tf and its idf.
    for word in tfdict:
        tfidfdict[word] = tfdict[word] * idf_dict[word]
    return tfidfdict



def find_tfidf_vector(sample):
    '''
        returns the tf-idf vectors
    '''
    global word_dict
    tfidfVector = [0.0] * len(word_dict)

      # For each unique word, if it is in the review, store its TF-IDF value.
    for i, word in enumerate(word_dict):
          if word in sample:
              tfidfVector[i] = sample[word]
    return tfidfVector
## end code 


## uncomment the following line and run this file
## in python to save the dictionaries

save_dicts()