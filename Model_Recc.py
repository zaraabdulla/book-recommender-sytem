## imports
import pandas as pd
import re
import math
import random
import pickle
import time
from datetime import datetime

from Data.Data import read_file
from Alg.Precomp_Tfidf import read_tfidf

## importing stopwords
from nltk.corpus import stopwords
## importing word_tokenize to convert sentences into words
from nltk.tokenize import word_tokenize
## measures cosine similiary between two vectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter

class Recommendation:
    def __init__(self, age, genre, description):
        self.age = age
        self.genre = genre
        self.description = description
        self.dataset = read_file()
        self.dataset_2 = read_file()
        self.tfidf = read_tfidf()
        
    def clean_data(self, data_to_clean):
        stop = stopwords.words('english')
        data_to_clean = data_to_clean.apply(lambda x: x.lower())
        data_to_clean = data_to_clean.apply(lambda x: [item for item in x.split() if item not in stop])

        return data_to_clean

    def get_recommendation(self):
        '''
            gets the recommendation by calling read_precomp_dicts() to read datafile
            then based on age and genre, dataset is grouped
            description is passed on to get_top_reccs() which will get top matches
        '''
        start_time = datetime.now()
        
        if (self.age == '' or self.description == ''):
            recc = dict.fromkeys(range(5), []) 
        else:
            chk_age = self.check_age(self.age)

            if(chk_age):
                self.check_genre(self.genre)
                recc = self.get_top_reccs(self.description)
            else:
                self.check_genre(self.genre)
                recc = self.get_top_reccs(self.description)
        end_time = datetime.now()
        time_take = end_time - start_time
        
        print("Time taken: ",time_take)
        return (recc)

    def check_age(self, age):
        '''
            checks the age
            if  age less than 18, data set will only contain children's lit or young adult genre
        '''
        
        dt = self.dataset
        int_age = int(age)

        if (int_age >= 18):
            
            return True
        else :
            self.dataset = dt[dt['Genre'].str.contains("Children's literature") | dt['Genre'].str.contains('Young Adult literature')]
            return False

    def check_genre(self, genre):
        '''
            checks genre
            if genre is anything other than 'all' dataset is grouped
            by the given genre
        '''
        dt = self.dataset

        if(genre == 'all'):
            pass
        else:
            self.dataset = dt[dt['Genre'].str.contains(genre, case = False)]

    def get_top_reccs(self, text):
        '''
            retrieves the top matches
            first precomputed tfidf vectors are read
            then both the new grouped dataset and the description are fitted and transformed
            (word counts, idf and tfidf are computed at once)
            tfidf_1 and tfidf_2 are sparse matrixes which are passed to find_cosine_similarity()
            which will find the cosine similary between the two
            it returns top_sim which contains indexes of the matches
            recc is returned as a dict based on the dataset values matching index in top_sim
        '''
        dt = self.dataset
        tf1 = self.tfidf
        sample = dt['Plot_Summary']

        tfidf_new = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True,
                            max_features = 100000, vocabulary = tf1.vocabulary_)

        tfidf_1 = tfidf_new.fit_transform(sample)
        tfidf_2 = tfidf_new.fit_transform([text])

        top_sim = self.find_cosine_similarity(tfidf_1, tfidf_2)

        recc = [{'title': dt["Book_Title"].iloc[index], 'author': dt["Book_Author"].iloc[index], 'genre': dt["Genre"].iloc[index]}
                for index in top_sim]
        
        return recc


    def clean_text(self, sent):
        '''
            convert the sentence into words and remove stop words
        '''
        tokens = word_tokenize(sent)
        rem_stopwords = [word for word in tokens if not word in stopwords.words()]
        return rem_stopwords

    def find_cosine_similarity(self, tfidf_1, tfidf_2):
        '''
            finds the cosine similarity between the two matrixes
            the result is sorted in a descending manner
            returns only the top 5 matches
        '''
        cos_sim = cosine_similarity(tfidf_1, tfidf_2)

        sort_sim = sorted(range(len(cos_sim)),key=cos_sim.__getitem__, reverse=True)
        sort_sim = sort_sim[0:5]
        
        return sort_sim




