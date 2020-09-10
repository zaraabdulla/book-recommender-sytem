## imports
import pandas as pd
import re
import math
import pickle
import csv

from Data.Data import read_file
from Alg.Precomp_Tfidf import read_tfidf

from datetime import datetime
## importing stopwords
from nltk.corpus import stopwords
## importing word_tokenize to convert sentences into words
from nltk.tokenize import word_tokenize
## measures cosine similiary between two vectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter

class Recommendation_scratch:
    def __init__(self, age, genre, description):
        self.age = age
        self.genre = genre
        self.description = description
        self.dataset = read_file()
        self.dataset_2 = read_file()
        self.word_dict = self.read_word_dict()
        self.tfidf_dict = self.read_tfidf_dict()
        self.init_length = 0

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
            first precomputed tfidf values (for the dataset) are read
            then tfidf vectors are calculated for the dataset after filtering is done
            the input description is cleaned and vectorized
            sentence vector and tfidf vector are matched using find_cosine_similarity() function
            it returns top_sim which contains indexes of the matches
            recc is returned as a dict based on the dataset values matching index in top_sim
        '''
        dt = self.dataset

        ranges = len(dt)
        self. init_length = ranges

        indexlist = list(dt.index.values)
        
        tdict = [None] * ranges

        for i in range(len(indexlist)):
            tdict[indexlist[i]] = self.tfidf_dict[indexlist[i]]

        tdict_vector = [self.calc_tfidf_vect(review) for review in tdict]
        
        token_text = self.clean_text(text)
        sentence_vec = self.sent_vec(token_text)

        top_sim = self.find_cosine_similarity(tdict_vector, sentence_vec)

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

    def sent_vec(self, text_tokens):
        '''
            for each word in the tokenised description,
            if word is in tfidf dict, the tfidf value is retrieved
            sent_ic is a dictionary where the key is the word and 
            the value is the tfidf value which was found
            then sentence_vector is found using calc_tfidf_vect() function
        '''
        d = {}
        var = []

        for word in text_tokens:
            for i in range (self.init_length):
                d = self.tfidf_dict[i]
                dicti = {}
                
                if d.get(word) is not None:
                    dicti[word] =  d.get(word)
                    var.append(dicti[word]) 
                    break
                else:
                    pass
        
        sent_ic = dict(zip(text_tokens, var))
        sentence_vector = [self.calc_tfidf_vect(sent_ic)]

        return sentence_vector

    def read_tfidf_dict(self):
        '''
            read precomputed tf-idf values
        '''
        tfidf_val =  pickle.load(open("Alg/tfidf_dict_all.pkl", "rb" ) )
        return tfidf_val
    
    def read_word_dict(self):
        '''
            read precomputed word dictionary
        '''
        word_dict = []
        with open('Alg/all_word_dict.txt', 'r', newline='', encoding='utf-8') as read_obj:
            for row in csv.reader(read_obj, quoting=csv.QUOTE_NONE):
                word_dict.append(row[0])  
        return word_dict

    def calc_tfidf_vect(self, review):
        '''
            calculate the tf-idf vectors
        '''
        tfidfVector = [0.0] * len(self.word_dict)
        if(review):
            # For each unique word, if it is in the review, store its TF-IDF value.
            for i, word in enumerate(self.word_dict):
                if word in review:
                    tfidfVector[i] = review[word]
        else:
            pass
        return tfidfVector

# m = Recommendation_scratch(20, 'all', 'pirates')
# sum = m.get_recommendation()
