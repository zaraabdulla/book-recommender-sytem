# imports
import pickle
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

file_name = 'D:\\college\\Digital_System\\Book_Recommender_System\\data\\books.csv'

def read_tfidf():
    '''
        reads the precomputed tfidf dictionary
    '''
    tfidf_dict = pickle.load(open("Alg/tfidf_all.pkl", 'rb'))
    return tfidf_dict

def save_tfidf():
    '''
        save tfidf's so it can be reused
    '''
    dataset = pd.read_csv(file_name)
    sample = dataset['Plot_Summary']
    tf_vect = TfidfVectorizer(ngram_range=(1,2), stop_words = "english", max_features = 100000)
    gen_tfidfs = tf_vect.fit(sample)
    pickle.dump(gen_tfidfs, open("Alg/tfidf_all.pkl", "wb"))

# uncomment and run the below line to save the tf-idf values
# save_tfidf()