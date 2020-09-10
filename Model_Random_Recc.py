#imports
import math
import random
import pandas as pd
import time
from datetime import datetime

from Data.Data import read_file

class Random_Recc:
    def __init__(self, age, genre):
        self.age = age
        self.genre = genre
        self.dataset = read_file()
        self.dataset_2 = read_file()
    
    def get_random_reccs(self):
        start_time = datetime.now()

        if (self.age == ''):
            recc = dict.fromkeys(range(5), []) 
        else:
            chk_age = self.check_age(self.age)

            if(chk_age):
                self.check_genre(self.genre)
                recc = self.get_rand()
            else:
                self.check_genre(self.genre)
                recc = self.get_rand()
        
        end_time =datetime.now()
        time_take = end_time - start_time
        print("Time taken: ",time_take)
        
        return (recc)

    def get_rand(self):
        '''
            returns random reccs
        '''
        dt = self.dataset
        indexlist = list(dt.index.values) 
        datas = self.dataset_2

        # print(len(indexlist))
        neww = []
        for i in range(5):
            neww.append(random.choice(indexlist))

        # recc = dict.fromkeys(range(5), [])
        recc = [{'title': datas["Book_Title"].iloc[index], 'author': datas["Book_Author"].iloc[index], 'genre': datas["Genre"].iloc[index]}
                for index in neww]
        return (recc)

    def check_age(self, age):
        '''
            checks the age
            if  age less than 18, data set will only contain children's lit or young adult genre
        '''
        int_age = int(age)
        dt = self.dataset
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



