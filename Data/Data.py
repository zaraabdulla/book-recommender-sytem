## imports
import pandas as pd

## variable for file path relative to main.py
file_name = "data/books.csv"

def read_file():
    '''
        reading the data csv file into the dataframe
        returns the dataframe
    '''
    dataframe = pd.read_csv(file_name)
    # print(dataframe)
    return dataframe

# read_file()

