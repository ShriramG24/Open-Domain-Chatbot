import pandas as pd
import nltk
from nltk.stem import LancasterStemmer

class PreprocessData:
    def clean_data(self, path):
        with open(path, 'r') as file:
            df = pd.read_json(file)
        
        
        
        df.to_json('new_dialogues.json', orient='records')

dataset = PreprocessData()
dataset.clean_data('dialogues.json')




