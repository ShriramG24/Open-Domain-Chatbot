import pandas as pd
import nltk
from nltk.stem import LancasterStemmer

class PreprocessData:
    def clean_data(path):
        with open(path, 'r') as file:
            df = pd.read_json(file)
        
        df.drop(df[(df.eval_score is None) or (df.profile_match == "")].index, inplace=True)
            

        df.to_json('ConvAI2_DS/new_dialogues.json', orient='records')






