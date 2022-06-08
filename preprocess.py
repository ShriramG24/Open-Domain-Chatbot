import json
import nltk
from nltk.stem import LancasterStemmer

class NLPPreprocess:
    def __init__(self, stemmer):
        self.stemmer = stemmer

    def tokenize(self, dialogue):
        return nltk.word_tokenize(dialogue)
        
    def stem(self, word):
        return self.stemmer.stem(word.lower())

    def bag_of_words(sentence, word_bank):
        pass






