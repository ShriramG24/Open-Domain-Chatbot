import nltk
import numpy as np
from nltk.stem import LancasterStemmer

def tokenize(dialogue):
    return nltk.word_tokenize(dialogue)
    
def stem(word):
    stemmer = LancasterStemmer()
    return stemmer.stem(word.lower())

def bag_of_words(sentence, word_bank):
    sentence = [stem(word) for word in sentence]
    return np.array([1 if word in sentence else 0 for word in word_bank]).astype(np.float32)
    





