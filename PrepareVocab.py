#author : Naveen Kambham
#This file contains the python code to prepare vocabulary from all the available documents

import string
from nltk.corpus import stopwords
from collections import Counter
from os import listdir

def read_doc(file):
	#opening and cleaning the document

	file = open(file,'r')
	data = file.read()
	file.close()
    
    #cleaning the data by removing stop words, non anlphabets and removing puncations
	tokens = data.split()
	table  = str.maketrans("","",string.punctuation)
	tokens =[s.translate(table) for s in tokens] 
	stop_words = set(stopwords.words('english'))

	tokens =[s for s in tokens if s not in stop_words and s.isalpha()]

	return tokens

def read_all_docs(folder,vocab_collection):
    
    for file in listdir(folder):
    	tokens =read_doc(folder+'/'+file)
    	vocab_collection.update(tokens)


def prepareVocab():
    vocab_collection = Counter()
    read_all_docs(r'E:\Drive\JobHunts\PythonCode\NLP\Movie_Reviews\reviews\pos',vocab_collection)
    read_all_docs(r'E:\Drive\JobHunts\PythonCode\NLP\Movie_Reviews\reviews\neg',vocab_collection)
    
    #removing words with less than 5 occurances and saving the words
    words =[key for key,count in vocab_collection.items() if count >= 5]
    words = '\n'.join(words) #adding new line seq
    file =open(r"E:\Drive\JobHunts\PythonCode\NLP\vocab.txt",'w')
    file.write(words)
    file.close()

prepareVocab()
	
	
