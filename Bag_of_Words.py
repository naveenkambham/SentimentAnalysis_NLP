#author : Naveen Kambham
#This file contains the python code to vectorize each document based on vocbulary
from os import listdir
import string
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from numpy import array 

def prepare_docs(dir,vocabfile):
	vocab = open(vocabfile,'r')
	vocab = vocab.read()
	vocab = vocab.split()
	vocab =set(vocab)
	lines=list()

	for file in listdir(dir):
		#reading the each file
		data_file = open(dir+'/'+file,'r')
		data = data_file.read()
		data_file.close()

		#cleaning the data by removing stop words, non anlphabets and removing puncations
		tokens = data.split()
		table  = str.maketrans("","",string.punctuation)
		tokens =[s.translate(table) for s in tokens] 
		stop_words = set(stopwords.words('english'))
		tokens =[s for s in tokens if s not in stop_words and s.isalpha()]

		#considering only words in vocab
		tokens =[s for s in tokens if s in vocab]
		line = ' '.join(tokens)
		lines.append(line)

	return lines


def process_docs():
	vocab_filepath =r"E:\Drive\JobHunts\PythonCode\NLP\vocab.txt"
	positive_reviews= prepare_docs(r'E:\Drive\JobHunts\PythonCode\NLP\Movie_Reviews\reviews\pos',vocab_filepath)
	negative_reviews= prepare_docs(r'E:\Drive\JobHunts\PythonCode\NLP\Movie_Reviews\reviews\neg',vocab_filepath)
	
	#tokenizing using Keras tokenizer
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(positive_reviews + negative_reviews)
	x_train = tokenizer.texts_to_matrix(positive_reviews+negative_reviews,mode='freq')
	#making it as a classification model and filling 1 for +ve review and 0 for -ve review
	y_train = array([1 for _ in range(len(positive_reviews))] + [0 for _ in range(len(negative_reviews))])

	#Testing data
	positive_reviews_test= prepare_docs(r'E:\Drive\JobHunts\PythonCode\NLP\Movie_Reviews\testing\pos',vocab_filepath)
	negative_reviews_test= prepare_docs(r'E:\Drive\JobHunts\PythonCode\NLP\Movie_Reviews\testing\neg',vocab_filepath)
	x_test = tokenizer.texts_to_matrix(positive_reviews_test+negative_reviews_test,mode='freq')
	y_test = array([1 for _ in range(len(positive_reviews_test))] + [0 for _ in range(len(negative_reviews_test))])
	

	#buidling simple NN model
	model = Sequential()

	model.add(Dense(50,input_shape=(x_test.shape[1],),activation='relu'))
	model.add(Dense(1,activation='sigmoid'))

	model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
	model.fit(x_train,y_train,epochs =50,verbose = 0)

	loss,acc = model.evaluate(x_test,y_test,verbose =0)
	print(acc *100)

	
process_docs()