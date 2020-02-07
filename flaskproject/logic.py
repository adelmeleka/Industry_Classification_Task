####################### imports ################################

import numpy as np
import csv

from nltk.corpus import stopwords
from nltk import word_tokenize

import nltk
nltk.download('stopwords')

import re

import nltk
nltk.download('punkt')

from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download('wordnet')

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC)
from imblearn.over_sampling import RandomOverSampler

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn import svm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn import svm
from sklearn.model_selection import GridSearchCV

import pickle


############################ Functions ############################## 

def read_dataset():
	#reading csv file
	datafile = open('Job titles and industries.csv', 'r')
	myreader = csv.reader(datafile)

	# Processing Dataset into a Data Frame
	job_title = []
	industry  = []

	for row in myreader:
	  job_title.append(row[0])
	  industry.append(row[1])

	#delete first entry of all arrays
	# as they're corresponding to 'job title','industry' header field
	del job_title[0]
	del industry[0]
	  
	#initializing data frame  
	frame_data =  pd.DataFrame(job_title, columns=["job_title"])
	frame_data["industry"] = industry

	return frame_data

noise_list = set(stopwords.words("english"))
# noise detection
def remove_noise(input_text):
    words = word_tokenize(input_text)
    noise_free_words = list()
    i = 0;
    for word in words:
        if word.lower() not in noise_list:
            noise_free_words.append(word)
        i += 1
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

def lemetize_words(input_text):
    words = word_tokenize(input_text)
    new_words = []
    lem = WordNetLemmatizer()
    for word in words:
        word = lem.lemmatize(word, "v")
        new_words.append(word)
    new_text = " ".join(new_words)
    return new_text

def dataset_cleaning(dataset):
    corpus = []
    
    for line in dataset["job_title"]:
        
        #transform all record into lower case
        review = re.sub('[^a-zA-Z]', ' ', line)
        review = review.lower()
         
        # remove non segnificant words
        #i.e: am, is, are ..ect
        review = remove_noise(review)
        
        #Lexicon Normalization: Lematization
        review = lemetize_words(review)
        
        corpus.append(review)
    
    dataset["cleaned_job_title"] = corpus
    return

def feature_extractor(frame_data):
	## Extract features with count vectorizer

	corpus = frame_data["cleaned_job_title"]

	vectorizer = CountVectorizer(min_df=1)

	X = vectorizer.fit_transform(corpus).toarray()

	return X,vectorizer

def label_extractor(frame_data):
	## Extract Y list containing  

	categories = frame_data["industry"].unique()
	category_dict = {value:index for index, value in enumerate(categories)}

	Y = frame_data["industry"].map(category_dict)

	return Y,category_dict

def split_balance(X,Y):

	## split dataset to 80-20 Train-Test
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1,)

	## balancing dataset
	smote_nc = SMOTENC(categorical_features=[0, 3], random_state=0)
	x_train_res, y_train_res = smote_nc.fit_resample(x_train, y_train)

	return x_train_res, y_train_res,x_test,y_test

def linear_svm(x_train, y_train):
	clf = svm.SVC(kernel='linear', C=1)
	clf.fit(x_train, y_train)
	#save model to disk
	filename = 'finalized_model.sav'
	pickle.dump(clf, open(filename, 'wb'))
	return clf

def model_test(text_input,clf,vectorizer,category_dict):
  test = []
  test.append(text_input)

  vec_text = vectorizer.transform(test).toarray()
  #using linear SVM because more accurate
  search_label = list(category_dict.values()).index(clf.predict(vec_text)[0])
  for name, index in category_dict.items():   
    if index == search_label:
        return name