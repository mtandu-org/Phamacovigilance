# import important packages
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
np.random.seed(0)

import re
import nltk
import warnings
import numpy as np
import pandas as pd 
import seaborn as sns
from argparse import Namespace
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
# List of features
feature_columns = ['text', 'Lebels']


# A function to load the dataset
def load_data(file_name):
       """
       data loading
       file_name: Show the name of file to load
       """
       df = pd.read_excel(file_name)
       

       return df

def train_test_split(df, train_percent=.8, seed=10):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.iloc[perm[:train_end]]
    test = df.iloc[perm[train_end:]]
    return train, test



# load test dataset
def load_test_data(filename):
	df=load_data(filename)
	name = re.compile('\[.*\]')
	df.text = df.text.str.strip()
	df['text'] = df['text'].apply(lambda x: x.replace(''.join(name.findall(x)), '') if type(x) is str else type(x))
	df['text'] = df['text'].astype(str)
	df['text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
	features  = df.loc[:,"text"].values
	labels    = df.loc[:,"Lebels"].values.astype('int')
	nltk.download('stopwords')
    #df.text = df.text.apply(remove_stopwords)
	tfidf_vectorizer = TfidfVectorizer (max_features=600)
	tfidf_features = tfidf_vectorizer.fit_transform(features).toarray()
	feature = tfidf_features
	label =   labels
	return feature, label




if __name__ == "__main__":
	df=pd.read_excel('../data/data_upsampled.xlsx')
	train, test = train_test_split(df)
	train.to_excel('../data/train.xlsx')
	test.to_excel('../data/test.xlsx')
	print(train.head())
	print(test.shape)
	print(train.shape)
	print(type(train))


