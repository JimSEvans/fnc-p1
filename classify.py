# This is a multi-class classification task. It is part 1 of the Fake News Challenge.
# The code below trains a model using all available training data, and generates predicted labels for the test set.
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
# from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize  
from sklearn.feature_extraction.text import TfidfVectorizer
import random
# 
random.seed(1)

bodies = pd.DataFrame.from_csv("train/train_bodies.csv")
stances = pd.DataFrame.from_csv("train/train_stances.csv", index_col=None)
#all_data = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)

def splitIntoSets(stancesDataFrame, percentage):
    
    dg = stancesDataFrame[stancesDataFrame['Stance']=='disagree']['Body ID'].tolist()
    uniqdg = set(dg)
    dg_test_set_size = round(len(uniqdg)*percentage)
    dg_test_set = set(random.sample(uniqdg, dg_test_set_size))
    dg_train_set = uniqdg - dg_test_set
    
    ag = stancesDataFrame[stancesDataFrame['Stance']=='agree']['Body ID'].tolist()
    uniqag_raw = set(ag)
    uniqag = uniqag_raw - uniqdg
    ag_test_set_size = round(len(uniqag)*percentage)
    ag_test_set = set(random.sample(uniqag, ag_test_set_size))
    ag_train_set = uniqag - ag_test_set
    
    dc = stancesDataFrame[stancesDataFrame['Stance']=='discuss']['Body ID'].tolist()
    uniqdc_raw = set(dc)
    uniqdc = uniqdc_raw - uniqag - uniqdg
    dc_test_set_size = round(len(uniqdc)*percentage)
    dc_test_set = set(random.sample(uniqdc, dc_test_set_size))
    dc_train_set = uniqdc - dc_test_set
    
    train_ids = list(dg_train_set | ag_train_set | dc_train_set)
    test_ids = list(dg_test_set | ag_test_set | dc_test_set)
    return {'train_ids': train_ids, 'test_ids': test_ids}

test_as_df = pd.DataFrame({'Body ID':test_ids})
train_as_df = pd.DataFrame({'Body ID':train_ids})

#train_stances_df = pd.merge(stances, train_as_df, how='inner', left_on='Body ID', right_on='Body ID', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
#train_df = pd.merge(train_stances_df, bodies, how='inner', left_on='Body ID', right_index=True, copy=True, indicator=False)


## make a cosine similarity matrix, where documents are the union of article bodies and headlines
headlines = stances['Headline']
headline_ids = stances.index
len_headlines = len(headlines)
articleBodies = bodies['articleBody']

len_articleBodies = len(articleBodies)
documents = headlines.tolist() + articleBodies.tolist()
tfidf = TfidfVectorizer().fit_transform(documents)
# no need to normalize, since Vectorizer will return normalized tf-idf
pairwise_similarity = (tfidf * tfidf.T).A
cos_df = pd.DataFrame(pairwise_similarity.toarray())
cos_df = cos_df.iloc[0:len_headlines,len_headlines:]
cos_df.columns = bodies.index

def getCosineSimilarity(row):
    return cos_df.loc[row.name, row['Body ID']]

stances['cosineSimilarity'] = stances.apply(lambda row: getCosineSimilarity(row), axis=1)

# This class is needed to break out the complaint column in the estimation pipeline that comes later

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples. 

    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

#class MultiItemSelector(BaseEstimator, TransformerMixin):
#    """For data grouped by feature, select subset of data at a provided key.
#
#    The data is expected to be stored in a 2D data structure, where the first
#    index is over features and the second is over samples. 
#
#    """
#    def __init__(self, keylist):
#        self.keylist = keylist
#
#    def fit(self, x, y=None):
#        return self
#
#    def transform(self, data_dict):
#        return [data_dict[key] for key in self.keylist]

# # This gets all features out of the headline and body column, to be vectorized with sklearn's DictVectorizer

# paper: 
# polarity/refuting agreement b/w headline & body
## RootDist?
# similarity b/w headline & body
 class CosineSimilarityData(BaseEstimator, TransformerMixin):
     """Extract all features from each document for DictVectorizer"""

     def fit(self, x, y=None):
         return self

     def transform(self, cosSims):
         return [{'cosine_similarity': cosSim} for cosSim in cosSims]


# custom tokenizer to process text strings
class StemTokenizer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def stem_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(self, text):
        lowercase_text = text.lower()
        tokens = word_tokenize(lowercase_text)
        filtered = [w for w in tokens if not w in stopwords.words('english')]
        return filtered

    def tokenize_and_stem(self, text):
        tokens = self.tokenize(text)
        if tokens is None:
            print "\ntokens is none\n" + text
        stems = self.stem_tokens(tokens, self.stemmer)
        return stems

    def __call__(self, doc):
        return self.tokenize_and_stem(doc)


# # custom tokenizer to process text strings (NOT USED IN MY FINAL MODEL)
 class LemmaTokenizer(object):
     def __init__(self):
         self.lemmatizer = WordNetLemmatizer()

     def stem_tokens(self, tokens, lemmatizer):
         lemmatized = []
         for item in tokens:
             lemmatized.append(lemmatizer.lemmatize(item))
         return lemmatized

     def tokenize(self, text):
         lowercase_text = text.lower()
         tokens = word_tokenize(lowercase_text)
         filtered = [w for w in tokens if not w in stopwords.words('english')]
         return filtered

     def tokenize_and_lemmatize(self, text):
         tokens = self.tokenize(text)
         if tokens is None:
             print "\ntokens is none\n" + text
         stems = self.stem_tokens(tokens, self.lemmatizer)
         return stems

     def __call__(self, doc):
         return self.tokenize_and_lemmatize(doc)


# Note: Since I only use complaint text features, I rewrote this pipeline more succinctly under this one.
# I left this here in case you want to see how I made models with other features.
# # Bring together features from all four sources (complaint, date, zip_code, state)

 classifier = Pipeline([
     # Combining complaint text features, date features, zip_code features, and state
     ('union', FeatureUnion(
         transformer_list=[
             # Pipeline for standard bag-of-words TF-IDF stemmed model for body
             ('articleBodyBoW', Pipeline([
                 ('selector', ItemSelector(key='articleBody')),
                 ('tfidf', TfidfVectorizer(tokenizer=StemTokenizer())),
                 # ('best', TruncatedSVD(n_components=150)),
             ])),
             # Featurizes dates according to DateData's transform method
             ('cosineSimilarity', Pipeline([
                 ('selector', ItemSelector(key='cosineSimilarity')),
                 ('features', CosineSimilarityData()),  # returns a list of dicts
                 ('vect', DictVectorizer()),  # list of dicts -> feature matrix
             ])),
             # Featurizes zip codes according to ZipData's transform method
             #('zip_code', Pipeline([
             #    ('selector', ItemSelector(key='zip_code')),
             #    ('features', ZipData()),  # returns a list of dicts
             #    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
             #])),         
             ## Featurizes states according to DateData's transform method (only one feature, the state)
             #('state', Pipeline([
             #    ('selector', ItemSelector(key='state')),
             #    ('features', StateData()),
             #    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
             #]))
         ],
         # weight components in FeatureUnion
         #transformer_weights={
         #    'complaint': 1.0,
         #    'date': 0.1,
         #    'state': 0.1,
         #    'zip_code': 0.1
         #},
     )),
     # Use logistic regression
     ('classifier', LogisticRegression(C=5.0)),
 ])

# Rewrote the above pipleline since I was only using the "complaint" features in the final model
#classifier = Pipeline([
#    ('column_selector', ItemSelector(key='complaint')),
#    ('vectorizer', TfidfVectorizer(tokenizer=StemTokenizer())),
#    ('classifier', LogisticRegression(C=5.0))
#    ])

























#
## read in the data
#full_df_orig = pd.read_csv('train.csv', sep=',')
#full_df = full_df_orig.replace(np.nan,'', regex=True)
#train_df = full_df
#test_df = pd.read_csv('test.csv', sep=',')
#
## # separate train and dev sets
## length = len(full_df)
## first_index_for_dev_set = int(round(length*0.8))
## np.random.seed(1)
## shuffled_df = full_df.iloc[np.random.permutation(len(full_df))]
## train_df = shuffled_df[0:first_index_for_dev_set]
## dev_df = shuffled_df[first_index_for_dev_set:]
#
#train = train_df[['date', 'complaint', 'state', 'zip_code']]
#target = train_df.issue
#
## set test to either dev set or true test set
## test = dev_df[['date', 'complaint', 'state', 'zip_code']]
#test = test_df[['date', 'complaint', 'state', 'zip_code']]
#
#print "training classifier..."
#classifier.fit(train, target)
## joblib.dump(classifier, 'classifier.pkl')
## classifier = joblib.load('classifier.pkl')
#
#print 'getting probabilities...'
#probabilities = classifier.predict_proba(test)
#
#
## # print "making submission file..."
## submission = pd.DataFrame(probabilities, columns=list(classifier.classes_))
## submission.insert(0, 'id', test_df.id)
## submission.to_csv('submission.csv', sep=',', index=False)
#
## uncomment the following if testing on dev set
## print "getting predicted labels..."
## y = classifier.predict(test)
## print classification_report(test_df.issue, y)
## print accuracy_score(test_df.issue, y)
#
#
#
