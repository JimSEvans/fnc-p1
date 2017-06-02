# This is a multi-class classification task. It is part 1 of the Fake News Challenge.
# The code below trains a model using all available training data, and generates predicted labels for the test set.
from pycorenlp import StanfordCoreNLP
from datetime import datetime
#from langdetect import detect
#from langdetect import DetectorFactory
import sys
#import numpy as np
import pandas as pd
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize  
import random
 
#DetectorFactory.seed = 0
random.seed(1)

bodies = pd.DataFrame.from_csv("train/train_bodies.csv")
stances = pd.DataFrame.from_csv("train/train_stances.csv", index_col=None)
#all_data = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)

# remember to start Stanford CoreNLP server (perhaps in separate terminal window or tab)
#nlp = StanfordCoreNLP('http://localhost:9000')

def countNeg(string):
    res = string.count("n\'t")
    tokens = string.split()
    tokens2 = [t for t in tokens if (t == 'not' or t == 'cannot')]
    res += len(tokens2)
    return res

def boolNeg(string):
    res = "n\'t" in string
    tokens = string.split()
    tokens2 = [t for t in tokens if (t == 'not' or t == 'cannot')]
    res = res or (len(tokens2) > 0)
    return res

#bodies['bodyBoolNeg'] = bodies.apply(lambda row: boolNeg(row['articleBody']), axis=1)
#bodies['bodyNeg'] = bodies.apply(lambda row: getNeg(row['articleBody']), axis=1)
#stances['stanceBoolNeg'] = stances.apply(lambda row: getNeg(row['Headline']), axis=1)



#def getNegWords(text):
#    output = nlp.annotate(text, properties={'annotators': 'depparse','outputFormat': 'json'})
#    try:
#        sentence_data = output['sentences'][0]
#        dep_parse = sentence_data['basicDependencies']
#        negated = []
#        for dikt in dep_parse:
#            if dikt['dep']=='neg':
#                negated.append(dikt['governorGloss'])
#        return(" ".join(negated))
#    except:
#        return("")

def splitIntoSets(stance_df, percentage):
# disagree
    dg = stance_df[stance_df['Stance']=='disagree']['Body ID'].tolist()
    uniqdg = set(dg)
    dg_test_set_size = round(len(uniqdg)*percentage)
    dg_test_set = set(random.sample(uniqdg, dg_test_set_size))
    dg_train_set = uniqdg - dg_test_set
# agree    
    ag = stance_df[stance_df['Stance']=='agree']['Body ID'].tolist()
    uniqag_raw = set(ag)
    uniqag = uniqag_raw - uniqdg
    ag_test_set_size = round(len(uniqag)*percentage)
    ag_test_set = set(random.sample(uniqag, ag_test_set_size))
    ag_train_set = uniqag - ag_test_set
# discuss    
    dc = stance_df[stance_df['Stance']=='discuss']['Body ID'].tolist()
    uniqdc_raw = set(dc)
    uniqdc = uniqdc_raw - uniqag - uniqdg
    dc_test_set_size = round(len(uniqdc)*percentage)
    dc_test_set = set(random.sample(uniqdc, dc_test_set_size))
    dc_train_set = uniqdc - dc_test_set
    
    train_ids = list(dg_train_set | ag_train_set | dc_train_set)
    test_ids = list(dg_test_set | ag_test_set | dc_test_set)
    test_ids_df = pd.DataFrame({'Body ID':test_ids})
    train_ids_df = pd.DataFrame({'Body ID':train_ids})
    train_stances_df = pd.merge(stances, train_ids_df, how='inner', left_on='Body ID', right_on='Body ID', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
    test_stances_df = pd.merge(stances, test_ids_df, how='inner', left_on='Body ID', right_index=True, copy=True, indicator=False)
    return [train_stances_df, test_stances_df]

if 'new' in sys.argv:
    # make a cosine similarity matrix, where documents are the union of article bodies and headlines
    headlines = stances['Headline']
    headline_ids = stances.index
    len_headlines = len(headlines)
    articleBodies = bodies['articleBody']
    len_articleBodies = len(articleBodies)
    documents = headlines.tolist() + articleBodies.tolist()
    tfidf = TfidfVectorizer().fit_transform(documents)
    # no need to normalize, since Vectorizer will return normalized tf-idf
    pairwise_similarity = (tfidf * tfidf.T)
    cos_df = pd.DataFrame(pairwise_similarity.toarray())
    cos_df = cos_df.iloc[0:len_headlines,len_headlines:]
    cos_df.columns = bodies.index
    #cos_df.round(decimals=3).to_csv('cosine_similarity.csv')
    #cos_df = pd.DataFrame.from_csv('cosine_similarity.csv')
    #def getCosineSimilarity(row):
    #    return cos_df.loc[row.name, row['Body ID']]
    #
    #stances['cosineSimilarity'] = stances.apply(lambda row: getCosineSimilarity(row), axis=1)
    #stances.to_csv('stances_w_cos.csv')
else:
    stances = pd.DataFrame.from_csv('stances.csv')
    bodies = pd.DataFrame.from_csv('bodies.csv')
    #bodies['bodyNegatedWords'] = bodies.apply(lambda row: getNegWords(row['articleBody']), axis=1)
    #bodies.to_csv('bodies_new.csv')
    #stances['headlineNegatedWords'] = stances.apply(lambda row: getNegWords(row['Headline']), axis=1)
    #stances.to_csv('stances_new.csv')
    #stances['detectedLang'] = stances.apply(lambda row: detect(row['Headline']), axis=1)
    #bodies['detectedLang'] = bodies.apply(lambda row: detect(row['articleBody']), axis=1)

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

# # This gets all features out of the headline and body column, to be vectorized with sklearn's DictVectorizer
class CosineSimilarityData(BaseEstimator, TransformerMixin):
    """Extract all features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self
    def transform(self, cosSims):
        return [{'cosineSimilarity': cosSim} for cosSim in cosSims]

class QuestionMarkAndNegData(BaseEstimator, TransformerMixin):
    """Extract all features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self
    def countNeg(self, string):
        res = string.count("n\'t")
        tokens = string.split()
        tokens2 = [t for t in tokens if (t == 'not' or t == 'cannot')]
        res += len(tokens2)
        return res
    def boolNeg(self, string):
        res = "n\'t" in string
        tokens = string.split()
        tokens2 = [t for t in tokens if (t == 'not' or t == 'cannot')]
        res = res or (len(tokens2) > 0)
        return res
    def transform(self, strings):
        return [
                {
                    'question_mark_count': string.count("?"),
                    'question_mark_bool': "?" in string,
                    'neg_count': self.countNeg(string),
                    'neg_bool': self.boolNeg(string)
                    } for string in strings
                ]
# custom tokenizer to process text strings
#class StemTokenizer(object):
#    def __init__(self):
#        self.stemmer = PorterStemmer()
#    def stem_tokens(self, tokens, stemmer):
#        stemmed = []
#        for item in tokens:
#            stemmed.append(stemmer.stem(item))
#        return stemmed
#    def tokenize(self, text):
#        lowercase_text = text.lower()
#        tokens = word_tokenize(lowercase_text)
#        filtered = [w for w in tokens if not w in stopwords.words('english')]
#        return filtered
#    def tokenize_and_stem(self, text):
#        tokens = self.tokenize(text)
#        if tokens is None:
#            print("\ntokens is none\n" + text)
#        stems = self.stem_tokens(tokens, self.stemmer)
#        return stems
#    def __call__(self, doc):
#        return self.tokenize_and_stem(doc)


# custom tokenizer to process text strings
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
            print("\ntokens is none\n" + text)
        stems = self.stem_tokens(tokens, self.lemmatizer)
        return stems
    def __call__(self, doc):
        return self.tokenize_and_lemmatize(doc)

classifier = Pipeline([
    # Combining complaint text features, date features, zip_code features, and state
    ('union', FeatureUnion(
        transformer_list=[
            # Pipeline for standard bag-of-words TF-IDF stemmed model for body
#            ('articleBodyBoW', Pipeline([
#                ('selector', ItemSelector(key='articleBody')),
#                ('tfidf', TfidfVectorizer(tokenizer=LemmaTokenizer())),
#                # ('best', TruncatedSVD(n_components=150)),
#            ])),
#            ('HeadlineBoW', Pipeline([
#                ('selector', ItemSelector(key='Headline')),
#                ('tfidf', TfidfVectorizer(tokenizer=LemmaTokenizer())),
#                # ('best', TruncatedSVD(n_components=150)),
#            ])),
            # Featurizes dates according to DateData's transform method
            ('QuestionMarkAndNegData', Pipeline([
                ('selector', ItemSelector(key='Headline')),
                ('features', QuestionMarkAndNegData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
            ('cosineSimilarity', Pipeline([
                ('selector', ItemSelector(key='cosineSimilarity')),
                ('features', CosineSimilarityData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ]))
        ],
    )),
    # Use logistic regression
    ('classifier', LogisticRegression(C=0.1,class_weight='balanced'))
    #('classifier', SVC(C=1.0,class_weight='balanced'))
])
QuestionMarkAndNegData
# Rewrote the above pipleline since I was only using the "complaint" features in the final model
#classifier = Pipeline([
#    ('column_selector', ItemSelector(key='complaint')),
#    ('vectorizer', TfidfVectorizer(tokenizer=StemTokenizer())),
#    ('classifier', LogisticRegression(C=5.0))
#    ])

[training_and_dev_set_stance, test_set_stance] = splitIntoSets(stances, 0.15)
[training_set_stance, dev_set_stance] = splitIntoSets(training_and_dev_set_stance, 0.15) 

training_on = training_set_stance
testing_on = dev_set_stance
if "real-test" in sys.argv:
    training_on = training_and_dev_set_stance
    testing_on = test_set_stance

training_set = pd.merge(training_on, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
test_set = pd.merge(testing_on, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)

## just unrelated vs related ******************* !!!!!!!!!!!!!!!!!!!!!!!!!!!!
#training_set['Stance'] = training_set.apply(lambda row: 'unrelated' if row['Stance'] == 'unrelated' else 'related', axis=1)
#test_set['Stance'] = test_set.apply(lambda row: 'unrelated' if row['Stance'] == 'unrelated' else 'related', axis=1)
## ******************* !!!!!!!!!!!!!!!!!!!!!!!!!!!!

classifier.fit(training_set.drop('Stance', axis = 1), training_set['Stance'])

# uncomment the following if testing on dev set
print("getting predicted labels...")
yhat = classifier.predict(test_set)
print(classification_report(test_set['Stance'], yhat))
print(accuracy_score(test_set['Stance'], yhat))

now = datetime.now()
date_parts = [str(x) for x in [now.month,now.day,now.year]]
time_parts = [str(x) for x in [now.hour,now.minute,now.second]]
now_str = "-".join(date_parts) + "_" + "-".join(time_parts)

joblib.dump(classifier, 'classifier' + now_str + '.pkl')
