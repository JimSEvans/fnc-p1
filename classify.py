# This is a multi-class classification task. It is basically part 1 of the Fake News Challenge.

#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# 22563.250 was the top score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
import copy
import util
import sys
import os.path
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from score import report_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC

if 'test' in sys.argv:
    test = 1

body_file = None
stance_file = None

#read into DFs the  article body file and stance/headline file
if os.path.isfile('train/train_bodies.csv'):
    body_file = 'train/train_bodies.csv'
else:
    body_file = 'train/orig_train_bodies.csv'

if os.path.isfile('train/train_stances.csv'):
    stance_file = 'train/train_stances.csv'
else:
    stance_file = 'train/orig_train_stances.csv'

#body_file = 'train/orig_train_bodies.csv'
#stance_file = 'train/orig_train_stances.csv'

w_body_file = 'train/train_bodies.csv' 
w_stance_file = 'train/train_stances.csv' 

bodies = pd.read_csv(body_file, index_col='Body ID')
stances = pd.read_csv(stance_file, index_col=None)
all_data = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, suffixes=('_x', '_y'), indicator=True)

def mkBodyCSV(dataframe):
    dataframe.to_csv(w_body_file, index=True)

def mkStanceCSV(dataframe):
    dataframe.to_csv(w_stance_file, index=False)

if len(all_data[all_data['_merge']!='both']) != 0:
    print ("STOP ! ! ! ! !")

bodycols = list(bodies)
stancecols = list(stances)

print('# bodies ' + str(len(bodies)))
print('# stances ' + str(len(stances)))

if 'bodyLang' not in bodycols:
    print('Using langdetect to detect language of body')
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    bodies['bodyLang'] = bodies.apply(lambda row: detect(row['articleBody']), axis=1)
    mkBodyCSV(bodies)
    #bodies.to_csv(w_body_file, index=True)
#
if 'tr_body' in sys.argv:
#if 'body_translated' not in bodycols:
    print('Using Google Translate to translate body')
    bodies['articleBody'] = bodies['articleBody']
    from googletrans import Translator
    translator = Translator()
    import csv
    import time
    with open('transl_body.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Body ID','articleBody'])
        for i in bodies.index:
            if bodies.loc[i,'bodyLang'] != 'en':
                done = 0
                while not done:
                    try:
                        t = translator.translate(bodies.loc[i,'articleBody'])
                        #print('\n#!#\n#!#\n#!#\n' + t.text)
                        writer.writerow([i,t.text])
                        bodies.loc[i,'articleBody'] = t.text
                        done = 1
                        bodies.loc[i,'body_translated'] = True
                    except: 
                        time.sleep(5)
                        print('waiting 5 seconds, then trying again...')
            else:
                bodies.loc[i,'body_translated'] = False
    mkBodyCSV(bodies)

if 'tr_headline' in sys.argv:
#if 'headline_translated' not in bodycols:
    print('Using Google Translate to translate headline')
    from googletrans import Translator
    translator = Translator()
    stances['headline_src_lang'] = None
    import csv
    import time
    with open('transl_headline.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Headline ID','Headline', 'headline_src_lang'])
        for i in list(stances.index):
            #if bodies.loc[i,'headlineLangLo'] != 'en':
            done = 0
            while not done:
                try:
                    t = translator.translate(stances.loc[i,'Headline'])
                    writer.writerow([i,t.text,t.src])
                    stances.loc[i,'Headline'] = t.text
                    stances.loc[i,'headline_src_lang'] = t.src
                    done = 1
                    bodies.loc[i,'headline_translated'] = True
                except: 
                    time.sleep(5)
                    print('waiting 5 seconds, then trying again...')
    mkStanceCSV(stances)
    #stances.to_csv(w_stance_file, index=False)
#
if 'cosine' not in stancecols:
    print('Getting cosine similarity between articles their headlines.')
    vectorizer = TfidfVectorizer(tokenizer=util.LemmaTokenizer())
    def getCosineSimilarity(text1, text2):
        tfidf = vectorizer.fit_transform([text1, text2])
        res = round(((tfidf * tfidf.T).A)[0,1], 5)
        #print(res)
        return res 
    all_data = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, suffixes=('_x', '_y'), indicator=True)

    stances['cosine'] = all_data.apply(lambda row: getCosineSimilarity(row['articleBody'], row['Headline']),axis=1)
    mkStanceCSV(stances)
    #stances.to_csv(w_stance_file, index=False)


if 'WMD' not in stancecols:
    print('Computing word mover\'s distance.')
    vectors = KeyedVectors.load_word2vec_format(
            "data/GoogleNews-vectors-negative300.bin.gz", binary=True
            )
    vectors_normal = copy.deepcopy(vectors)
    vectors_normal.init_sims(replace=True)
    all_data = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, suffixes=('_x', '_y'), indicator=True)
    stances['WMD'] = all_data.apply(
            lambda row: util.getWordMoversDistance(
                vectors, row['articleBody'], row['Headline']
                ), axis=1)
    stances['WMDnormal'] = all_data.apply(
            lambda row: util.getWordMoversDistance(
                vectors_normal, row['articleBody'], row['Headline']
                ), axis=1)
    mkStanceCSV(stances)
    #stances.to_csv("train/train_stances.csv")

if 'negated_headline' not in stancecols:
    print('Getting negated words from headline with Stanford CoreNLP')
    from pycorenlp import StanfordCoreNLP
    # make sure you start the server in another tab/window first!
    my_nlp = StanfordCoreNLP('http://localhost:9000')
    stances['negated_headline'] = stances.apply(lambda row: util.getNegatedWords(row['Headline'], my_nlp),axis=1)
    mkStanceCSV(stances)

if 'negated_body' not in bodycols:
    print('Getting negated words from body with Stanford CoreNLP')
    from pycorenlp import StanfordCoreNLP
    # make sure you start the server in another tab/window first!
    my_nlp = StanfordCoreNLP('http://localhost:9000')
    bodies['negated_body'] = bodies.apply(lambda row: util.getNegatedWords(row['articleBody'], my_nlp),axis=1)
    mkBodyCSV(bodies)


if 'closest_by_cos' not in stancecols:
    print('Getting closest sentence to headline and related features')
    vectorizer = TfidfVectorizer(tokenizer=util.LemmaTokenizer())
    vectors = KeyedVectors.load_word2vec_format(
            "data/GoogleNews-vectors-negative300.bin.gz", binary=True
            )
    vectors_normal = copy.deepcopy(vectors)
    vectors_normal.init_sims(replace=True)
    all_data = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, suffixes=('_x', '_y'), indicator=True)
    all_data2 = all_data.merge(
            all_data.apply(lambda row: pd.Series(util.getClosest(vectors,vectors_normal,vectorizer,row['Headline'],row['articleBody'])),axis=1),
        left_index=True, 
        right_index=True
    )
    all_data2.to_csv('closest.csv', index=False)
    stances = pd.concat([stances, all_data2], axis=1)
    mkStancCSV(stances)

maxWMD = stances[stances['WMD']!=np.inf]['WMD'].max()

if 'negated_words_cosine_similarity' not in stancecols:
    print('getting neg_words_similarity features')
    vectorizer = TfidfVectorizer(tokenizer=util.LemmaTokenizer())
    vectors = KeyedVectors.load_word2vec_format(
            "data/GoogleNews-vectors-negative300.bin.gz", binary=True
            )
    vectors.init_sims(replace=True)
    all_data = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, suffixes=('_x', '_y'), indicator=True)
    all_data_negated_sim = all_data.merge(
            all_data.apply(lambda row: pd.Series(util.getNegatedSim(vectors,vectorizer,row['negated_headline'],row['negated_body'])),axis=1),
        left_index=True, 
        right_index=True
    )
    all_data_negated_sim.to_csv('negated_similarity.csv', index=False)
    stances = pd.concat([stances, all_data_negated_sim[['negated_words_cosine_similarity','negated_words_WMD']]], axis=1)

    mkStanceCSV(stances)



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

#
class QuestionMarkAndNegData(BaseEstimator, TransformerMixin):
    """Extract all features from each document for DictVectorizer"""
    def countNeg(self, hlstring, bdstring):
        hlres = hlstring.count("n\'t")
        hltokens = hlstring.split()
        hltokens2 = [t for t in hltokens if (t == 'not' or t == 'cannot')]
        hlres += len(hltokens2)
        bdres = bdstring.count("n\'t")
        bdtokens = bdstring.split()
        bdtokens2 = [t for t in bdtokens if (t == 'not' or t == 'cannot')]
        bdres += len(bdtokens2)
#bool
        hlboolres = "n\'t" in hlstring
        hlbooltokens = hlstring.split()
        hlbooltokens2 = [t for t in hlbooltokens if (t == 'not' or t == 'cannot')]
        hlboolres = hlboolres or (len(hlbooltokens2) > 0)
        bdboolres = "n\'t" in bdstring
        bdbooltokens = bdstring.split()
        bdbooltokens2 = [t for t in bdbooltokens if (t == 'not' or t == 'cannot')]
        bdboolres = bdboolres or (len(bdbooltokens2) > 0)
        return {'hl_question_mark_count': hlstring.count("?"),
                    'hl_question_mark_bool': "?" in hlstring,
                    'hl_neg_count': hlres,
                    'hl_neg_bool': hlboolres,
                    'bd_question_mark_bool': "?" in bdstring,
                    'bd_neg_count': bdres,
                    'bd_neg_bool': bdboolres,
                    'abs_count_diff': hlres-bdres,
                    'abs_bool_diff': hlboolres-bdboolres
                }
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        #scaler = MinMaxScaler()
        res =  df.apply(lambda row: self.countNeg(row['Headline'],row['articleBody']), axis=1)
        #foo = scaler.fit_transform(np.array([x['hl_question_mark_count'] for x in res]).reshape(-1,1))
        return(res)

refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

discuss_words = [
        'allege', 'alleges', 'alleged', 'allegedly',
        'reportedly','reports','reported',
        'apparently', 'appears',
        'suggests',
        'seemingly', 'seems',
        'claims', 'claimed', 'claiming'#,
        #'accuses', 'accused', 'accusation'
    ]

class DiscussWordData(BaseEstimator, TransformerMixin):
    """Extract all features from each document for DictVectorizer"""
    def countNeg(self, hlstring, bdstring):
        hltokens = hlstring.split()
        hltokens2 = [t for t in hltokens if t in discuss_words]
        hlres = len(hltokens2)
        bdtokens = bdstring.split()
        bdtokens2 = [t for t in bdtokens if t in discuss_words]
        bdres = len(bdtokens2)
#bool
        hlbooltokens = hlstring.split()
        hlbooltokens2 = [t for t in hlbooltokens if t in discuss_words]
        hlboolres = len(hlbooltokens2) > 0
        bdbooltokens = bdstring.split()
        bdbooltokens2 = [t for t in bdbooltokens if (t == 'not' or t == 'cannot')]
        bdboolres = len(bdbooltokens2) > 0
        return {
                    'dhl_neg_count': hlres,
                    'dhl_neg_bool': hlboolres,
                    'dbd_neg_count': bdres,
                    'dbd_neg_bool': bdboolres,
                    'dabs_count_diff': hlres-bdres,
                    'dabs_bool_diff': hlboolres-bdboolres
                }
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        res =  df.apply(lambda row: self.countNeg(row['Headline'],row['articleBody']), axis=1)
        return(res)

class RefutingWordData(BaseEstimator, TransformerMixin):
    """Extract all features from each document for DictVectorizer"""
    def countNeg(self, hlstring, bdstring):
        hltokens = hlstring.split()
        hltokens2 = [t for t in hltokens if t in refuting_words]
        hlres = len(hltokens2)
        bdtokens = bdstring.split()
        bdtokens2 = [t for t in bdtokens if t in refuting_words]
        bdres = len(bdtokens2)
#bool
        hlbooltokens = hlstring.split()
        hlbooltokens2 = [t for t in hlbooltokens if t in refuting_words]
        hlboolres = len(hlbooltokens2) > 0
        bdbooltokens = bdstring.split()
        bdbooltokens2 = [t for t in bdbooltokens if (t == 'not' or t == 'cannot')]
        bdboolres = len(bdbooltokens2) > 0
        return {
                    'rhl_neg_count': hlres,
                    'rhl_neg_bool': hlboolres,
                    'rbd_neg_count': bdres,
                    'rbd_neg_bool': bdboolres,
                    'rabs_count_diff': hlres-bdres,
                    'rabs_bool_diff': hlboolres-bdboolres
                }
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        res =  df.apply(lambda row: self.countNeg(row['Headline'],row['articleBody']), axis=1)
        return(res)


## custom tokenizer to process text strings
class WMDData(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, wmds):
        return [{'wmd': wmd if wmd != np.inf else maxWMD}
                for wmd in wmds]

class CosineSimilarityData(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self
    def transform(self, cosines):
        return [{'cos': cosine}
                for cosine in cosines]

class ClosestCosData(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self
    def transform(self, closestcoss):
        return [{'closest_cos': closestcos}
                for closestcos in closestcoss]

class ClosestWMDData(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self
    def transform(self, closestWMDs):
        return [{'closest_wmd': closestWMD if closestWMD != np.inf else maxWMD}
                for closestWMD in closestWMDs]
class AnyData(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self
    def transform(self, columnValues):
        return [{'data': x}
                for x in columnValues]
#
## custom tokenizer to process text strings
#
def fnc_loss_func(ground_truth, predictions):
    related = ['agree', 'disagree', 'discuss']
    score = 0
#    print('len ground truth')
#    print(len(ground_truth))
#    print(ground_truth)
    for i in range(0, len(ground_truth)):
        print(i)
        gt = ground_truth[i]
        pred = predictions[i]
        if gt == 'unrelated':
            if pred == 'unrelated':
                score += 0.25
        else:
            if pred in related:
                score += 0.25
                if pred == gt:
                    score += 0.75
    num_unrelated = len([x for x in ground_truth if x == 'unrelated'])
    num_related = len(predictions) - num_unrelated
    maximum = num_unrelated * 0.25 + num_related * 1.00
    norm_score = score/maximum
    print('normalized score: ' + str(round(norm_score,2)))
    return(norm_score)

fnc_scorer = make_scorer(fnc_loss_func)

    #diff = np.abs(ground_truth - predictions).max()
    #return np.log(1 + diff)

tuned_parameters = [
    #{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
]

classifier = Pipeline([
    # Combining complaint text features, date features, zip_code features, and state
    ('union', FeatureUnion(
        transformer_list=[
            # Pipeline for standard bag-of-words TF-IDF stemmed model for body
            ('articleBodyBoW', Pipeline([
                ('selector', ItemSelector(key='articleBody')),
                ('tfidf', TfidfVectorizer(tokenizer=util.LemmaTokenizer())),
                #('best', TruncatedSVD(n_components=100)),
            ])),
            ('HeadlineBoW', Pipeline([
                ('selector', ItemSelector(key='Headline')),
                ('tfidf', TfidfVectorizer(tokenizer=util.LemmaTokenizer()))
                #('best', TruncatedSVD(n_components=100)),
            ])),
           # Featurizes dates according to DateData's transform method
            ('DiscussWordData', Pipeline([
                ('features', DiscussWordData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ('scaler', MaxAbsScaler())
            ])),
            ('RefutingWordData', Pipeline([
                ('features', RefutingWordData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ('scaler', MaxAbsScaler())
            ])),
            ('QuestionMarkAndNegData', Pipeline([
                ('features', QuestionMarkAndNegData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ('scaler', MaxAbsScaler())
            ])),
            ('wmdnormal', Pipeline([
                ('selector', ItemSelector(key='WMDnormal')),
                ('features', WMDData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                #('scaler', MaxAbsScaler())
            ])),
            #('wmd', Pipeline([
            #    ('selector', ItemSelector(key='WMD')),
            #    ('features', WMDData()),  # returns a list of dicts
            #    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            #    #('scaler', MaxAbsScaler())
            #])),
            ('closestCos', Pipeline([
                ('selector', ItemSelector(key='closest_cos')),
                ('features', ClosestCosData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                #('scaler', MaxAbsScaler())
            ])),
            ('negatedCos', Pipeline([
                ('selector', ItemSelector(key='negated_words_cosine_similarity')),
                ('features', AnyData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                #('scaler', MaxAbsScaler())
            ])),
            #('negatedWMD', Pipeline([
            #    ('selector', ItemSelector(key='negated_words_WMD')),
            #    ('features', ClosestWMDData()),  # returns a list of dicts
            #    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            #    #('scaler', MaxAbsScaler())
            #])),
            ('closestWMDnormal', Pipeline([
                ('selector', ItemSelector(key='closest_wmd_normal')),
                ('features', ClosestWMDData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                #('scaler', MaxAbsScaler())
            ])),
           # ('closestWMD', Pipeline([
           #     ('selector', ItemSelector(key='closest_wmd')),
           #     ('features', ClosestWMDData()),  # returns a list of dicts
           #     ('vect', DictVectorizer()),  # list of dicts -> feature matrix
           #     #('scaler', MaxAbsScaler())
           # ])),
            ('cosineSimilarity', Pipeline([
                ('selector', ItemSelector(key='cosine')),
                ('features', CosineSimilarityData()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                #('scaler', MaxAbsScaler())
            ]))
        ]
    )),
    # Use logistic regression
    #('classifier', LogisticRegression(C=1.0,class_weight='balanced'))
    ('classifier', GridSearchCV(LogisticRegression(C=1.0,class_weight='balanced'),param_grid=tuned_parameters,scoring=fnc_scorer))
#    ('classifier', LogisticRegression())
    #('classifier', LogisticRegression(C=0.1,class_weight={'agree':4,'disagree':4,'discuss':4,'unrelated':1}))
    #('classifier', SVC(C=1.0,class_weight='balanced'))
    #('classifier', SVC(C=1.0,class_weight={'agree':4,'disagree':4,'discuss':4,'unrelated':1}))
    #('classifier', SVC(C=0.1,class_weight={'agree':4,'disagree':4,'discuss':4,'unrelated':1}))
    #('classifier', MLPClassifier(alpha=0.05))
    #('classifier', MLPClassifier(alpha=0.03))
])

[training_and_dev_set_stance, test_set_stance] = util.splitIntoSets(stances, 0.15)
[training_set_stance, dev_set_stance] = util.splitIntoSets(training_and_dev_set_stance, 0.15) 

training_on = None
testing_on = None
#print("************ using SUBSET only")
test = 0
if 'test' in sys.argv:
    print('using real test set')
    training_on = training_and_dev_set_stance
    testing_on = test_set_stance
    test = 1
else:
    training_on = training_set_stance
    testing_on = dev_set_stance


training_set = pd.merge(training_on, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=True)
test_set = pd.merge(testing_on, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=True)

classifier.fit(training_set.drop('Stance', axis = 1), training_set['Stance'])
print(classifier.best_params_)
#print("getting predicted labels...")
#hypotheses = classifier.predict(test_set)
##hypotheses = ['unrelated'] * len(test_set)
#c_r = classification_report(test_set['Stance'], hypotheses)
#print(c_r)
#print(accuracy_score(test_set['Stance'], hypotheses))
#print(report_score(test_set['Stance'], hypotheses))
#
#splits= c_r.split('\n')[2:6]
#fscore_mean = np.mean([float(x.strip().split()[3]) for x in splits])
#print('fscore mean')
#print(fscore_mean)
#
#now = datetime.now()
#date_parts = [str(x) for x in [now.month,now.day,now.year]]
#time_parts = [str(x) for x in [now.hour,now.minute,now.second]]
#now_str = ''
#joblib.dump(classifier, 'classifiers/classifier' + now_str + '.pkl')
#now_str = "-".join(date_parts) + "_" + "-".join(time_parts)
