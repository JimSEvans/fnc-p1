#df.merge(df.apply(lambda row: pd.Series({'c':row['a'] + row['b'], 'd':row['a']**2*row['b']}), axis = 1), left_index=True, right_index=True)
# 22563.250 was the top score
# This is a multi-class classification task. It is part 1 of the Fake News Challenge.
import util
import sys
import os.path
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from score import report_score
from datetime import datetime
from nltk.stem.porter import PorterStemmer
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

#print(" ************ REMOVE INDEXING FOR REAL RUN *************")
bodies = pd.read_csv(body_file)
stances = pd.read_csv(stance_file, index_col=None)
#bodies = pd.read_csv(body_file, index_col='Body ID')
#stances = pd.read_csv(stance_file, index_col=None)
all_data = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, suffixes=('_x', '_y'), copy=True, indicator=False)

bodycols = list(bodies)
stancecols = list(stances)

print('len bodies ' + str(len(bodies)))
print('len stances ' + str(len(stances)))

#stances.to_csv('train/stancesX.csv', index=False)

if 'bodyLang' not in bodycols:
    print('Using langdetect to detect language of body')
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    bodies['bodyLang'] = bodies.apply(lambda row: detect(row['articleBody']), axis=1)
    bodies.to_csv(w_body_file, index=False)
#
if 'tr_body' in sys.argv:
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
                    except: 
                        time.sleep(5)
                        print('waiting 5 seconds, then trying again...')
            else:
                pass
    bodies.to_csv(w_body_file, index=False)
#
if 'tr_headline' in sys.argv:
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
                except: 
                    time.sleep(5)
                    print('waiting 5 seconds, then trying again...')
    stances.to_csv(w_stance_file, index=False)
#
if 'cosine' not in stancecols:
    print('Getting cosine similarity between articles their headlines.')
    vectorizer = TfidfVectorizer(tokenizer=util.LemmaTokenizer())
    def getCosineSimilarity(text1, text2):
        tfidf = vectorizer.fit_transform([text1, text2])
        res = round(((tfidf * tfidf.T).A)[0,1], 5)
        #print(res)
        return res 

    stances['cosine'] = all_data.apply(lambda row: getCosineSimilarity(row['articleBody'], row['Headline']),axis=1)
    stances.to_csv(w_stance_file, index=False)



if 'WMD' not in stancecols:
    print('Computing word mover\'s distance.')
    vectors = KeyedVectors.load_word2vec_format(
            "data/GoogleNews-vectors-negative300.bin.gz", binary=True
            )
    stances['WMD'] = all_data.apply(
            lambda row: util.getWordMoversDistance(
                vectors, row['articleBody'], row['Headline']
                ), axis=1)
    stances.to_csv("train/train_stances.csv")

if 'negated_body' not in bodycols:
    print('Getting negated words with Stanford CoreNLP')
    from pycorenlp import StanfordCoreNLP
    # make sure you start the server in another tab/window first!
    my_nlp = StanfordCoreNLP('http://localhost:9000')
#    all_data.merge(
#            all_data.apply(lambda s: pd.Series({'feature1':s+1, 'feature2':s-1})), 
#left_index=True, right_index=True
#)
    stances['negated_body'] = all_data.apply(lambda row: util.getNegatedWords(row['articleBody'], my_nlp), axis=1)

    stances.to_csv("train/train_stances.csv")

#    stances['negatedWMD'] = all_data.apply(lambda row: util.getWordMoversDistance(vectors, row['articleBody'], row['Headline']), axis=1) 

if 'closest_sentence' not in stancecols:
    all_data.merge(
        all_data.apply(lambda r: pd.Series(util.getClosest(r['Headline'],r['articleBody']))),
        left_index=True, 
        right_index=True
    )

# get cosine similarity / WMD between each sentence and the headline
# choose the most similar sentence.
# save sentence AND cosine AND WMD



## This class is needed to break out the complaint column in the estimation pipeline that comes later
#class ItemSelector(BaseEstimator, TransformerMixin):
#    """For data grouped by feature, select subset of data at a provided key.
#    The data is expected to be stored in a 2D data structure, where the first
#    index is over features and the second is over samples. 
#    """
#    def __init__(self, key):
#        self.key = key
#    def fit(self, x, y=None):
#        return self
#    def transform(self, data_dict):
#        return data_dict[self.key]
#
#
#class QuestionMarkAndNegData(BaseEstimator, TransformerMixin):
#    """Extract all features from each document for DictVectorizer"""
#    def countNeg(self, hlstring, bdstring):
#        hlres = hlstring.count("n\'t")
#        hltokens = hlstring.split()
#        hltokens2 = [t for t in hltokens if (t == 'not' or t == 'cannot')]
#        hlres += len(hltokens2)
#        bdres = bdstring.count("n\'t")
#        bdtokens = bdstring.split()
#        bdtokens2 = [t for t in bdtokens if (t == 'not' or t == 'cannot')]
#        bdres += len(bdtokens2)
##bool
#        hlboolres = "n\'t" in hlstring
#        hlbooltokens = hlstring.split()
#        hlbooltokens2 = [t for t in hlbooltokens if (t == 'not' or t == 'cannot')]
#        hlboolres = hlboolres or (len(hlbooltokens2) > 0)
#        bdboolres = "n\'t" in bdstring
#        bdbooltokens = bdstring.split()
#        bdbooltokens2 = [t for t in bdbooltokens if (t == 'not' or t == 'cannot')]
#        bdboolres = bdboolres or (len(bdbooltokens2) > 0)
#        return {'hl_question_mark_count': hlstring.count("?"),
#                    'hl_question_mark_bool': "?" in hlstring,
#                    'hl_neg_count': hlres,
#                    'hl_neg_bool': hlboolres,
#                    'bd_question_mark_bool': "?" in bdstring,
#                    'bd_neg_count': bdres,
#                    'bd_neg_bool': bdboolres,
#                    'abs_count_diff': hlres-bdres,
#                    'abs_bool_diff': hlboolres-bdboolres
#                }
#    def fit(self, df, y=None):
#        return self
#
#    def transform(self, df):
#        #scaler = MinMaxScaler()
#        res =  df.apply(lambda row: self.countNeg(row['Headline'],row['articleBody']), axis=1)
#        #foo = scaler.fit_transform(np.array([x['hl_question_mark_count'] for x in res]).reshape(-1,1))
#        return(res)
#
#refuting_words = [
#        'fake',
#        'fraud',
#        'hoax',
#        'false',
#        'deny', 'denies',
#        # 'refute',
#        'not',
#        'despite',
#        'nope',
#        'doubt', 'doubts',
#        'bogus',
#        'debunk',
#        'pranks',
#        'retract'
#    ]
#
#discuss_words = [
#        'allege', 'alleges', 'alleged', 'allegedly',
#        'reportedly','reports','reported',
#        'apparently', 'appears',
#        'suggests',
#        'seemingly', 'seems',
#        'claims', 'claimed', 'claiming',
#        'accuses', 'accused', 'accusation'
#    ]
#
#class DiscussWordData(BaseEstimator, TransformerMixin):
#    """Extract all features from each document for DictVectorizer"""
#    def countNeg(self, hlstring, bdstring):
#        hltokens = hlstring.split()
#        hltokens2 = [t for t in hltokens if t in discuss_words]
#        hlres = len(hltokens2)
#        bdtokens = bdstring.split()
#        bdtokens2 = [t for t in bdtokens if t in discuss_words]
#        bdres = len(bdtokens2)
##bool
#        hlbooltokens = hlstring.split()
#        hlbooltokens2 = [t for t in hlbooltokens if t in discuss_words]
#        hlboolres = len(hlbooltokens2) > 0
#        bdbooltokens = bdstring.split()
#        bdbooltokens2 = [t for t in bdbooltokens if (t == 'not' or t == 'cannot')]
#        bdboolres = len(bdbooltokens2) > 0
#        return {
#                    'dhl_neg_count': hlres,
#                    'dhl_neg_bool': hlboolres,
#                    'dbd_neg_count': bdres,
#                    'dbd_neg_bool': bdboolres,
#                    'dabs_count_diff': hlres-bdres,
#                    'dabs_bool_diff': hlboolres-bdboolres
#                }
#    def fit(self, df, y=None):
#        return self
#
#    def transform(self, df):
#        res =  df.apply(lambda row: self.countNeg(row['Headline'],row['articleBody']), axis=1)
#        return(res)
#
#class RefutingWordData(BaseEstimator, TransformerMixin):
#    """Extract all features from each document for DictVectorizer"""
#    def countNeg(self, hlstring, bdstring):
#        hltokens = hlstring.split()
#        hltokens2 = [t for t in hltokens if t in refuting_words]
#        hlres = len(hltokens2)
#        bdtokens = bdstring.split()
#        bdtokens2 = [t for t in bdtokens if t in refuting_words]
#        bdres = len(bdtokens2)
##bool
#        hlbooltokens = hlstring.split()
#        hlbooltokens2 = [t for t in hlbooltokens if t in refuting_words]
#        hlboolres = len(hlbooltokens2) > 0
#        bdbooltokens = bdstring.split()
#        bdbooltokens2 = [t for t in bdbooltokens if (t == 'not' or t == 'cannot')]
#        bdboolres = len(bdbooltokens2) > 0
#        return {
#                    'rhl_neg_count': hlres,
#                    'rhl_neg_bool': hlboolres,
#                    'rbd_neg_count': bdres,
#                    'rbd_neg_bool': bdboolres,
#                    'rabs_count_diff': hlres-bdres,
#                    'rabs_bool_diff': hlboolres-bdboolres
#                }
#    def fit(self, df, y=None):
#        return self
#
#    def transform(self, df):
#        res =  df.apply(lambda row: self.countNeg(row['Headline'],row['articleBody']), axis=1)
#        return(res)
#
#
##    def boolNeg(self, hlstring):
##        hlres = "n\'t" in hlstring
##        hltokens = hlstring.split()
##        hltokens2 = [t for t in hltokens if (t == 'not' or t == 'cannot')]
##        hlres = hlres or (len(hltokens2) > 0)
##        bdres = "n\'t" in bdstring
##        bdtokens = bdstring.split()
##        bdtokens2 = [t for t in bdtokens if (t == 'not' or t == 'cannot')]
##        bdres = bdres or (len(bdtokens2) > 0)
##        return [hlres,bdres,abs(hlres-bdres)]
##    def transform(self, df):
##        hl = df['Headline']
##        bd = df['articleBody']
##        return [
##                {
##                    'question_mark_count': string.count("?"),
##                    'question_mark_bool': "?" in string,
##                    'neg_count': self.countNeg(string),
##                    'neg_bool': self.boolNeg(string)
##                    } for string in strings
##                ]
## custom tokenizer to process text strings
##class StemTokenizer(object):
##    def __init__(self):
##        self.stemmer = PorterStemmer()
##    def stem_tokens(self, tokens, stemmer):
##        stemmed = []
##        for item in tokens:
##            stemmed.append(stemmer.stem(item))
##        return stemmed
##    def tokenize(self, text):
##        lowercase_text = text.lower()
##        tokens = word_tokenize(lowercase_text)
##        filtered = [w for w in tokens if not w in stopwords.words('english')]
##        return filtered
##    def tokenize_and_stem(self, text):
##        tokens = self.tokenize(text)
##        if tokens is None:
##            print("\ntokens is none\n" + text)
##        stems = self.stem_tokens(tokens, self.stemmer)
##        return stems
##    def __call__(self, doc):
##        return self.tokenize_and_stem(doc)
#
## custom tokenizer to process text strings
#
#classifier = Pipeline([
#    # Combining complaint text features, date features, zip_code features, and state
#    ('union', FeatureUnion(
#        transformer_list=[
#            # Pipeline for standard bag-of-words TF-IDF stemmed model for body
##            ('articleBodyBoW', Pipeline([
##                ('selector', ItemSelector(key='articleBody')),
##                ('tfidf', TfidfVectorizer(tokenizer=LemmaTokenizer())),
##                # ('best', TruncatedSVD(n_components=150)),
##            ])),
##            ('HeadlineBoW', Pipeline([
##                ('selector', ItemSelector(key='Headline')),
##                ('tfidf', TfidfVectorizer(tokenizer=LemmaTokenizer())),
##                # ('best', TruncatedSVD(n_components=150)),
##            ])),
#            # Featurizes dates according to DateData's transform method
#            ('DiscussWordData', Pipeline([
#                ('features', DiscussWordData()),  # returns a list of dicts
#                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
#                ('scaler', MaxAbsScaler())
#            ])),
#            ('RefutingWordData', Pipeline([
#                ('features', RefutingWordData()),  # returns a list of dicts
#                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
#                ('scaler', MaxAbsScaler())
#            ])),
#            ('QuestionMarkAndNegData', Pipeline([
#                ('features', QuestionMarkAndNegData()),  # returns a list of dicts
#                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
#                ('scaler', MaxAbsScaler())
#            ])),
##            ('cosineSimilarity', Pipeline([
##                ('selector', ItemSelector(key='cosineSimilarity')),
##                ('features', CosineSimilarityData()),  # returns a list of dicts
##                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
##                ('scaler', MaxAbsScaler())
##            ]))
#            ('cosineSimilarity', Pipeline([
#                ('features', CosineSimilarityData()),  # returns a list of dicts
#                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
#                ('scaler', MaxAbsScaler())
#            ]))
#        ],
#    )),
#    # Use logistic regression
#    ('classifier', LogisticRegression(C=0.1,class_weight='balanced'))
#    #('classifier', SVC(C=1.0,class_weight='balanced'))
#])
## Rewrote the above pipleline since I was only using the "complaint" features in the final model
##classifier = Pipeline([
##    ('column_selector', ItemSelector(key='complaint')),
##    ('vectorizer', TfidfVectorizer(tokenizer=StemTokenizer())),
##    ('classifier', LogisticRegression(C=5.0))
##    ])
#
#[training_and_dev_set_stance, test_set_stance] = splitIntoSets(stances, 0.15)
#[training_set_stance, dev_set_stance] = splitIntoSets(training_and_dev_set_stance, 0.15) 
#
#training_on = training_set_stance
#testing_on = dev_set_stance
#if "real-test" in sys.argv:
#    training_on = training_and_dev_set_stance
#    testing_on = test_set_stance
#
##if "competition-test" in sys.argv:
#
#training_set = pd.merge(training_on, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
#test_set = pd.merge(testing_on, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
#
### just unrelated vs related ******************* !!!!!!!!!!!!!!!!!!!!!!!!!!!!
##training_set['Stance'] = training_set.apply(lambda row: 'unrelated' if row['Stance'] == 'unrelated' else 'related', axis=1)
##test_set['Stance'] = test_set.apply(lambda row: 'unrelated' if row['Stance'] == 'unrelated' else 'related', axis=1)
### ******************* !!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#classifier.fit(training_set.drop('Stance', axis = 1), training_set['Stance'])
#
## uncomment the following if testing on dev set
#print("getting predicted labels...")
#yhat = classifier.predict(test_set)
#print(classification_report(test_set['Stance'], yhat))
#print(accuracy_score(test_set['Stance'], yhat))
#
##def plot_confusion_matrix(cm, classes,
##                          normalize=False,
##                          title='Confusion matrix',
##                          cmap=plt.cm.Blues):
##    """
##    This function prints and plots the confusion matrix.
##    Normalization can be applied by setting `normalize=True`.
##    """
##    plt.imshow(cm, interpolation='nearest', cmap=cmap)
##    plt.title(title)
##    plt.colorbar()
##    tick_marks = np.arange(len(classes))
##    plt.xticks(tick_marks, classes, rotation=45)
##    plt.yticks(tick_marks, classes)
##    if normalize:
##        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
##        print("Normalized confusion matrix")
##    else:
##        print('Confusion matrix, without normalization')
##    print(cm)
##    thresh = cm.max() / 2.
##    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
##        plt.text(j, i, cm[i, j],
##                 horizontalalignment="center",
##                 color="white" if cm[i, j] > thresh else "black")
##    plt.tight_layout()
##    plt.ylabel('True label')
##    plt.xlabel('Predicted label')
### Compute confusion matrix
##cnf_matrix = confusion_matrix(test_set['Stance'], yhat)
##np.set_printoptions(precision=2)
### Plot non-normalized confusion matrix
##plt.figure()
##plot_confusion_matrix(cnf_matrix, classes=class_names,
##                      title='Confusion matrix, without normalization')
### Plot normalized confusion matrix
##plt.figure()
##plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
##                      title='Normalized confusion matrix')
##plt.show()
#print(report_score(test_set['Stance'], yhat))
#
#now = datetime.now()
#date_parts = [str(x) for x in [now.month,now.day,now.year]]
#time_parts = [str(x) for x in [now.hour,now.minute,now.second]]
#now_str = "-".join(date_parts) + "_" + "-".join(time_parts)
#
#joblib.dump(classifier, 'classifiers/classifier' + now_str + '.pkl')
