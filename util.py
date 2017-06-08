import string
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.mystopwords = stopwords.words('english') + ['n\'t','wo']
    def stem_tokens(self, tokens, lemmatizer):
        lemmatized = []
        for item in tokens:
            lemmatized.append(lemmatizer.lemmatize(item))
        return lemmatized
    def tokenize(self, text):
        lowercase_text = text.lower()
        tokens = word_tokenize(lowercase_text)
        filtered = [w for w in tokens if (w not in self.mystopwords and w[0] not in list(string.punctuation))]
        return filtered
    def tokenize_and_lemmatize(self, text):
        tokens = self.tokenize(text)
        if tokens is None:
            print("\ntokens is none\n" + text)
        stems = self.stem_tokens(tokens, self.lemmatizer)
        return stems
    def __call__(self, doc):
        return self.tokenize_and_lemmatize(doc)


def getWordMoversDistance(vectors, text1string, text2string):
    text1 = text1string.lower().split()
    text2 = text2string.lower().split()
    mystopwords = stopwords.words('english')
    text1words = [w for w in text1 if w not in mystopwords]
    text2words = [w for w in text2 if w not in mystopwords]
    distance = vectors.wmdistance(text1words, text2words)
#    if distance = np.inf:
#        distance = 
    return distance

def getNegatedWords(text, nlp):
    output = nlp.annotate(
            text, properties={
                'annotators': 'depparse','outputFormat': 'json'
                }
            )
    try:
        sentence_data = output['sentences'][0]
        dep_parse = sentence_data['basicDependencies']
        negated = []
        for dikt in dep_parse:
            if dikt['dep']=='neg':
                negated.append(dikt['governorGloss'])
        return(" ".join(negated))
    except:
        return("")






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
#    
    train_ids = list(dg_train_set | ag_train_set | dc_train_set)
    test_ids = list(dg_test_set | ag_test_set | dc_test_set)
    test_ids_df = pd.DataFrame({'Body ID':test_ids})
    train_ids_df = pd.DataFrame({'Body ID':train_ids})
    train_stances_df = pd.merge(stances, train_ids_df, how='inner', left_on='Body ID', right_on='Body ID', sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
    test_stances_df = pd.merge(stances, test_ids_df, how='inner', left_on='Body ID', right_index=True, copy=True, indicator=False)
    return [train_stances_df, test_stances_df]
