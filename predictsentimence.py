import pandas as pd
import nltk
import string
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('omw_1.4')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
def custom_encoder(df):
    df.replace(to_replace = [1, 2], value = 0, inplace = True)
    df.replace(to_replace = [3, 4, 5], value = 1, inplace = True)
    return df
def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree
def tokenization(text):
  words = nltk.word_tokenize(text)
  return words
def remove_stopwords(text):
  output = [i for i in text if i not in stopwords]
  return output
def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text

def preprocess(df_col):
    corpus = [ ]
    for item in df_col:
        new_item = remove_punctuation(item)
        new_item = new_item.lower()
        new_item = tokenization(new_item)
        new_item = remove_stopwords(new_item)
        new_item = lemmatizer(new_item)
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

def predictsentimence(text):
    print('text inside api:',text)
    df = pd.read_csv("D:\Project\WEB APP\data.csv", encoding='latin-1', names = ['text', 'label'])
    df.label = custom_encoder(df.label)
    corpus = preprocess(df.text)
    cv = CountVectorizer(ngram_range = (1,2))
    traindata = cv.fit_transform(corpus)
    X = traindata
    y = df.label
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=43)
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(x_train, y_train)
    text = [text]
    input = cv.transform(preprocess(text))
    prediction = clf.predict(input)
    if prediction == 0:
        score=0
    if prediction == 1:
        score=1
    return str(score)


# predictsentimence('it is good')