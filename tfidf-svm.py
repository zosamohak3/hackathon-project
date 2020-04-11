import pandas as pd
import string
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix , f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

def tfid_svm(df):
    vect = TfidfVectorizer(stop_words='english', analyzer='word',ngram_range=(1,2))
    tfidf_mad = vect.fit_transform(df.CONTENT)
    feature_names = vect.get_feature_names()
    dense = tfidf_mad.todense()
    denselist = dense.tolist()
    dfm = pd.DataFrame(denselist, columns=feature_names)
    dfm.head()
    x_train, x_test , y_train , y_test = train_test_split(dfm, df.CLASS, test_size = 0.3)
    clf = LinearSVC()
    clf.fit(x_train,y_train)
    return x_train, x_test , y_train , y_test, clf


