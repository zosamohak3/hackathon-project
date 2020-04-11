import pandas as pd
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

lemma = WordNetLemmatizer()
def lemmatize(txt):
    tokens = word_tokenize(txt.lower())
    tokens = [lemma.lemmatize(lemma.lemmatize(lemma.lemmatize(w,'v'),'n'),'a')for w in tokens]
    return ' '.join(tokens)

def pre(file,colomn,colomns):
    df=pd.read_csv(file)
    df=df[colomns].copy()
    df[colomn] = df.headline_text.apply(lambda txt: lemmatize(txt))
    return df
            
