import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def lda(csv_file,colomn,n_of_topics,n_of_words):
    reviews_datasets = csv_file
    reviews_datasets = reviews_datasets.head(20000)
    reviews_datasets.dropna()
    reviews_datasets.head()
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
    doc_term_matrix = count_vect.fit_transform(reviews_datasets[colomn].values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=n_of_topics, random_state=42)
    LDA.fit(doc_term_matrix)
    first_topic = LDA.components_[0]    
    top_topic_words = first_topic.argsort()[-10:]
    for i,topic in enumerate(LDA.components_):
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-n_of_words:]])


