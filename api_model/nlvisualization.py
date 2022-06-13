from logging import log
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
import seaborn as sns
from pyspark.sql import DataFrame, dataframe, functions as F, types as T
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import word2vec
import multiprocessing
from time import time
import nltk

nltk.download('punkt')

NLTK_STOPWORDS = nltk.corpus.stopwords.words('portuguese')

PATH_READ = '/content/drive/My Drive/'
#PATH_READ = '/opt/dna/find-keywords/datalake'

class NlVisualization:


    def __init__(self, 
                filename: str,
                column_filter: str,
                column_text: str,
                whats_process: str):
        
        self.filename = filename
        self.column_filter = column_filter
        self.column_text = column_text
        self.whats_process = whats_process

    @classmethod
    def wordCloud_Topics(self,
                filename: str,
                column_filter: str,              
                column_text: str,
                whats_process: str):
    

        df = pd.read_csv(f"{PATH_READ}/{filename}_tratado.csv", sep=';', encoding='utf-8')

        if column_filter == '':
            pass
        else:
            df = df[df[column_filter].notnull()]
            df = df[df[column_filter].notna()]
            df = df[df[column_filter] != '']
            df = df[df[column_filter] !='[]']

        list_str = df[column_text].values.tolist()
        sent = ','.join([str(i) for i in list_str])

        tokenizer = RegexpTokenizer(r'\w+')
        sent_words = tokenizer.tokenize(sent)

        if whats_process == 'bigram':
            brigram = bigrams(sent_words)
            join_bigram = [' '.join((a, b)) for a, b in brigram]
        else:
            brigram = trigrams(sent_words)
            join_bigram = [' '.join((a, b, c)) for a, b, c in brigram]

        freq_dist = FreqDist(join_bigram)

        wc = WordCloud(
            min_font_size=14,
            max_font_size=170,
            max_words=50,
            background_color='white',
            colormap='winter',
            width=700,
            height=550,
            stopwords=NLTK_STOPWORDS,
        )

        wordcloud_bigrama = wc.generate_from_frequencies(freq_dist)
        plt.figure(2, figsize=(19, 19 / 1.6180))
        plt.imshow(wordcloud_bigrama, interpolation='bilinear')
        plt.tight_layout(pad=0)
        plt.axis('off')

        plt.show()        


        return None
    
    @classmethod
    def plot_10_most_common_words(self,
                filename: str,
                column_filter: str,              
                column_text: str):
        
        df = pd.read_csv(f"{PATH_READ}/{filename}_tratado.csv", sep=';', encoding='utf-8')

        if column_filter == '':
            pass
            df = df[~df[column_text].isnull()]
        else:                
            df = df[df[column_filter].notnull()]
            df = df[df[column_filter].notna()]
            df = df[df[column_filter] != '']

        number_cluster = 20
        list_str = df[column_text].values.tolist()

        vectorizer = CountVectorizer(stop_words=NLTK_STOPWORDS)
        count_vect = vectorizer.fit_transform(list_str)

        words = vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))

        for t in count_vect:
            total_counts += t.toarray()[0]

        count_dict = zip(words, total_counts)
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:number_cluster]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        plt.figure(2, figsize=(19, 19 / 1.6180))
        plt.subplot(title=str(number_cluster) + ' most common words')

        sns.set_context('notebook', font_scale=1.75, rc={'lines.linewidth': 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel('words')
        plt.ylabel('counts')

        plt.show()

        return None 