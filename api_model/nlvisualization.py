from locale import strcoll
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
from api_model.utils.logger import logger
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from kneebow.rotor import Rotor
from sklearn.cluster import MiniBatchKMeans
from api_model.utils.functions import TransforDatas


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

    @classmethod
    def pareto_plot(self,
        filename: str,
        x: str = None,
        y: str = None,
        title: str = None,
        limite: int = None,
        show_pct_y: bool = False,
        pct_format: str = '{0:.0%}',
    ):

        df = pd.read_csv(f"{PATH_READ}/{filename}_tratado.csv", sep=';', encoding='utf-8')
        
        logger.info('plot graph of pareto with categorized')
        df = df[x].value_counts().head(20)
        tmp = pd.DataFrame({x: df.index, 'count':df.values})
        
        xlabel = x
        ylabel = y
        x = tmp[x].values
        y = tmp[y].values
        weights = y / y.sum()
        cumsum = weights.cumsum()

        fig, ax1 = plt.subplots()
        ax1.bar(x, y)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.tick_params('x', rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(x, cumsum, '-ro', alpha=0.5)
        ax2.set_ylabel('', color='r')
        ax2.tick_params('y', colors='r')

        plt.rcParams['figure.figsize'] = (46, 9)
        vals = ax2.get_yticks()
        ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

        # hide y-labels on right side
        if not show_pct_y:
            ax2.set_yticks([])

        formatted_weights = [pct_format.format(x) for x in cumsum]
        for i, txt in enumerate(formatted_weights):
            ax2.annotate(txt, (x[i], cumsum[i]), fontweight='heavy')

        if title:
            plt.title(title)

        plt.show()

    @classmethod
    def model_2vectors(self,
        filename: str, content_column: str = None, vector_size: int = None, min_count: int = None, window: int = None
    ):
        cores = multiprocessing.cpu_count()  # Count the number of cores in a computer

        model = word2vec.Word2Vec(
            min_count=min_count,
            window=window,
            vector_size=vector_size,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            workers=cores - 1,
        )

        return model

    @classmethod
    def tsne_plot(self,
        filename:str,
        content_column: str,
        word: str,
        k: int = None,
        n_iter: int = None,
        vector_size: int = None,
        min_count: int = None,
        window: int = None,
    ):

        logger.info('plot graph of maps words')

        df = pd.read_csv(f"{PATH_READ}/{filename}_tratado.csv", sep=';', encoding='utf-8')
        # tokenizer column to model word2vec
        df.loc[:, f'{content_column}_len'] = df.apply(lambda row: nltk.word_tokenize(row[content_column]), axis=1)

        model = self.model_2vectors(df, None, vector_size=vector_size, min_count=min_count, window=window)

        t = time()
        model.build_vocab(df[f'{content_column}_len'], progress_per=10000)
        logger.debug(f'Time to build vocab: {format(round((time() - t) / 60, 2))}')

        list_names = [t[0] for t in model.wv.most_similar(positive=[word], topn=k)][10:]

        arrays = np.empty((0, vector_size), dtype='f')
        word_labels = [word]
        color_list = ['red']

        # adds the vector of the query word
        arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
        # gets list of most similar words
        close_words = model.wv.most_similar([word])

        # adds the vector for each of the closest words to the array
        for wrd_score in close_words:
            wrd_vector = model.wv.__getitem__([wrd_score[0]])
            word_labels.append(wrd_score[0])
            color_list.append('blue')
            arrays = np.append(arrays, wrd_vector, axis=0)

        # adds the vector for each of the words from list_names to the array
        for wrd in list_names:
            wrd_vector = model.wv.__getitem__([wrd])
            word_labels.append(wrd)
            color_list.append('green')
            arrays = np.append(arrays, wrd_vector, axis=0)

        # Reduces the dimensionality from 300 to 50 dimensions with PCA
        reduc = PCA(n_components=20, svd_solver='full').fit_transform(arrays)
        # Finds t-SNE coordinates for 2 dimensions
        np.set_printoptions(suppress=True)

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=n_iter).fit_transform(reduc)

        # Sets everything up to plot
        df = pd.DataFrame(
            {
                'x': [x for x in tsne_model[:, 0]],
                'y': [y for y in tsne_model[:, 1]],
                'words': word_labels,
                'color': color_list,
            }
        )

        fig, _ = plt.subplots()
        fig.set_size_inches(9, 9)

        # Basic plot
        p1 = sns.regplot(data=df, x='x', y='y', fit_reg=False, marker='o', scatter_kws={'s': 40, 'facecolors': df['color']})

        # Adds annotations one by one with a loop
        for line in range(0, df.shape[0]):
            p1.text(
                df['x'][line],
                df['y'][line],
                '  ' + df['words'][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom',
                size='medium',
                color=df['color'][line],
                weight='normal',
            ).set_size(15)

        plt.xlim(tsne_model[:, 0].min() - 50, tsne_model[:, 0].max() + 50)
        plt.ylim(tsne_model[:, 1].min() - 50, tsne_model[:, 1].max() + 50)

        plt.title('t-SNE visualization for {}'.format(word.title()))

    @classmethod
    def tsne_plot_all(self,
        filename:str,
        content_column: str,
        n_iter: int = None,
        vector_size: int = None,
        min_count: int = None,
        window: int = None,
    ):

        logger.info('plot graph of maps all of words')

        df = pd.read_csv(f"{PATH_READ}/{filename}_tratado.csv", sep=';', encoding='utf-8')
        # tokenizer column to model word2vec
        df.loc[:, f'{content_column}_len'] = df.apply(lambda row: nltk.word_tokenize(row[content_column]), axis=1)

        model = self.model_2vectors(df, f'{content_column}_len', vector_size=vector_size, min_count=min_count, window=window)

        labels = []
        tokens = []

        for word in model.wv.key_to_index:
            tokens.append(model.wv.key_to_index[word])
            labels.append(word)

        tsne_model = TSNE(perplexity=40, n_components=1, init='pca', n_iter=n_iter)
        tk = np.array(tokens).reshape(1, -1)
        new_values = tsne_model.fit_transform(tk)
        x = []
        y = []

        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(2, figsize=(38, 28 / 1.6180))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(18, 12), textcoords='offset points', ha='right', va='bottom')

        plt.show()

    @classmethod
    def get_freq_table(filename:str, coutent_column: str):

        logger.info('plot table with frequency of labels')

        df = pd.read_csv(f"{PATH_READ}/{filename}_tratado.csv", sep=';', encoding='utf-8')

        counts = df[coutent_column].value_counts()
        freq = pd.DataFrame(
            {
                coutent_column: counts.index,
                'Frequência Absoluta': counts.values,
                'Frequência Relativa': ['{:.2f}'.format(100 * v) for v in counts.values / counts.sum()],
            }
        )
        freq = freq.set_index(freq[coutent_column])
        freq.drop(coutent_column, 1, inplace=True)

        return freq

    @classmethod
    def plot_elbow(self, K: int = None, sum_of_squared_distances: float = None, optimal_k: int = None):

        plt.plot(range(1, K + 1), sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')

        plt.axvline(x=optimal_k)
        plt.show()


    @classmethod
    def plot_kmeans_clustering(self, resume_features: np.array = [], kmens_label: np.array = []):

        plt.figure('Cluster K-Means')
        plt.scatter(resume_features[:, 0], resume_features[:, 1], c=kmens_label)
        plt.xlabel('Dividend Yield')
        plt.ylabel('Returns')
        plt.title('Cluster K-Means')
        plt.show()


    @classmethod
    def clustering_model(self,
        filename: strcoll, content_column: str = None, model: str = 'kmeans', plot: bool = True, max_k: int = None, **kwargs
    ):

        df = pd.read_csv(f"{PATH_READ}/{filename}_tratado.csv", sep=';', encoding='utf-8')
        # TO-DO : Add Elbow Method
        logger.info('Start optmal clustering of kmeans model')
        if model == 'kmeans':

            #data = df.select(content_column).rdd.flatMap(lambda x: x).collect()
            data = df[content_column].values

            tfidf = TfidfVectorizer(use_idf=True)
            features = tfidf.fit_transform(data)

            logger.info('Find number of clusters')
            sum_of_squared_distances = []
            K = max_k
            for k in range(1, K + 1):
                km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, init_size=1000, batch_size=1000, **kwargs)
                km = km.fit(features)
                sum_of_squared_distances.append(km.inertia_)

            rotor = Rotor()

            data = [[-sum_of_squared_distances[x], x] for x in range(K)]
            rotor.fit_rotate(data)

            optimal_k = rotor.get_elbow_index() + 1

            if plot:
                self.plot_elbow(K, sum_of_squared_distances, optimal_k)

                logger.info(f'Optimal number of clusters: {optimal_k}')
                clst = MiniBatchKMeans(n_clusters=optimal_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000)

            logger.info('Using model of optmal clustering finded')
            nc = range(1, optimal_k)
            kmeans = [KMeans(n_clusters=i).fit(features) for i in nc]

            logger.info('Dimension of Reduction from compreendition vectorial space')
            c = len(kmeans) - 1
            resumed_features = LinearDiscriminantAnalysis(n_components=2).fit_transform(
                features.toarray(), kmeans[c].labels_
            )
            resumed_features.shape

            logger.info('plot scatter of clustering')
            self.plot_kmeans_clustering(resumed_features, kmeans[c].labels_)

        else:
            logger.info('Please Select An Available Clustering Method.')
            return []

        clst.fit(data)

        logger.info(f'Predict model in dataframe, {kmeans[c].labels_}')

        # classifier_udf = F.udf(
        #     lambda text: str(kmeans[c].predict(tfidf.transform([text]))[0]) if text else None, T.StringType()
        # )

        # df = df.withColumn('ml_kmeans', classifier_udf(content_column))
        # logger.info('Finishing process')

        df['ml_kmeans'] = kmeans[c].labels_

        df = TransforDatas.save_file(df, filename=filename, meth=None)
        logger.info('New DataFrame Saved with Kmeans leabels')

        return clst, optimal_k, df       