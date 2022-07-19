import regex
import re
import pickle
import spacy
import logging
import subprocess
import sys
import unicodedata
import nltk

from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram, CountVectorizer, IDF
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.linalg import SparseVector
from pyspark.sql import Window, DataFrame, SparkSession, Column, functions as F, types as T
from pyspark import StorageLevel
from api_model.utils.logger import logger


NLTK_STOPWORDS = nltk.corpus.stopwords.words('portuguese')
NLP = spacy.load("pt_core_news_sm")

MAX_COLS = {
    'negativity'
}
AVG_COLS = {'response_time'}

class NLExtractor: 

    def __init__(self):
      self.data_type =[]


    @classmethod
    def cleaner(self, text):
        
        #TO-DO: Keep Acronyms

        text = text.lower()
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\'", " \' ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ) ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\:", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r"\$", "  ", text)
        text = re.sub(r"\-", "  ", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = text.strip()

        return text


    @classmethod
    def remove_special_characters(self,text):
        try:
            return ''.join(
                c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) not in ['Mn', 'N'] and c.isalpha()
            ).lower()
        except IOError as e:
            print('Error tokenize', e)


    @classmethod
    def udf_split_text(self, text):
    
        if type(text) == str:
            text = text.encode()
        else :
            text = text.astype(str).encode()
        for x in text.decode().replace('\n', ' ').split(' '):
            yield x


    @classmethod
    def udf_clean_text(self,text):
        try:
            text = self.cleaner(text)
            logging.info(f'Sucess separed pontuation {text}')   
            out = []
            for x in self.udf_split_text(text):
                this_text = self.remove_special_characters(x)
                if len(this_text) > 2:
                    out.append(this_text)
            logging.info('Sucess clean_text')                    
            return ' '.join(' '.join(out).strip().split())
        except IOError:
            logging.info('Error clean_text')


    @classmethod
    def clean_text(self,text):
        try:
            text = self.cleaner(text)
            logging.info(f'Sucess separed pontuation {text}')   
            out = []
            for x in self.udf_split_text(text):
                if len(x) > 2:
                    out.append(x)
            logging.info('Sucess clean_text')                    
            return ' '.join(' '.join(out).strip().split())
        except IOError:
            logging.info('Error clean_text')


    @classmethod
    def tokenizer(self, text):
        #text = re.split(r'\s+',text)
        text = word_tokenize(text)
        return text


    @classmethod
    def filter_stop_words(self, text_tokens, additional_stop_words = []):
        stop_words = NLTK_STOPWORDS
        if len(additional_stop_words) > 0:
            for i in additional_stop_words:
                stop_words.append(i)
        #return [word for word in text_tokens if not word in stop_words]
        return ' '.join([word for word in text_tokens.split() if word not in stop_words])


    @classmethod
    def new_stopwords(self, text_tokens, additional_stop_words = []):
        stop_words = NLTK_STOPWORDS
        if len(additional_stop_words) > 0:
            for i in additional_stop_words:
                stop_words.append(i)        
        pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
        text_tokens = pattern.sub('', text_tokens)
        return text_tokens

    
    @classmethod
    def convert_list_string(self,lista):
        out = []
        for i in lista:
            if len(i) > 2:
                out.append(i)
        return ' '.join(' '.join(out).strip().split())

    @classmethod
    def lemmatizer(self, lista, parser='spacy'):
        try:
            out = []
            lista = self.tokenizer(lista)
            for x in NLP(str(lista)):
                if ( not x.is_punct and
                    not x.is_space and
                    not x.is_stop and
                    len(x) > 1 and
                    self.remove_special_characters(x.text) != '' and 
                    self.remove_special_characters(x.text) != None ):
                    if x.pos_ == 'VERB':
                        out.append(self.remove_special_characters(x.lemma_))
                    else:
                        out.append(self.remove_special_characters(x.text))
            return ' '.join(' '.join(out).strip().split())
            #return list(dict.fromkeys(out))
        except IOError as e:
            print('Error text_lema', e)        
            pass


    def n_grams(self, lst, n_gram_size, filter_stop_words=False):
        n_grams_list = list(ngrams(lst,n_gram_size))
        return n_grams_list


    @classmethod
    def pattern_matcher(self, text, pattern, mode="dictionary", anonymizer = False, custom_anonymizer = '<UNK>'):

        output = []
        
        if mode == "regex":

            if isinstance(pattern, str):
                pattern = [pattern]

            for reg in pattern:
                if anonymizer:
                    text = re.sub(reg, custom_anonymizer, text)
                    output = text
                else:    
                    for match in re.finditer(reg, text):
                        output.append(self.detect_pattern(match.group().strip() , match.group(),match.group(), match.start(), match.end()))
        
        elif mode == "dictionary" or isinstance(pattern, dict):
            
            if isinstance(pattern, list):
                pattern = {p:[] for p in pattern}

            for key, synonims in pattern.items():

                pattern = r"(\b(" + "|".join([key] + synonims) + r")\b)"

                if anonymizer:
                    text = re.sub(pattern, custom_anonymizer, text)
                    output = text
                else:
                    if not re.search(r"\d", pattern):
                        pattern += r"{e<=1}"

                    for match in regex.finditer(pattern, text):
                        output.append(match.group().strip())
                        break
        else:
            print('Invalid Type.')
        
        output = self.convert_list_string(output)
        
        return output


    def ner(self,text, parser='spacy', ner_type = '', anonymizer = False, custom_anonymizer = '<UNK>'):
        
        output = []
        
        spacy_nlp = NLExtractor._instance.spacy_nlp
        
        if isinstance(ner_type, str):
            ner_type = [ner_type]
            
        if parser == 'spacy':

            if self._spacy_load == None:
                spacy_nlp = self.load_spacy_modules()
            else:
                spacy_nlp = self._spacy_load

            if spacy_nlp == None:
                return []

            ner = spacy_nlp(text)
            for entidade in ner.ents:   

                if anonymizer:
                    if len(ner_type) == 0:
                        print('WARNING: Please Specify An Entity Type.')
                        return text

                    for ner in ner_type:
                        if entidade.label_ == ner:
                            try:
                                text = re.sub(entidade.text, custom_anonymizer, text)
                            except:
                                text = text
                    output = text
                else:
                    output.append(
                        self.detect_pattern(entidade.text, entidade.text, entidade.label_, entidade.start_char, entidade.end_char)
                    )
        else:
            print('Invalid Parser.')

        return output


    @classmethod
    def detect_pattern(self, match, key, ner_type, start_index, end_index):
        # TODO: add score 
        return {
            "entity": match,
            "value": key,
            "type": ner_type, 
            "startIndex": start_index,
            "endIndex": end_index,
        }       


    def load_spacy_modules(self):
        try:
            self._spacy_load = spacy.load("pt_core_news_sm")
        except: 
            #TO-DO: Recognize and install spacy packages on runtime
            print('Warning!!!')
            print('Please Run the following command:')
            print('python -m spacy download pt_core_news_sm')
            print('before using spaCy modules')
            _spacy_load = None


    @classmethod
    def udf_type_keywords(self, text, pattern,  mode="dictionary"):
        """
        Search pattern in text and return what was found separated by pipe
        """

        if mode == "dictionary" or isinstance(pattern, dict):
            if isinstance(pattern, list):
                pattern = {p:[] for p in pattern}
                mentions = []
                for word in pattern:
                    regx = fr'\b{word}\b'

                    n_mentions = len(re.findall(regx, text))
                    if n_mentions:
                        mentions.append(word)
                return '|'.join(mentions)
        return None


    @classmethod
    def udf_extract_digits(self, text, pattern):
        document_terms = []
        
        cont_pattern = pattern
        for match in re.finditer(cont_pattern, str(text),re.IGNORECASE):
            document_terms.append(match.group().strip())

        document_terms = self.remove_duplicates(document_terms)
        document_terms = [re.sub('[^a-z0-9 ]+', '', str(document_terms))]
        return ''.join(','.join(document_terms).strip()).split()


    @classmethod
    def remove_duplicates(self, lista):
        l = []
        for i in lista:
            if i not in l:
                l.append(i)
        l.sort()
        return l


    def remove_specific_numbers(self, lista, numbers):
        return [token for token in lista if token not in numbers]

 
    @F.udf(T.StringType())
    def most_important_ngram(vocabulary, v):
        """
        Spark UDF that extracts the most relevant words, 2-grams and 3-grams with TF-IDF

        :param Vocabulary: Vocabulary of words by CountVectorizerModel
        :type vocabulary: str
        :param v: SparseVector with TF-IDF relevance values
        :type v: SparceVector
        :return: Dictionary with most relevant word, 2-gram anda 3-gram
        :rtype: dict
        """

        if isinstance(v, SparseVector):
            kv = {vocabulary[i]: cnt for (i, cnt) in zip(v.indices.tolist(), v.values.tolist())}
            return max(kv, key=kv.get, default='')
        return ''


    def spark(mode='default') -> SparkSession:
        """Retrieve current spark session.

        :param mode: str
            The session mode. Should be either "default" or "test".

        :return: SparkSession
        """
        logger.info(f'[INFO] Creating {mode} Spark Session')
        if mode == 'default':
            return SparkSession.builder.config("spark.driver.memory", "12g").getOrCreate()
        else:
            raise ValueError(f'Illegal value "{mode}" mode parameter. '
                            'It should be either "default", "test" or "deltalake".')        

    @classmethod
    def most_relevant_ngram(self,
        x: DataFrame,
        text_column: str,
        output_column_prefix: str,
        id_field: str = 'id',
        where: Column = None,
        stop_words='portuguese',
        features: int = 4096,
        min_df: float = 3.0,
        keep_intermediate: bool = False,
        texts_to_filter: str = (),
    ):
        """
        Computes and creates features to select the most relevant word, 2-grams and 3-grams.

        :param x: DataFrame Input containing pre-processed text
        :type x: DataFrame
        :param text_column: Name of the column containing the evaluating documents
        :type text_column: str
        :param output_column_prefix: Prefix used to create the output column.
        :type output_column_prefix: str
        :param id_field: Name of the column of the unique id to join the result on the original data frame in
                        case filter_expr was not null, defaults to 'id'
        :type id_field: str, optional
        :param where: Expression used to filter the dataframe if you want to extract the most relevant ngrams for a specific context, defaults to None
        :type where: column, optional
        :param stop_words: List of words fed to `StopWordsRemover` transformer, defaults to 'portuguese'
        :type stop_words: str, optional
        :param features: Max size of the vocabulary, defaults to 4096
        :type features: int, optional
        :param min_df: Specifies the minimum number of different documents a term must appear in to be included in the vocabulary
        :type min_df: float, optional
        :param keep_intermediate: Retain intermediate representation
        :type keep_intermediate: bool, optional
        :return: DataFrame containing most relevant word, 2-grams and 3-grams columns
        :rtype: DataFrame
        """

        o = x

        if isinstance(stop_words, str):
            stop_words = StopWordsRemover.loadDefaultStopWords('portuguese')

        if where is not None:
            x = o.filter(where)

        tokenizer = Tokenizer(inputCol=text_column, outputCol=f'{text_column}_aux_tokenized')

        words_remover = StopWordsRemover(
            inputCol=tokenizer.getOutputCol(), outputCol=f'{text_column}_aux_filtered', stopWords=stop_words
        )

        bigram = NGram(n=2, inputCol=words_remover.getOutputCol(), outputCol=f'{text_column}_aux_bigrams')
        trigram = NGram(n=3, inputCol=words_remover.getOutputCol(), outputCol=f'{text_column}_aux_trigrams')
        words_cv = CountVectorizer(
            inputCol=words_remover.getOutputCol(), outputCol=f'{text_column}_aux_words_cv', vocabSize=features, minDF=min_df
        )

        bigram_cv = CountVectorizer(
            inputCol=bigram.getOutputCol(), outputCol=f'{text_column}_aux_bigrams_cv', vocabSize=features, minDF=min_df
        )

        trigram_cv = CountVectorizer(
            inputCol=trigram.getOutputCol(), outputCol=f'{text_column}_aux_trigrams_cv', vocabSize=features, minDF=min_df
        )

        words_idf = IDF(inputCol=words_cv.getOutputCol(), outputCol=f'{output_column_prefix}_words_idf')
        bigram_idf = IDF(inputCol=bigram_cv.getOutputCol(), outputCol=f'{output_column_prefix}_bigrams_idf')
        trigram_idf = IDF(inputCol=trigram_cv.getOutputCol(), outputCol=f'{output_column_prefix}_trigrams_idf')

        pipeline = Pipeline(
            stages=[
                tokenizer,
                words_remover,
                bigram,
                trigram,
                words_cv,
                bigram_cv,
                trigram_cv,
                words_idf,
                bigram_idf,
                trigram_idf,
            ]
        )

        model = pipeline.fit(x)
        x = model.transform(x)

        word_field = f'{output_column_prefix}_word'
        bi_gram_field = f'{output_column_prefix}_bigram'
        tri_gram_field = f'{output_column_prefix}_trigram'

        for field, col, vocabulary in (
            (word_field, words_idf.getOutputCol(), model.stages[4].vocabulary),
            (bi_gram_field, bigram_idf.getOutputCol(), model.stages[5].vocabulary),
            (tri_gram_field, trigram_idf.getOutputCol(), model.stages[6].vocabulary),
        ):
            x = x.withColumn(field, self.most_important_ngram(F.array([F.lit(x) for x in vocabulary]), col))

        if not keep_intermediate:
            x = x.drop(
                bigram.getOutputCol(),
                trigram.getOutputCol(),
                tokenizer.getOutputCol(),
                words_remover.getOutputCol(),
                bigram_cv.getOutputCol(),
                trigram_cv.getOutputCol(),
                words_cv.getOutputCol(),
                words_idf.getOutputCol(),
                bigram_idf.getOutputCol(),
                trigram_idf.getOutputCol(),
            )

        x.persist(StorageLevel.DISK_ONLY)
        x.count()

        o.persist(StorageLevel.DISK_ONLY)
        o.count()

        if where is not None:
            id_r = f'filtered_{id_field}'
            x = x.select(id_field, bi_gram_field, tri_gram_field, word_field).withColumnRenamed(id_field, id_r)

            x = o.join(x, o[id_field] == x[id_r], how='left').drop(id_r)

        return x, model
    #TO-DO: Datetime Converter
    
    @classmethod
    def group_df(self, df, interlocutor, message_content):
        print('Grouping messages into one unique record')

        schema = []

        logger.debug(f'Message Author Column: message_author')
        logger.debug(f'Message content Column: {message_content}')

        bot_identifiers = interlocutor.get('bot_identifiers', [])
        client_identifiers = interlocutor.get('client_identifiers', [])

        logger.debug(f'{bot_identifiers}, {client_identifiers}')

        has_attendant = F.when(
            F.col('message_author').isin(bot_identifiers + client_identifiers), F.lit(0)
        ).otherwise(F.lit(1))

        df = df.withColumn('has_attendant', has_attendant)

        max_columns = [col for col in df.columns if col.endswith('_findint')]

        logger.debug(f'separadores cliente e operador, {interlocutor}')

        msgs_df = self._group_messages_df(df, bot_identifiers, interlocutor)
        kpis_df = self._group_kpis_df(df, max_columns, 'avg_columns')


        return kpis_df.join(msgs_df, on='issue_id', how='full')


    @classmethod
    def _group_messages_df(self, df, bot_identifiers, separators):
        all_message_columns = self._get_all_messages_columns(df, bot_identifiers, separators)

        return (
            df.orderBy('issue_id', 'message_order')
            .groupBy('issue_id')
            .agg(
                *all_message_columns,
                F.array_join(
                    F.collect_list(
                        F.concat(
                            F.col('message_time'),
                            F.lit(' - Author: '),
                            F.col('message_author'),
                            F.lit(' - Message: '),
                            F.col('original_message'),
                        )
                    ),
                    '\n---\n',
                ).alias('original_messages'),
            )
        )

    @classmethod
    def _group_kpis_df(self, df, additional_max_cols: list, additional_avg_cols: list):
        ignore_columns = ['issue_id', 'message_content', 'original_message']

        group_max_columns = MAX_COLS.union(additional_max_cols).intersection(df.columns)
        group_avg_columns = AVG_COLS.union(additional_avg_cols).intersection(df.columns)
        group_columns = set(df.columns).difference(group_max_columns.union(group_avg_columns).union(ignore_columns))

        logger.debug(f'MAX columns: {group_max_columns}')
        logger.debug(f'AVG columns: {group_avg_columns}')
        logger.debug(f'FIRST columns: {group_columns}')

        # !TODO: first is non-deterministic, could be getting something other than first record
        kpis_df = (
            df.orderBy('issue_id', 'message_order')
            .groupBy('issue_id')
            .agg(
                *(F.first(col).alias(col) for col in group_columns),
                *(F.max(col).alias(col) for col in group_max_columns),
                *(F.round(F.avg(col)).alias(f'average_{col}') for col in group_avg_columns),
            )
        )

        return kpis_df

    @classmethod
    def _get_all_messages_columns(self, df, bot_identifiers, separators):
        all_messages_col = self._join_messages(df, 'all_messages', bot_identifiers)

        columns = [all_messages_col]

        for col_separator, vals_separator in separators.items():
            for val_separator in vals_separator:
                first_value = val_separator.split('|')[0]
                alias = f'all_messages_{col_separator}_{first_value}'
                condition = F.col(col_separator).isin(val_separator.split('|'))

                current_all_messages_col = self._join_messages(df, alias, bot_identifiers, condition)

                columns.append(current_all_messages_col)

        return columns

    @classmethod
    def _join_messages(self, df, alias, bot_identifiers, condition=None):
        if condition is None:
            condition = F.lit(True)

        logger.debug(f'Joining messages to {alias}')      

        return F.array_join(
            F.collect_list(
                F.when(
                    (~F.col('message_author').isin(bot_identifiers)) & (condition),
                    F.col('message_content'),
                )
            ),
            ' ',
        ).alias(alias)


    @classmethod
    def populate_columns(self,df, columns):
        """
        Creates and fills the specified columns with values that are deducted from other columns that are present and correctly filled in the DataFrame

        :param df: Input DataFrame
        :type df: DataFrame
        :param columns: Columns that will be created and filled
        :type columns: list
        :return: DataFrame with the specified columns created and filled
        :rtype: DataFrame
        """

        if 'message_order' in columns:
            df = self.populate_message_order(df)
        return df

    @classmethod
    def populate_message_order(self, df, id_field = 'issue_id', message_time = 'message_time'):
        """
        Creates and fills the `message_order` column

        Based on the id of the conversation and the message time, the order of the messages is deducted and filled

        :param df: Input DataFrame
        :type df: DataFrame
        :param id_field: The column that has the conversation id, defaults to 'issue_id'
        :type id_field: str
        :param message_time: The column that contains the time of each message, defaults to 'message_time'
        :type message_time: str
        :return: DataFrame with the message_order filled
        :rtype: DataFrame
        """

        print('Generating `message_order` column')
        win = Window.partitionBy(id_field).orderBy(F.col(message_time).asc())
        df = df.withColumn('message_order', F.row_number().over(win))
        return df
