import regex
import re
import pickle
import spacy
import logging
import subprocess
import sys
import unicodedata
import nltk

from nltk import FreqDist
from nltk import stem
from nltk import download,data
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram, CountVectorizer, IDF
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.linalg import SparseVector
from pyspark.sql import DataFrame, Column, functions as F, types as T
from pyspark import StorageLevel


NLTK_STOPWORDS = nltk.corpus.stopwords.words('portuguese')

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
        text = re.split(r'\s+',text)
        return text


    @classmethod
    def filter_stop_words(self, text_tokens, additional_stop_words = []):
        stop_words = NLTK_STOPWORDS
        if len(additional_stop_words) > 0:
            for i in additional_stop_words:
                stop_words.append(i)
        return [word for word in text_tokens if not word in stop_words]

    
    @classmethod
    def convert_list_string(self,lista):
        out = []
        for i in lista:
            if len(i) > 2:
                out.append(i)
        return ' '.join(' '.join(out).strip().split())


    def lemmatizer(self, lista, parser='spacy'):
        nlp = NLExtractor._instance.spacy_nlp
        if parser == 'spacy':
            
            if self._spacy_load == None:
                nlp = self.load_spacy_modules()
            else:
                nlp = self._spacy_load

            if nlp == None:
                return []
        try:
            out = []
            for x in nlp(str(lista)):
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
            #return ' '.join(' '.join(out).strip().split())
            return list(dict.fromkeys(out))
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
                        output.append(self.detect_pattern(match.group().strip(), key, key, match.start(), match.end()))
                        break
        else:
            print('Invalid Type.')
        
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

        if texts_to_filter:
            x = x.withColumn(text_column, F.replace(text_column, texts_to_filter, F.replacement))            

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

        word_vocab = model.stages[4].vocabulary
        bigram_vocab = model.stages[5].vocabulary
        trigram_vocab = model.stages[6].vocabulary        

        word_field = f'{output_column_prefix}_word'
        bi_gram_field = f'{output_column_prefix}_bigram'
        tri_gram_field = f'{output_column_prefix}_trigram'

        def get_argmax_word(vocab, col):
            @F.udf()
            def get_argmax_word_(col):
                y = col.toArray()
                if texts_to_filter:
                    y = F.array([-1 if F.contains(vocab[idx], F.replacement) else elem for idx, elem in enumerate(y)])
                if all(elem <= min_df for elem in y):
                    return None
                return vocab[y.argmax().item()]
            return get_argmax_word_(col)

        x = (x.withColumn(word_field, get_argmax_word(vocab=word_vocab, col=words_idf.getOutputCol()))
                .withColumn(bi_gram_field, get_argmax_word(vocab=bigram_vocab, col=bigram_idf.getOutputCol()))
                .withColumn(tri_gram_field, get_argmax_word(vocab=trigram_vocab, col=trigram_idf.getOutputCol()))
                .drop(bigram.getOutputCol(), trigram.getOutputCol(), tokenizer.getOutputCol(),
                    words_remover.getOutputCol(), bigram_cv.getOutputCol(),
                    trigram_cv.getOutputCol(), words_cv.getOutputCol(), words_idf.getOutputCol(),
                    bigram_idf.getOutputCol(), trigram_idf.getOutputCol()))

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

