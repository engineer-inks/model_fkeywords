import regex
import re
import pickle
import spacy
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

NLTK_STOPWORDS = nltk.corpus.stopwords.words('portuguese')

class NLExtractor:

    _instance = None
    _spacy_load = None    

    def __init__(self):
      self.data_type =[]

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                data.find('stemmers/rslp')
            except LookupError:
                download('rslp')
            try:
                data.find('stopwords')
            except LookupError:
                download('stopwords')
            
            cls._instance.spacy_nlp = cls._instance.spacy_nlp = spacy.load("pt_core_news_sm")       

        return cls._instance
    

    def cleaner(self, string, custom_cleaner=''):
        
        #TO-DO: Keep Acronyms

        string = string.lower()
        string = re.sub(r",", " , ", string)
        string = re.sub(r"\'", " \' ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\:", " : ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"\$", "  ", string)
        string = re.sub(r"\s{2,}", " ", string)

        string = string.strip()

        return string


    def remove_special_characters(self,s):
        try:
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) not in ['Mn', 'N'] and c.isalpha()
            ).lower()
        except IOError as e:
            print('Error tokenize', e)    


    def tokenizer(self, string):
        string = re.split(r'\s+',string)
        return string


    def filter_stop_words(self, lst, additional_stop_words = []):

        stop_words = NLTK_STOPWORDS
        stop_words.pop(stop_words.index('não'))
        if len(additional_stop_words) > 0:
            stop_words += additional_stop_words
        return [token for token in lst if token not in stop_words]


    def stemmer(self, lst, version='nltk'):

        pt_stem = stem.RSLPStemmer()
        
        return [pt_stem.stem(t) for t in lst]


    def pos_tagger(self, lst, POS_TAGGER = './data/POS_tagger_bigram.pkl'):
        #TO-DO: Bug on Pos-tagger load
        pos_tagger = pickle.load(open(POS_TAGGER, 'rb'))
        return pos_tagger.tag(lst)


    def lemmatizer(self, pos_lst, parser='spacy'):
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
            for x in nlp(str(pos_lst)):
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
        

    def histogram(self, lst, parser='spacy'):
        check_type = set([isinstance(el,list) for el in lst])

        if len(check_type) > 1:
            print('Please Double Check your Token List')
            return {}
        elif True in check_type:
            lst = sum(lst,[])
            
        return dict(FreqDist(lst))


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


    def udf_clean_text(self, text):
        out = []
        lem_ = re.sub(r'[^\w\s]','',text)
        for i in lem_:
            out.append(self.remove_special_characters(i.text))
        return out


    def udf_teste(self, message):
        return print(self.message)    
            
    #TO-DO: Datetime Converter

