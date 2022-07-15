from datetime import datetime
from xmlrpc.client import DateTime
import pandas as pd
from api_model.nlextract import NLExtractor
from pyspark.sql import functions as F, types as T
from api_model.utils.logger import logger
from datetime import date
import os
import re
from pytz import timezone


sao_paulo_timezone = timezone('America/Sao_Paulo')


PATH_READ = '/content/'
PATH_SAVE = '/content/drive/My Drive/'

#PATH_READ = '/opt/dna/find-keywords/datalake/'
#PATH_SAVE = '/opt/dna/find-keywords/datalake/'


class TransforDatas(NLExtractor):
    def __init__(self):
        self.data_type =[]
    
    @classmethod
    def convert_dataframe(self, df, id_database, column_text, response_time, filename, prefix, prefix_sep, interlocutor, encoding):

        column_name = ' '.join([str(item) for item in list(interlocutor.keys())])
        logger.info(f'rename valus of interlocutor column: {column_name}')

        logger.info('convert dataframe pandas to pyspark')
        df = df.rename(columns={id_database:'issue_id', column_text:'message_content', response_time:'message_time', column_name:'message_author'})

        logger.debug(f'save temp file')
        df = self.save_file(df=df, filename=filename, meth='temp')

        logger.debug(f'load temp file')
        dfPyspark = self.load_file(prefix=prefix, filename=filename, prefix_sep=prefix_sep, encoding=encoding, meth='temp')

        logger.info('created message order')
        dfPyspark = self.populate_message_order(dfPyspark, id_field='issue_id', message_time='message_time')    

        logger.debug(f'print new columns of pyspark dataframe: {dfPyspark.printSchema()}')

        return dfPyspark


    @classmethod
    def filter_columns(self, df):

        out = ['issue_id','message_content','message_time', 'message_author']
        list_1 = [col for col in df.columns if col in out]
        list_2 = [col for col in df.columns if col.endswith('_findint')]
        columns_to_extends = []

        for i in list_1:
            columns_to_extends.append(i)
        
        for y in list_2:
            columns_to_extends.append(y)

        logger.debug(f'get only columns to findkey words {columns_to_extends}')

        return columns_to_extends

    @classmethod
    def statistics_dataframe(self, df, column_text):

        logger.info('check numbers words by rows')
        df['numbers_words'] = df[column_text].apply(lambda x: len(str(x).split(' ')))

        df_mim = df[df['numbers_words']<=3]
        logger.info(f'numbers of rows < 3 words from line {df_mim["numbers_words"].count()}')

        df_max = df[df['numbers_words']>3]
        logger.info(f'numbers of promotors lines {df_max["numbers_words"].count()}')

        return df

    @classmethod
    def text_mining(self, df, column_text):

        logger.info(f'remove special characters and pontuation of column_text')
        df[column_text] =  df[column_text].apply(lambda x: self.udf_clean_text(x))

        logger.info('tranform text in text lemma')
        df[column_text] =  df[column_text].apply(lambda x: self.lemmatizer(x))

        return df

    @classmethod
    def stop_words_text(self, df, column_text, additional_stop_words):

        logger.info('remove stop words from text')
        logger.info(f'result before of process stop words \n {df[column_text].head(5)}')
        df[column_text] =  df[column_text].apply(lambda x: self.filter_stop_words(x, additional_stop_words))
        logger.info(f'result after of process stop words \n {df[column_text].head(5)}')

        return df

    @classmethod
    def word_search(self, df, list_pattern, type_find, column_text):

        logger.info('collect words and find in column_text')
        logger.debug(f'dict: {list_pattern}')
        try:
            for key in list_pattern:
                if type_find == 'fixo':
                    df[f'{key}_findint'] =  df[column_text].apply(lambda x: self.udf_type_keywords(x,list_pattern[key],mode="dictionary"))
                else:
                    df[f'{key}_findint'] =  df[column_text].apply(lambda x: self.pattern_matcher(x,list_pattern[key],mode="dictionary"))
        except IOError as e:
            logger.debug(f'não tem mais listas para rodar dados {e}')
        pass

        return df

    @classmethod
    def merge_dataframe(self, df, df_all):

        logger.info('convert to pandas again')
        df = df.toPandas()

        logger.debug(f'dataframe old {list(df_all.columns)}')
        logger.debug(f'dataframe new {list(df.columns)}')      
        
        df_merge = pd.concat([df_all, df], axis=1)
        logger.debug(f'joined dataframes {list(df_merge.columns)}  \n and numbers of rows is {df_merge["numbers_words"].count()}')

        return df_merge

    @classmethod
    def normalize_data(self, df, column_text, response_time, format_data, id_database):

        logger.info('convert column_text column to string type')
        df[column_text] = df[column_text].astype(str)    

        logger.info('put column_text in lower case')
        df[column_text] = df[column_text].str.lower()

        logger.info('normalize id column')
        df[id_database] = df[id_database].astype(str)   

        if response_time != '':
            logger.info('convert date column string in date column timestamp')
            df[response_time] = df[response_time].apply(lambda x: self.convert_datetime(x, format_data))
            logger.debug(f'dataformat: {df[response_time].head(5)}')
        else:
            logger.info('create date column timestamp')
            df[response_time] =pd.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(f'dataformat: {df[response_time].head(5)}')

        return df


    @classmethod
    def convert_datetime(self, dateformat, formats):
        
        if dateformat and isinstance(dateformat, str):
            for dtformat in formats.split('|'):
                try:
                    time = datetime.astimezone(sao_paulo_timezone).strftime(dateformat, dtformat)
                    return time
                except Exception:
                    continue
        elif isinstance(dateformat, datetime):
            return dateformat

        return None

    
    @classmethod
    def save_file(self, df, filename, meth):
        
        if meth == 'temp':

            logger.info('save csf file')
            filename_renomead = f'{filename}_temp'
            file_save = f'{filename_renomead}.csv'
            df.to_csv(f'{PATH_SAVE}/{file_save}', sep=';',encoding='utf-8',index=False)
            logger.info('Finishing Process')

        else:

            logger.info('save csf file')
            filename_renomead = f'{filename}_tratado'
            file_save = f'{filename_renomead}.csv'
            df.to_csv(f'{PATH_SAVE}/{file_save}', sep=';',encoding='utf-8',index=False)
            logger.info('Finishing Process') 

        return df

    
    @classmethod
    def load_file(self, prefix, filename, prefix_sep, encoding, meth):

        if meth == 'temp':

            logger.info('load csf file temp')
            filename_renomead = f'{filename}_temp'
            file_save = f'{filename_renomead}.csv'
            spark = self.spark('default')

            df = (spark.read.option('quote', '\"')
                    .option('escape', '\"')
                    .option('multiline', 'true')
                    .option('header', 'true')
                    .option('sep', ';')
                    .csv(f'{PATH_SAVE}/{file_save}'))

            logger.info('Finishing Process')

            return df        

        else:

            logger.info(f'read file {PATH_READ}{prefix}')
            if prefix == 'xlsx':
                df = pd.read_excel(f"{PATH_READ}/{filename}.{prefix}", engine='openpyxl')
                logger.debug(f'Schema of dataframe is {df.info()}')
            if prefix == 'csv':
                df = pd.read_csv(f"{PATH_READ}/{filename}.{prefix}", sep=prefix_sep, encoding=encoding)
                logger.debug(f'Schema of dataframe is {df.info()}')

        return df

    
    @classmethod
    def delete_file(self, filename):

        filepath = f'{PATH_SAVE}/{filename}.csv'

        if os.path.exists(filepath):
            logger.info(f'deleting file: {filepath}')
            os.remove(filepath)
        
        else:
            logger.info("Can not delete the file as it doesn't exists")

        return None
     
