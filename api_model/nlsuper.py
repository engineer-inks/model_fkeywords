from api_model.nlextract import NLExtractor
import pandas as pd
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, Column, functions as F, types as T


class NlExtractorProcess(NLExtractor):

    def __init__(self, 
                filename: str,
                prefix: str,
                prefix_sep: str,
                column_text: str,
                whats_process: str,
                list_pattern: dict,
                id_database: str,
                type_find: str,
                additional_stop_words: list,
                activate_stopwords: str
                ):
            
        self.filename = filename
        self.prefix = prefix
        self.prefix_sep = prefix_sep
        self.column_text = column_text
        self.whats_process = whats_process
        self.list_pattern = list_pattern
        self.id_database = id_database
        self.type_find = type_find
        self.additional_stop_words = additional_stop_words
        self.activate_stopwords = activate_stopwords

        super().__init__()
    
    @classmethod
    def call_process(self,
        filename,
        prefix,
        prefix_sep,
        column_text,
        whats_process,
        list_pattern,
        id_database,
        type_find,
        additional_stop_words,
        activate_stopwords):
        
        """
        whats_process = 'complete'
            return: process all pipeline
        whats_process = 'partial'
            return: findkeywords and process bigrams
        whats_process = 'only_keywords'
            return: findkeywords       
        """
        
        df_prefix = f'{prefix}'
        path_read = '/content/'
        path_save = '/content/drive/My Drive/'

        #path_read = '/opt/dna/find-keywords/datalake/'
        #path_save = '/opt/dna/find-keywords/datalake/'

        print(f'read file {path_read}{df_prefix}')
        if df_prefix == 'xlsx':
            df = pd.read_excel(f"{path_read}/{filename}.{prefix}", engine='openpyxl')
            print(f'eschema of dataframe is {df.info()}')
        if df_prefix == 'CSV':
            df = pd.read_csv(f"{path_read}/{filename}.{prefix}", sep=prefix_sep, encoding='latin-1')
            print(f'eschema of dataframe is {df.info()}')

        print('convert column_text column to string type')
        df[column_text] = df[column_text].astype(str)

        print('put column_text in lower case')
        df[column_text] = df[column_text].str.lower()
      
        if whats_process == 'complete':
            print(f'Start Complete Process')

            if activate_stopwords == 'sim':
                print('remove stop words from text')
                df[column_text] =  df[column_text].apply(lambda x: self.filter_stop_words(x, additional_stop_words))
                df[column_text] =  df[column_text].apply(lambda x: self.convert_list_string(x))

            print(f'remove special characters and pontuation of column_text')
            df[column_text] =  df[column_text].apply(lambda x: self.udf_clean_text(x))

            print('collect words and find in column_text')
            print(f'dict: {list_pattern}')
            try:
                for key in list_pattern:
                    if type_find == 'fixo':
                        df[key] =  df[column_text].apply(lambda x: self.udf_type_keywords(x,list_pattern[key],mode="dictionary"))
                    else:
                        df[key] =  df[column_text].apply(lambda x: self.pattern_matcher(x,list_pattern[key],mode="dictionary"))
            except IOError as e:
                print(f'não tem mais listas para rodar dados {e}')
            pass

            print('check numbers words by rows')
            df['numbers_words'] = df[column_text].apply(lambda x: len(str(x).split(' ')))

            df_mim = df[df['numbers_words']<=3]
            print(f'numbers of rows < 3 words from line {df_mim["numbers_words"].count()}')

            df_max = df[df['numbers_words']>3]
            print(f'numbers of promotors lines {df_max["numbers_words"].count()}')

            df_all = df

            print('convert dataframe pandas to pyspark')
            df_part = df_all[[id_database,column_text]]

            spark = self.spark('default')
            mySchema = StructType([ StructField(id_database, StringType(), True)\
                       ,StructField(column_text, StringType(), True)])

            sparkDF = spark.createDataFrame(df_part,schema=mySchema)
            print(f'{sparkDF.printSchema()}')

            print('remove null values of dataset')
            sparkDF = sparkDF.filter(
                            (F.col(column_text) != '')
                            & (F.col(column_text) != ' ')
                            & (F.col(column_text).isNotNull()
                            ))

            print(f'count rows after remove null values {sparkDF.count()}')
            print(f'{sparkDF.show(5,truncate=False)}')          

            print('process bigrams and trigrams of column_text')
            output_prefix =  'countent'
            df_pandas, model = self.most_relevant_ngram(
                sparkDF, column_text, id_field=id_database, output_column_prefix=output_prefix
            )
            print(f'dataframe with wordCloud ..{df_pandas.show(5,truncate=False)}')

            print('convert to pandas again')
            df = df_pandas.toPandas()

            print(f'dataframe old {list(df_all.columns)}')
            print(f'dataframe new {list(df.columns)}')

            df_all[id_database] = df_all[id_database].astype(str)          
            
            df_merge = pd.concat([df_all, df], axis=1)
            print(f'joined dataframes {list(df_merge.columns)}  \n and numbers of rows is {df_merge["numbers_words"].count()}')

        if whats_process == 'partial':
            print(f'Start Partial Process')

            print('collect words and find in column_text')
            print(f'dict: {list_pattern}')
            try:
                for key in list_pattern:
                    if type_find == 'fixo':
                        df[key] =  df[column_text].apply(lambda x: self.udf_type_keywords(x,list_pattern[key],mode="dictionary"))
                    else:
                        df[key] =  df[column_text].apply(lambda x: self.pattern_matcher(x,list_pattern[key],mode="dictionary"))
            except IOError as e:
                print(f'não tem mais listas para rodar dados {e}')
            pass

            print('check numbers words by rows')
            df['numbers_words'] = df[column_text].apply(lambda x: len(str(x).split(' ')))

            df_mim = df[df['numbers_words']<=3]
            print(f'numbers of rows < 3 words from line {df_mim["numbers_words"].count()}')

            df_max = df[df['numbers_words']>3]
            print(f'numbers of promotors lines {df_max["numbers_words"].count()}')

            df_all = df

            print('convert dataframe pandas to pyspark')
            df_part = df_all[[id_database,column_text]]

            spark = self.spark('default')
            mySchema = StructType([ StructField(id_database, StringType(), True)\
                       ,StructField(column_text, StringType(), True)])

            sparkDF = spark.createDataFrame(df_part,schema=mySchema)
            print(f'{sparkDF.printSchema()}')

            print('remove null values of dataset')
            sparkDF = sparkDF.filter(
                            (F.col(column_text) != '')
                            & (F.col(column_text) != ' ')
                            & (F.col(column_text).isNotNull()
                            ))

            print(f'count rows after remove null values {sparkDF.count()}')
            print(f'{sparkDF.show(5,truncate=False)}')          

            print('process bigrams and trigrams of column_text')
            output_prefix =  'countent'
            df_pandas, model = self.most_relevant_ngram(
                sparkDF, column_text, id_field=id_database, output_column_prefix=output_prefix
            )
            print(f'dataframe with wordCloud ..{df_pandas.show(5,truncate=False)}')

            print('convert to pandas again')
            df = df_pandas.toPandas()

            print(f'dataframe old {list(df_all.columns)}')
            print(f'dataframe new {list(df.columns)}')
            
            df_all[id_database] = df_all[id_database].astype(str)          
            
            df_merge = pd.concat([df_all, df], axis=1)
            print(f'joined dataframes {list(df_merge.columns)}  \n and numbers of rows is {df_merge["numbers_words"].count()}')

        if whats_process == 'only_keywords':
            print(f'Start Only KeyWords Find Process')

            print('collect words and find in column_text')
            print(f'dict: {list_pattern}')
            try:
                for key in list_pattern:
                    if type_find == 'fixo':
                        df[key] =  df[column_text].apply(lambda x: self.udf_type_keywords(x,list_pattern[key],mode="dictionary"))
                    else:
                        df[key] =  df[column_text].apply(lambda x: self.pattern_matcher(x,list_pattern[key],mode="dictionary"))
            except IOError as e:
                print(f'não tem mais listas para rodar dados {e}')
            pass
            df_merge = df                           

        print('save csf file')
        filename_renomead = f'{filename}_tratado'
        file_save = f'{filename_renomead}.csv'
        df_merge.to_csv(f'{path_save}/{file_save}', sep=';',encoding='utf-8',index=False)
        print('Finishing Process')
        
        return df_merge


    def spark(mode='default') -> SparkSession:
        """Retrieve current spark session.

        :param mode: str
            The session mode. Should be either "default" or "test".

        :return: SparkSession
        """
        print(f'[INFO] Creating {mode} Spark Session')
        if mode == 'default':
            return SparkSession.builder.config("spark.driver.memory", "12g").getOrCreate()
        else:
            raise ValueError(f'Illegal value "{mode}" mode parameter. '
                            'It should be either "default", "test" or "deltalake".')

__all__ = ['call_process']