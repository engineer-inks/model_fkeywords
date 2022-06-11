from api_model.nlextract import NLExtractor
import pandas as pd
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, Column, functions as F, types as T


class NlExtractorProcess(NLExtractor):
    extrator: NLExtractor = NotImplemented
    def __init__(self, 
                filename: str,
                prefix: str,
                prefix_sep: str,
                column_text: str,
                whats_process: str,
                list_pattern: dict,
                id_database: str):

        self.filename = filename
        self.prefix = prefix
        self.prefix_sep = prefix_sep
        self.column_text = column_text
        self.whats_process = whats_process
        self.list_pattern = list_pattern
        self.id_database = id_database

        super().__init__(self.call_process(
                            filename=filename,
                            prefix=prefix,
                            prefix_sep=prefix_sep,
                            column_text=column_text,
                            whats_process=whats_process,
                            list_pattern=list_pattern,
                            id_database=id_database
                            ))

    
    @classmethod
    def call_process(self,
        filename,
        prefix,
        prefix_sep,
        column_text,
        whats_process,
        list_pattern,
        id_database):
        
        """
        whats_process = 'complete'
            return: process all pipeline
        whats_process = 'partial'
            return: remove pontuation, findkeywords and process bigrams
        whats_process = 'only_keywords'
            return: findkeywords and process bigrams         
        """
        
        df_prefix = f'{prefix}'

        print(f'read file {df_prefix}')
        if df_prefix == 'xlsx':
            df = pd.read_excel(f"{filename}.{prefix}", engine='openpyxl')
            print(f'eschema of dataframe is {df.info()}')
        if df_prefix == 'CSV':
            df = pd.read_csv(f"{filename}.{prefix}", sep=prefix_sep, encoding='latin-1')
            print(f'eschema of dataframe is {df.info()}')

        print('convert column_text column to string type')
        df[column_text] = df[column_text].astype(str)

        print('put column_text in lower case')
        df[column_text] = df[column_text].str.lower()

        if whats_process == 'complete':

            print(f'remove special characters of column_text')
            df[column_text] =  df[column_text].apply(lambda x: self.udf_clean_text(x))

            print('collect words and find in column_text')
            print(f'dict: {list_pattern}')
            try:
                for key in list_pattern:
                    df[key] =  df[column_text].apply(lambda x: self.udf_type_keywords(x,list_pattern[key],mode="dictionary"))
            except IOError as e:
                print(f'não tem mais listas para rodar dados {e}')
            pass

            print('convert dataframe pandas to pyspark')
            df_part = df[[id_database,column_text]]

            spark = SparkSession.builder.getOrCreate()
            mySchema = StructType([ StructField(column_text, StringType(), True)\
                       ,StructField(f"{id_database}", IntegerType(), True)])
            sparkDF = spark.createDataFrame(df_part,schema=mySchema)

            print('process bigrams and trigrams of column_text')

            where = F.col(column_text).isNotNull()
            df_pandas, model = self.most_relevant_ngram(
                x=sparkDF, text_column=column_text, output_column_prefix='content', id_field=id_database, where=where
            )            

            print('convert to pandas again')
            df = df_pandas.toPandas()
            df_merge = pd.merge(df_pandas,df)            


        print('save csf file')
        filename_renomead = f'{filename}_tratado'
        file_save = f'{filename_renomead}.csv'
        df_merge.to_csv(f'/drive/My Drive/Colab Notebooks/{file_save}', sep=';',encoding='utf-8',index=False)
        
        return df


__all__ = ['call_process']