from api_model.nlextract import NLExtractor
import pandas as pd
from api_model.utils.functions import TransforDatas
from pyspark.sql import SparkSession, functions as F, types as T
from api_model.utils.logger import logger, Colorize


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
                activate_stopwords: str,
                interlocutor: dict,
                response_time: str,
                format_data: str,
                encoding: str
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
        self.interlocutor = interlocutor
        self.response_time = response_time
        self.format_data = format_data
        self.encoding = encoding

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
        activate_stopwords,
        interlocutor,
        response_time,
        format_data,
        encoding):
        
        """
        whats_process = 'complete'
            return: process all pipeline
        whats_process = 'partial'
            return: findkeywords and process bigrams
        whats_process = 'only_keywords'
            return: findkeywords       
        """

        logger.info('Load CSV')
        df = TransforDatas.load_file(prefix, filename, prefix_sep, encoding, meth=None)

        logger.info('Normalize Datas Values')
        df = TransforDatas.normalize_data(df=df, column_text=column_text, response_time=response_time, format_data=format_data, id_database=id_database)
      
        if whats_process == 'complete':
            logger.info(f'Start Complete Process')

            if activate_stopwords == 'sim':
                logger.info('Using StopWords')
                df = TransforDatas.stop_words_text(df=df, column_text=column_text, additional_stop_words=additional_stop_words)

            logger.info('Start Word Search')
            df = TransforDatas.word_search(df=df, list_pattern=list_pattern, type_find=type_find, column_text=column_text)                

            logger.info('Start Text Mining')
            df = TransforDatas.text_mining(df=df, column_text=column_text)

            
            logger.info('Send Some Statistics of DataFrame')
            df = TransforDatas.statistics_dataframe(df=df, column_text=column_text)

            logger.info('Called Pyspark DataFrame')
            sparkDF = TransforDatas.convert_dataframe(df=df, id_database=id_database, column_text=column_text, response_time=response_time, filename=filename, prefix=prefix, prefix_sep=prefix_sep, interlocutor=interlocutor, encoding=encoding)

            logger.info('remove null values of dataset')
            sparkDF = sparkDF.filter((F.col('message_content') != '')
                        & (F.col('message_content') != ' ')
                        & (F.col('message_content').isNotNull()))

            logger.debug(f'count rows after remove null values {sparkDF.count()}')   

            logger.debug('created a new collect dict of interlocutor')
            out = dict()
            for key, index in interlocutor.items():
                out = {'message_author':index}
            logger.debug(f'new collect dict {out}')

            logger.debug('created a original column')
            if 'original_message' not in df.columns:
                sparkDF = sparkDF.withColumn('original_message', F.col('message_content'))            
            
            logger.info('agroup all menssages for ticket')
            sparkDF = self.group_df(df=sparkDF, interlocutor=out, message_content='message_content')       

            logger.info(f'numbers of rows agrouped {sparkDF.count()}')

            output_prefix =  'countent'
            logger.info(f'Generating wordcloud columns for "{Colorize.get_color("all_messages", color="cyan")}" with prefix "{output_prefix}"')           
            sparkDF, _ = self.most_relevant_ngram(
                sparkDF, 'all_messages', id_field=id_database, output_column_prefix=output_prefix
            )

            logger.debug(f'columns process in most relevant ngrams {sparkDF.printSchema()}')
            df = sparkDF.toPandas()            

        if whats_process == 'partial':
            print(f'Start Partial Process')

            if activate_stopwords == 'sim':
                logger.info('Using StopWords')
                df = TransforDatas.stop_words_text(df=df, column_text=column_text, additional_stop_words=additional_stop_words)

            logger.info('Start Word Search')
            df = TransforDatas.word_search(df=df, list_pattern=list_pattern, type_find=type_find, column_text=column_text)
            
            logger.info('Send Some Statistics of DataFrame')
            df = TransforDatas.statistics_dataframe(df=df, column_text=column_text)

            logger.info('Called Pyspark DataFrame')
            sparkDF = TransforDatas.convert_dataframe(df=df, id_database=id_database, column_text=column_text, response_time=response_time, filename=filename, prefix=prefix, prefix_sep=prefix_sep, interlocutor=interlocutor)

            logger.info('remove null values of dataset')
            sparkDF = sparkDF.filter((F.col('message_content') != '')
                        & (F.col('message_content') != ' ')
                        & (F.col('message_content').isNotNull()))

            logger.debug(f'count rows after remove null values {sparkDF.count()}')   

            logger.debug('created a new collect dict of interlocutor')
            out = dict()
            for key, index in interlocutor.items():
                out = {'message_author':index}
            logger.debug(f'new collect dict {out}')

            logger.debug('created a original column')
            if 'original_message' not in df.columns:
                sparkDF = sparkDF.withColumn('original_message', F.col('message_content'))            
            
            logger.info('agroup all menssages for ticket')
            sparkDF = self.group_df(df=sparkDF, interlocutor=out, message_content='message_content')       

            logger.info('process bigrams and trigrams of column_text')
            output_prefix =  'countent'
            sparkDF, _ = self.most_relevant_ngram(
                sparkDF, 'all_messages', id_field=id_database, output_column_prefix=output_prefix
            )
            df = sparkDF.toPandas()
            # logger.info('Merge DataFrames')
            # df = TransforDatas.merge_dataframe(df=df_pandas, df_all=df_all)

        if whats_process == 'only_keywords':
            print(f'Start Only KeyWords Find Process')

            if activate_stopwords == 'sim':
                logger.info('Using StopWords')
                df = TransforDatas.stop_words_text(df=df, column_text=column_text, additional_stop_words=additional_stop_words)

            logger.info('Start Word Search')
            df = TransforDatas.word_search(df=df, list_pattern=list_pattern, type_find=type_find, column_text=column_text)
            
            logger.info('Send Some Statistics of DataFrame')
            df = TransforDatas.statistics_dataframe(df=df, column_text=column_text)

        logger.info('Finishing Process and Save csv File')
        df = TransforDatas.save_file(df, filename=filename, meth=None)
        logger.info(f'schema of dataframe saved: {df.info()}')

        TransforDatas.delete_file(filename=f'{filename}_temp')
        logger.debug('file temp deleted')        

        return df


__all__ = ['call_process']