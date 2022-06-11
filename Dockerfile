ARG base_image
FROM $base_image

ARG config_path

LABEL maintainer="ink@myrabr.com"

## install image of python 3 last version
RUN python -m pip install --root --upgrade pip
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --upgrade setuptools
RUN pip install jupyterlab
RUN python -m spacy download pt_core_news_sm
RUN pip install nltk
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('punkt')"

ADD $config_path/spark-defaults.conf $SPARK_HOME/conf/
