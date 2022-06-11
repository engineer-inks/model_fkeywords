ARG base_image
FROM $base_image

ARG config_path

ARG jfrog_user
ARG jfrog_pass

LABEL maintainer="ink@myrabr.com"
USER $root
#RUN apt-get upgrade && apt-get update

## install image of python 3 last version
RUN python -m pip install --root --upgrade pip
ADD requirements.txt .
RUN pip install -r requirements.txt
#RUN apt-get install openjdk-11-jdk
RUN pip install --upgrade setuptools
RUN pip install jupyterlab
RUN python -m spacy download pt_core_news_sm
RUN pip install nltk
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('punkt')"

ADD $config_path/spark-defaults.conf $SPARK_HOME/conf/

#ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
#ENV PYSPARK_SUBMIT_ARGS="--master local[3] pyspark-shell"

RUN echo 'c.NotebookApp.contents_manager_class = "notedown.NotedownContentsManager"' >> ~/.jupyter/jupyter_notebook_config.py