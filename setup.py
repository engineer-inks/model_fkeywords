from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools import setup, find_packages
from distutils.command.install import install as _install


import spacy
import subprocess
import sys

required_libs = ['api',
'numpy', 'nltk', 
'cython', 'gensim','pyspark', 
'scikit-learn', 'wordcloud',
 'spacy', 'kneebow', 'regex', 'seaborn']

setup(
   
    name='api',
    packages=find_packages(include=required_libs),
    version='0.1.0',
    description='A Natural Language Processing Library',
    author='Eneas Rodrigues',
    license='MIT',
    install_requires=required_libs,
    tests_require=['pytest==4.4'],
    #TO-DO: Fix dependency links : not working with bdist_wheel
    dependency_links = ["git+https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.2.0/pt_core_news_sm-3.2.0.tar.gz"],
    test_suite='tests',
    
)

