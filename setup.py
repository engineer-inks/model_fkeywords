from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools import setup, find_packages
from distutils.command.install import install as _install

import os

PROJECT_DIR = os.path.dirname(__file__)
DEPENDENCIES = open(os.path.join(PROJECT_DIR, 'requirements.txt')).readlines()

setup(
   
    name='api',
    version='0.1.0',
    description='A Natural Language Processing Library',
    author='Eneas Rodrigues',
    license='MIT',
    packages=find_packages(include=['api.*']),
    install_requires=[d for d in DEPENDENCIES if '://' not in d],
    python_requires='>=3.7',
    #TO-DO: Fix dependency links : not working with bdist_wheel
    dependency_links = ["git+https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.2.0/pt_core_news_sm-3.2.0.tar.gz"],
    tests_require=['pytest', 'parameterized'],
    zip_safe=False
    
)

