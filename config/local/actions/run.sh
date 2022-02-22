#!/bin/bash

python setup.py -q develop

jupyter notebook --ip=0.0.0.0 \
                 --port="$JUPYTER_PORT" --allow-root \
                 --NotebookApp.notebook_dir='./notebooks' \
                 --NotebookApp.token='' \
                 --NotebookApp.password=''
