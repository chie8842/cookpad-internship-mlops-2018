#!/bin/bash

nohup jupyter notebook --port=8888 --ip=0.0.0.0 --no-browser --NotebookApp.token='test' &

