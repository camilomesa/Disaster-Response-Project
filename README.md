# Disaster Response Project
This project implements a random forest classifier model to label peoples messages by disaster category. The training data provided by [Figure Eight](https://appen.com/) was mined using ETL and NLP pipelines.


![alt text](https://www.weather.gov/images/safety/ia-2008-2-lg.jpg) 

### Table of contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgments](#licensing)


## Installation <a name="installation"></a>
This project was written using the Anaconda distribution of Python. The code should run with no issues using Python versions 3.* after installing the following:

**Pip Install:** json, plotly, pandas, joblib, flask, scikit-learn, scipy, matplotlib

from operator import add
import numpy as np


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

import sys
import pickle
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


## File Descriptions<a name="files"></a>

Here's the file structure of the project:
- app
  - template
   - master.html  # main page of web app
   - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
