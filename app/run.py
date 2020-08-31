import json
import plotly
import pandas as pd
import joblib
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



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Implement the StartingVerbExtractor class
    
    """
    def starting_verb(self, text):
         # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    #Data graph 1, percentage of messages in each category and medium
    msgs = df[df.columns[3:]].groupby('genre').sum()
    
    #Data graph 2, count of number of categories each message belongs to
    conteo = df.drop(columns = ['id', 'message', 'original', 'genre']).sum(axis = 1, skipna = True)
    
    
    
    #For graph 3, percentage of messages in each category
    categories = df[df.columns[4:]]
    percentage_messag_cat = (categories.sum(axis = 0)*(100/26215)).sort_values(ascending=True)
    cat_names = list(percentage_messag_cat.keys())
   



    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data':[
              Bar(name = 'News',
              y = list(msgs.loc['news'].sort_values().index),
              x = msgs.loc['news'].sort_values(),
              orientation = 'h',
              ),
        
              Bar(name = 'Direct',
              y = list(msgs.loc['direct'].sort_values().index),
              x = msgs.loc['direct'].sort_values(),
              orientation = 'h',
              ),
        
              Bar(name = 'social',
              y = list(msgs.loc['social'].sort_values().index),
              x = msgs.loc['social'].sort_values(),
              orientation = 'h',
              ),
            ], 
                  
            'layout':{
                'title' : 'Count of messages in each category by medium (news, direct, social)',
                'yaxis_title' : 'Message categories',
                'width ' : 1000,
                'height' :  1600,
                'barmode' : 'group'
            }
        },
        
        {
            'data': [
                Bar(
                x = [str(x) for x in list(conteo.value_counts().index)],
                y = list(conteo.value_counts().values)
                )
            ],
            'layout' : {
                'title' : 'Frequency of number of categories of a message',
                'yaxis' : {'title' :'Frequency'},
                'xaxis' : {'title' : 'Total number of categories of a message',
                          'tickmode' : 'array',
                          'tickvals' : list(conteo.value_counts().index),
                          'ticktext' : [str(x) for x in list(conteo.value_counts().index)]}
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()