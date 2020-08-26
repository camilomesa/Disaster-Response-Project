#!/usr/bin/env python
# coding: utf-8

# In[18]:


# import packages
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download(['punkt', 'wordnet'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


# In[19]:


def load_data(database_file):
    """
    Load dataset from database with read_sql_table and defines features and target variables
    Args: database_file to be entered as string, or .db extension
    Returns: X (features array), Y (target variables dataframe)
    """
    # read in file
    engine = create_engine('sqlite:///'+database_file)
    df = pd.read_sql_table('Messages', engine)
    # define features and label arrays
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y


# In[20]:


X , Y = load_data('DisasterResponse.db')


# In[21]:


def tokenize(text):
    """
    Tokenizes, lemmatizes and cleans messages
    Args: text (messages)
    Returns: clean_tokens (list of tokens)
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# In[24]:


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


# In[25]:


def build_model():
    """Builds multioutput classifier with a RandomForestClassifier estimator pipeline and implements GridSearchCV to select the best model using crossvalidation of parameters.
    Args: none
    Returns: model_pipilene(best model obtained from GridSearchCV)
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    
    # define parameters for GridSearchCV
    parameters = {
        'features__text_pipeline__vect__max_df': [0.5],
        'clf__estimator__n_estimators': [10],
        'features__text_pipeline__vect__ngram_range': [(1, 1)]
    }
    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_macro', cv=None, n_jobs=-1) 
    #print("\nBest Parameters:", model_pipeline.best_params_)
    return model_pipeline


# In[26]:


def train(X, Y, model):
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    
    # fit model
    model = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    #format Y_pred to match the format of Y_test
    index = [str(x) for x in list(Y_test.index)]
    category_names = list(Y.columns)
    Y_pred = pd.DataFrame(Y_pred, index = index , columns = category_names )
    
    # output model test results
    
    print(classification_report(Y_test, Y_pred, target_names= category_names))
    print("\nBest Parameters:", model.best_params_)


    return model



# In[ ]:


#def evaluate_model(model, X_test, Y_test, category_names):
    
 #   pipeline.fit(X_train,Y_train)
  #  print(classification_report(Y_test, Y_pred, target_names= category_names))


# In[27]:


def export_model(model):
    # Export model as a pickle file
    pickle.dump(model, open('train_classifier.pkl','wb'))


# In[28]:


def run_pipeline(data_file):
    X, Y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, Y, model)  # train model pipeline
    export_model(model)  # save model


# In[29]:


run_pipeline('DisasterResponse.db')


# In[ ]:


#if __name__ == '__main__':
 #   data_file = sys.argv[1]  # get filename of dataset
  #  run_pipeline(data_file)  # run data pipeline

