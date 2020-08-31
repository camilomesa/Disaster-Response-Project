# Disaster Response Project
This project implements a random forest classifier model to categorize messages sent by people during natural disasters. After classification, the messages can be directed to the appropriate disaster relief agency. The training data provided by [Figure Eight](https://appen.com/) was mined using ETL and natural language processing pipelines.


![Source: weather.gov](https://www.weather.gov/images/safety/ia-2008-2-lg.jpg) 


The README file includes how to run the Python scripts and web app Comments are used effectively and each function has a docstring.

### Table of contents
1. [Summary](#summary)
2. [File Descriptions](#files)
3. [Usage](#ussage)
4. [Licensing, Authors, and Acknowledgments](#licensing)


## Summary <a name="summary"></a>

In this project we analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. 
The data consists of messages from social media, news or disaster response organizations and their corresponding classes (e.g. earthquake, fire, etc.).

The main components of the project are the following:
- ETL Pipeline: Loads the messages and categories datasets, cleans the data, and stores it in a SQLite database
- ML Pipeline: Loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file
- Flask Web App: Displays the results and implements the machine learning model to classify new messages provided by the user. A statistical visualization of the training dataset is provided as well.




## File Descriptions<a name="files"></a>

Here's the file structure of the project:
- app (web application)
  - template
    * master.html (main page of web app)
    * go.html  (classification result page of web app)
  - run.py  (flask file that runs web app)

- data
  - disaster_categories.csv  (training data to process)
  - disaster_messages.csv  (training data to process)
  - process_data.py (ETL script)
  - DisasterResponse.db (database with clean data)

- models
  - train_classifier.py (natural language processing pipeline script)
  - classifier.pkl (saved model)

- notebooks (jupyter notebooks used to develop the source code of the ETL, machine learning pipelines and web application

- README.md


## Usage

Run the following commands in the project's root directory to set up your database and model.

0. Update and/or synch the versions of python/packages execute:
'pip3 install -U scikit-learn scipy matplotlib'
'pip install --upgrade pip'

1. To run ETL pipeline that cleans data and stores in database:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. To run ML pipeline that trains classifier and saves
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


