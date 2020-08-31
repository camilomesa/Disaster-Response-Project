# Disaster Response Project
This project implements a random forest classifier model to categorize messages sent by people during natural disasters. After classification, the messages can be directed to the appropriate disaster relief agency. The training data provided by [Figure Eight](https://appen.com/) was mined using ETL and natural language processing pipelines.


![alt text](https://www.weather.gov/images/safety/ia-2008-2-lg.jpg) 


The README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository. Comments are used effectively and each function has a docstring.

### Table of contents
1. [Summary](#summary)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgments](#licensing)


## Summary <a name="summary"></a>

In this project we analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. 
The data consists of messages from social media, news or disaster response organizations and their corresponding classes (e.g. earthquake, fire, etc.).

The main components of the project are the following:
- ETL Pipeline: Loads the messages and categories datasets, cleans the data, and stores it in a SQLite database
- ML Pipeline: Loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file
- Flask Web App: Displays the results and implements the machine learning model to classify new messages provided by the user. A statistical visualization of the training dataset is provided as well.




## File Descriptions<a name="files"></a>

Here's the file structure of the project:
- app
  - template
   * master.html (main page of web app)
   * go.html  (classification result page of web app)
  - run.py  (flask file that runs app)

- data
 - disaster_categories.csv  (data to process)
 - disaster_messages.csv  (data to process)
 - process_data.py (ETL script)
 - InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
