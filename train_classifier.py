import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,label_ranking_average_precision_score
from sklearn.model_selection  import GridSearchCV

nltk.download(['punkt','stopwords','wordnet'])

def load_data(database_filepath):
    """
    Function: load data from database and return X and y.
    Args:
    database_filepath(str): database file name included path
    Return:
      X(pd.DataFrame): messages for X
      y(pd.DataFrame): labels part in messages for y
      category_names(str):category names
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql('SELECT * from Messages', engine)
    X = df['message']
    # result is multiple classifty.
    y = df.iloc[:,4:]
    category_names=y.columns.values
    return X, y, category_names

def tokenize(text):
    """
    Function: tokenize the text
    Args:  source string
    Return:
    clean_tokens(str list): clean string list
    """
    #normalize text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())

    #token messages
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]

    #sterm and lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model_0():
    """
    Function: build model that consist of pipeline
    Args:
      N/A
    Return
      cv(model): Grid Search model
    """

    pipeline = Pipeline([
        ('vect',TfidfVectorizer(stop_words='english',tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100,random_state=20)))
    ])
    parameters = {
        'clf__estimator__criterion':['entropy']
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    return cv
def build_model():
    """
    Function: build model that consist of pipeline
    Args:
      N/A
    Return
      cv(model): Grid Search model
    """

    pipeline = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200,random_state=20)))
        ])
    parameters = {
        'clf__estimator__min_samples_leaf': [1,10],
        
       
        }
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: evaluate model
    Args:
      model,
      X_test: X test dataset
      Y_test: y test dataset
      category_names:category names of y
    Return
      N/A
    """
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report

    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), 
        target_names=category_names))

def save_model(model, model_filepath):
    """
    Function: save model as pickle file.
    Args:
      model:target model
    Return:
      N/A
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    