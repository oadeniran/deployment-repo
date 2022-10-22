import numpy as np
import pandas as pd
import re
import string
from flask import Flask, request, render_template, session, redirect, url_for, flash
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pickle
import gc
import xgboost 

stop_words =['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
            'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',
             'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
             'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
             'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
             'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
             'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 
             'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
             'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
             'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
              "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
               "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
               'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def remove_stopwords(row):
    sentence = row['facts']
    cleaned = []
    sentence = re.sub(r'</*[a-z]+>', '', sentence)
    sentence = re.sub(r'\n', '', sentence)
    for punct in string.punctuation:
        sentence = sentence.replace(punct, '')
    for word in sentence.split():
        if word not in stop_words:
            cleaned.append(word)
    return ' '.join(cleaned).lower()

app = Flask(__name__)
app.secret_key = 'team-anaconda'

model = xgboost.XGBClassifier()
model.load_model('Smodel.bin')

with open('Sbow_cv.pk', 'rb') as f:
    cv = pickle.load(f)

@app.route('/', methods = ['GET', 'POST'])
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/prediction', methods = ['POST'])
def prediction():
    feature_list = request.form.to_dict()
    print(feature_list)
    feature_list['issue_area'] = int(feature_list['issue_area'])
    final = pd.DataFrame(feature_list, index = [1])
    final['facts'] = final.apply(remove_stopwords, axis = 1)
    trasf_facts = pd.DataFrame(cv.transform(final['facts']).toarray())
    trsf_data = pd.concat([trasf_facts, final.reset_index().drop(['index','facts'], axis = 1)], axis = 1)
    prediction = model.predict(trsf_data)
    if prediction == 1:
        prediction = 'WIN'
    else:
        prediction = 'LOSE'

    
    return render_template('index.html', prediction = prediction)

@app.route('/help')
def help():
    return render_template('help.html')
    

if __name__ == "__main__":
    app.run(debug = True)
