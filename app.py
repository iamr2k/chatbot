from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
import re
import pandas as pd
import numpy as np

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
processed = pd.read_csv("data.csv")


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
   
    if request.method == 'POST':

        message = request.form.get('message')
                
        

        with open('matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open('vector.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)


        result1 = []
        result2 = []
        result3 = []

        message = message
       
        query_vect = tfidf_vectorizer.transform([message])
            
        similarity = cosine_similarity(query_vect, tfidf_matrix)
        top_5_simmi = similarity[0].argsort()[-5:][::-1]
       
  
        for i in top_5_simmi:
            try:
                result1.append(processed.iloc[i]['question'])   
            except:
                result1.append("na") 
            try:  
                result2.append(similarity[0, i])
            except:
                result2.append("na")
            try:
                result3.append(processed.iloc[i]['answer'])
            except:
                result3.append("na")
        print("***************************",result1)
            



        

    return render_template('result.html', result1=result1,result2=result2,result3=result3)



if __name__ == '__main__':
    app.run(debug=True)
