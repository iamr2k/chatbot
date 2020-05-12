from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
import re
import pandas as pd
import numpy as np
from googlesearch import search 


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.optimizers import Adam
from keras import models as models
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questiondf = pd.read_csv("question.csv")
asindf = pd.read_csv("asindf.csv")
answerdf = pd.read_csv("answer.csv")






def clean(x):
    x = re.sub(r'http\S+', '', x)
    x = re.sub(r'[^\w]', ' ', x).lower()
    x = re.sub(r'(?:^| )\w(?:$| )', ' ', x).strip()
    x = x.replace('  ',' ')

    return x

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        user_input_question = clean(request.form.get('message'))
        question0 = [[0]]
        question0[0][0] = user_input_question
        question = np.array(question0)
        data = pd.DataFrame(question)
        data["cleaned"] = question
 
        texts_test1 = data.cleaned.astype(str)
        model1 = models.load_model('lstm.h5')
        with open('dnntokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
 
 
        sequences_test1 = tokenizer.texts_to_sequences(texts_test1)
        x_test1 = pad_sequences(sequences_test1, maxlen=20)
        pred31 = model1.predict(x_test1)
        if (pred31 >0.5) == True : 
            pred32 = "yes/no"
        else :
            pred32 = "open-ended"

        print('**************** QuestionType = ',pred32)
        questtype = pred32

        with open('pca.pkl', 'rb') as f :
            pca = pickle.load(f)
        with open('kmeans.pkl', 'rb') as f :
            kmeans = pickle.load(f)
        with open('tfidf.pkl', 'rb') as f :
            tfidf = pickle.load(f)
 
        def cluster (userinput):
             message = [[0]]
             message[0][0] = userinput
             message = np.array(message)
             data = pd.DataFrame(message)
             data["content"] = message
             tf_idf = tfidf.transform(data["content"])
             tf_idf_norm = normalize(tf_idf)
             tf_idf_array = tf_idf_norm.toarray()
             Y_sklearn = pca.transform(tf_idf_array)
             prediction = kmeans.predict(Y_sklearn)
             clusterid = str(prediction[0])
             return clusterid
        def productq(inputq):
            j = 'product not found'
            query = "amazon "+inputq
            try :
                for j in search(query, tld="com", num=1, stop=1):
                j = j.split(sep='.com/')[1]
                j = j.split(sep='/')[0]
             except :
                j = 'product not found'
            return j
        

         
        with open('asindftfidfvector.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('asindftfidf.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
 
        question= question0[0][0]
        clusterid = cluster(question)
        clusterresult = ' cluster'+clusterid
        print("****************clusterresult",clusterresult)
        

 
        query_vect = tfidf_vectorizer.transform([question])
             
        similarity = cosine_similarity(query_vect, tfidf_matrix)
        top_5_simmi = similarity[0].argsort()[-5:][::-1]
 
        result = []
        for i in top_5_simmi:
            result.append(asindf.iloc[i]['asin'])
        print('*************** Product code (asin)',result[0])
        
        question = question + " " +str(result[0]) + " " +pred32 + " " +clusterresult
        print("********************************* Final transformed question = ",question)
        productcode = str(result[0])
        productname = productq(productcode)
        with open('qtfidfvector.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('questiontfidf.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
 
        query_vect = tfidf_vectorizer.transform([question])
        similarity = cosine_similarity(query_vect, tfidf_matrix)
        top_5_simmi = similarity[0].argsort()[-15:][::-1]
 
        result0 = []
        result1 = []
        result2 = []
 
 
        for i in top_5_simmi:
            result0.append(questiondf.iloc[i]['question'])
            result1.append(answerdf.iloc[i]['answer'])       
            result2.append(similarity[0, i])
 
 

    return render_template('result.html', result0=result0,result1=result1,result2=result2,productcode=productcode,productname = productname,questtype=questtype,clusterid=clusterid)
 


if __name__ == '__main__':
    app.run(debug=True)
