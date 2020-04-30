from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
import re
import pandas as pd
import numpy as np



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.optimizers import Adam
from keras import models as models

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

question = pd.read_csv("question.csv")
asindf = pd.read_csv("asindf.csv")
answer = pd.read_csv("answer.csv")
features = pd.read_csv("feature.csv")




def clean(x):
    x = re.sub(r'http\S+', '', x)
    x = re.sub(r'[^\w]', ' ', x).lower())
    x = re.sub(r'(?:^| )\w(?:$| )', ' ', x).strip())
    x = x.replace('  ',' ')

    return x

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
   
    if request.method == 'POST':

       user_input_question = request.form.get('message')

       question0 = [[0]]
       question0[0][0] = user_input_question
       question = np.array(question0)
       data = pd.DataFrame(question0)
       data["cleaned"] = question0

       texts_test1 = data.cleaned.astype(str)
       model1 = models.load_model('/content/faq/lstm.h5')
       with open('/content/faq/dnntokenizer.pkl', 'rb') as f:
           tokenizer = pickle.load(f)

       sequences_test1 = tokenizer.texts_to_sequences(texts_test1)

       word_index = tokenizer.word_index
       x_test1 = pad_sequences(sequences_test1, maxlen=20)
       pred31 = model1.predict(x_test1)
       if (pred31 >0.5) == True : 
           pred32 = "yes/no"
       else :
           pred32 = "open-ended"

       with open('/content/faq/asindftfidfvector.pkl', 'rb') as f:
           tfidf_vectorizer = pickle.load(f)
       with open('/content/faq/asindftfidf.pkl', 'rb') as f:
           tfidf_matrix = pickle.load(f)

       question= question0[0][0]

       query_vect = tfidf_vectorizer.transform([question])
            
       similarity = cosine_similarity(query_vect, tfidf_matrix)
       top_5_simmi = similarity[0].argsort()[-5:][::-1]

       result = []
       for i in top_5_simmi:
           result.append(asindf.iloc[i]['asin'])

       question = question + " " +str(result[0]) + " " +pred32

       with open('/content/faq/qtfidfvector.pkl', 'rb') as f:
           tfidf_vectorizer = pickle.load(f)
       with open('/content/faq/questiontfidf.pkl', 'rb') as f:
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
           #print("top questions", processed[processed.asin == asindf.iloc[i]['asin'] ].question)
           result2.append(similarity[0, i])


        

    return render_template('result.html', result1=result0,result2=result2,resul3=result1)



if __name__ == '__main__':
    app.run(debug=True)
