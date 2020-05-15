from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
import re
import pandas as pd
import numpy as np
from googlesearch import search 
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questiondf = pd.read_csv("question.csv")
asindf = pd.read_csv("asindf.csv")
answerdf = pd.read_csv("answer.csv")

with open('asindftfidfvector.pkl', 'rb') as f:
    asintfidf_vectorizer = pickle.load(f)
with open('asindftfidf.pkl', 'rb') as f:
    asintfidf_matrix = pickle.load(f)
with open('qtfidfvector.pkl', 'rb') as f:
    qtfidf_vectorizer = pickle.load(f)
with open('questiontfidf.pkl', 'rb') as f:
    qtfidf_matrix = pickle.load(f)

def clean(x):
    x = x.lower()
    x = re.sub(r'[^\w]', ' ', x)
    return x
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
#Greeting Inputs
import random
GREETING_INPUTS = ["hi", "hello", "greetings", "wassup", "hey"]
GREETING_RESPONSES=["Hey", "hi",  "what's good", "hello", "hey there"]
def greeting(sentence):
  for word in sentence.split():
    if word.lower() in GREETING_INPUTS:
      return random.choice(GREETING_RESPONSES)

def cosine(question) :
    result0 = []
    result1 = []
    result2 = []
    query_vect = qtfidf_vectorizer.transform([question])
    similarity = cosine_similarity(query_vect, qtfidf_matrix)
    top_5_simmi = similarity[0].argsort()[-15:][::-1]

    for i in top_5_simmi:
        result0.append(questiondf.iloc[i]['question'])
        result1.append(answerdf.iloc[i]['answer'])       
        result2.append(similarity[0, i])

    return result0 , result1 , result2

def asin(question) :
    query_vect = asintfidf_vectorizer.transform([question])
    similarity = cosine_similarity(query_vect, asintfidf_matrix)
    top_5_simmi = similarity[0].argsort()[-5:][::-1]
    result = []
    result1 = []
    for i in top_5_simmi:
        result.append(asindf.iloc[i]['asin'])  
        result1.append(similarity[0, i])  
    return result , result1


def response(user_response , firstq , typeq , variable):
    user_response = clean(user_response)
    robo_response = ''
    score = [0]
    try :
        if (typeq == 0) : #first robo response
            firstq = user_response
            question , answer , score = cosine(user_response)
            flag = 1
            if ((score[0] > 0.9) and (typeq == 0)): #lucky
                robo_response = robo_response + answer[0]
                flag = 8
                variable = question
            elif((score[0] <= 0.9) and (typeq== 0)):
                productcode , similarity = asin(user_response)
                if similarity[0]>0.4:
                    productname = productq(productcode[0])
                    robo_response = robo_response+"I am confused with the product you are talking about "
                    if productname != 'product not found':
                        robo_response = robo_response+"Is it "+productname + " ?"
                        flag = 2
                        variable = productcode
                    else :
                        robo_response = robo_response+"I am finding similar questions for you"
                        flag = 3
                        variable = productcode
                elif (similarity[0]<= 0.4):
                    robo_response = robo_response+"I am confused with the product you are talking about i found similar product based on your question reply me its index"
                    num = 0
                    for i in productcode:
                        num += 1
                        k = i
                        if  productq(i) != 'product not found':
                            k = productq(i)
                        robo_response = robo_response + "\n" + str(num) + k
                        flag = 4
                    variable = productcode
                    robo_response = robo_response+"\n Reply product number if your product is in it"
        elif(typeq== 101):
            productcode , similarity = asin(user_response)
            if similarity[0]>0.4:
                productname = productq(productcode[0])
                robo_response = robo_response+"I am confused with the product you are talking about "
                if productname != 'product not found':
                    robo_response = robo_response+"Is it "+productname + " ?"
                    flag = 2
                    variable = productcode
                else :
                    robo_response = robo_response+"I am finding similar questions for you"
                    flag = 3
                    
        elif (typeq== 102):
            productcode , similarity = asin(user_response)
            robo_response = robo_response+"I am confused with the product you are talking about i found similar product based on your question reply me its index"
            num = 0
            for i in productcode:
                num += 1
                k = i
                if  productq(i) != 'product not found':
                    k = productq(i)
                robo_response = robo_response + "\n" + str(num) + k
                flag = 4
            variable = productcode
            robo_response = robo_response+"\n Reply product number if your product is in it"

        elif typeq == 1 : #flag1 lucky answers
            robo_response = variable[int(float(user_response))]
        elif typeq == 2 : #flag2 yes/no
            if "yes" in user_response:
                question , answer , score = cosine(firstq + variable[0] )
                robo_response = robo_response+"Here I found top questions and answers related to your query \n 1."+question[0] +"\n 2."+question[1] +"\n 3."+question[2] +"\n 4."+question[3] +"\n 5."+question[4]
                flag = 6
                variable = answer
            elif "no" in user_response:
                productcode = variable
                robo_response = robo_response+"I am confused with the product you are talking about i found similar product based on your question reply me its index"
                num = 0
                for i in productcode:
                    num += 1
                    k = i
                    if  productq(i) != 'product not found':
                        k = productq(i)
                    robo_response = robo_response + "\n" + str(num) + k
                robo_response = robo_response+"\n Reply product number if your product is in it"
                flag = 4
                variable = productcode
        


        elif typeq == 3 :  #flag 3
            question , answer , score = cosine(variable)
            robo_response = robo_response+"Here I found top questions and answers related to your query \n 1."+question[0] +"\n 2."+question[1] +"\n 3."+question[2] +"\n 4."+question[3] +"\n 5."+question[4]
            flag = 6
            variable = answer
        elif typeq == 4 : #choose product
            try:
                if 0 < (int(float(user_response)))< 6 :
                    k = variable[int(float(user_response))-1]
                    question , answer , score = cosine(firstq + k)
                    robo_response = robo_response+"Here I found top questions and answers related to your query \n 1."+question[0] +"\n 2."+question[1] +"\n 3."+question[2] +"\n 4."+question[3] +"\n 5."+question[4]
                    flag = 6
                    variable = answer
                elif "not" in user_response :
                    robo_response = "Please tell me more about it"
                    flag = 0
                elif "no" in user_response :
                    robo_response = "Please tell me more about it"
                    flag = 0
                else :
                    robo_response = "Invalid input Do you want to search other question"
                    flag = 7
                    variable = 0
            except :
                if "not" in user_response :
                    robo_response = "Please tell me more about it"
                    flag = 0
                elif "no" in user_response :
                    robo_response = "Please tell me more about it"
                    flag = 0
                else :
                    robo_response = "Invalid input Do you want to search other question"
                    flag = 7
                    variable = 0
            
        elif typeq == 6 : #answers
            try :
                if 0 < (int(float(user_response)))< 6 :
                    robo_response = variable[int(float(user_response))-1]
                    robo_response = robo_response+"If you want another questions answer , tell me its number"
                    flag = 6
                    
                
                elif "no" in user_response :
                    robo_response = "Thank you for your query"
                    flag = 0
                    variable = 0
                    firstq = 0
                else :
                    robo_response = "Invalid input Do you want to search other question"
                    flag = 7
                    variable = 0
            except :
                robo_response = "Invalid input Do you want to search other question"
                flag = 7
                variable = 0
            
        elif typeq == 7: # reset
            if "yes" in user_response:
                flag = 0
                variable = 0
                firstq = 0
                robo_response = "Okay ... tell me your next question"
            elif "no" in user_response:
                flag =9
                robo_response = "Thank you"

        elif typeq == 8 :#not lucky
            if "no" in user_response:
                flag = 101
            else :
                flag = 7
        elif typeq == 9 or user_response == 'nextquestion' or user_response == 'stop' :#thankyou
            robo_response = "you can tell me another question"
            flag = 0
            variable = 0
            firstq = 0
    except :
            robo_response = "Invalid input tell me another question"
            flag = 0
            variable = 0
            firstq = 0

    return robo_response , flag , variable ,firstq


    
typeq = 0
firstq = 0
variable = 0
resp = 0

def robot(user_response):
  global resp 
  global typeq 
  global variable 
  global firstq
  user_response = clean(user_response)
  if(user_response != 'bye'):
    if('thanks' in user_response  or 'thank' in user_response or 'thankyou'in user_response ):
      k = "DOCBot: You are welcome !"
    if('want' in user_response  or 'know' in user_response or 'question'in user_response ):
      k = "Yeah ! Tell me")
    else:
      if(greeting(user_response) != None):
        k = "DOCBot: "+greeting(user_response))
      else:
        resp , typeq , variable , firstq = response(user_response ,firstq ,typeq , variable)
        k = "Bot: "+str(resp)
  return k




app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")    

def get_bot_response():
    userText = request.args.get('msg')
    response = robot(userText)
    return response


if __name__ == "__main__":
    app.run()
