# A.I-Chatbot

Mar 2020 – May 2020

Project descriptionBusiness Objective:
An e-commerce company wants to build an
algorithm to retrieve top 5 Question and
answers based on the user given user input.

Solution :
The input given by the user is first cleaned and features extracted using 3 different algorithms. LSTM Deep Learning classifier will first identify the type of user question (Yes/No ,Open-ended , etc.) The second algorithm Gensim model will identify the 'Product' from the user user input , and the final K-means clustering algorithm will find the cluster in which the user input will fall . This three predictions are embedded to the user input and fed into Cosine similarity algorithm to find similar questions from the Tfidf vector space.
See the deployed model in heroku
http://aiquestionsearch.herokuapp.com/
