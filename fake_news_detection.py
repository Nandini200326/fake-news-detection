
import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
news_df = pd.read_csv("D:\\fake_news_detection\\train.csv\\train.csv")
news_df.head()
news_df.isna().sum()
news_df = news_df.fillna(" ")
news_df.isna().sum()
news_df["content"]= news_df['author']+" "+news_df["title"]

ps =PorterStemmer()
def stemming(content):
    stemmed=re.sub('[^a-zA-Z]'," ",content)
    stemmed=stemmed.lower()
    stemmed=stemmed.split()
    stemmed=[ps.stem(word) for word in stemmed if not word in stopwords.words('english')]
    stemmed=" ".join(stemmed)
    return stemmed
news_df['content']=news_df['content'].apply(stemming)
x=news_df['content'].values
y=news_df['label'].values
vector = TfidfVectorizer()
vector.fit(x)
x=vector.transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=1)
model = LogisticRegression()
model.fit(x_train,y_train)


st.title("fake news detection" )
input_text=st.text_input("enter the news article ")


def prediction(input_text):
    input_data=vector.transform([input_text])
    prediction=model.predict(input_data)
    return prediction[0]

if input_text:
    pred=prediction(input_text)
    if pred==1:
        st.write("the news is fake")
    else:
        st.write("the news is real")    