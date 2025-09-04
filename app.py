#text_data_sentiments_logistic_regression 
#pip install nltk
import pandas as pd
import pickle as pk
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

st.write("Sentiments Data with logistic regression")

load_model = pk.load(open("sentiments_text_logisticregression.pickle", 'rb'))

nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

text = st.text_area("Enter your text:--")

if st.button("predict"):
   # df = pd.DataFrame({
   #    'cleaned':[text]
   #    })  we can write the code of 3 lines above and continue from else: line  or simply use if else condtion to modify it
      #  sentiment = input("Enter text = ") which is already given just before the dataframe
   if text.strip() == "":
      st.write("⚠️ Please enter some text")
   else:
      # Put text in dataframe
      sentiment_data = {'predict_sentiments':[text]}
      sentiment_data_df = pd.DataFrame(sentiment_data)

      # Clean text
      sentiment_data_df['predict_sentiments'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), sentiment_data_df['predict_sentiments']))
      sentiment_data_df['predict_sentiments'] = sentiment_data_df['predict_sentiments'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

      # Predicticting result 
      # predict_news_cat = load_model.predict(sentiment_data_df['predict_sentiments'])
      result = load_model.predict(sentiment_data_df['predict_sentiments'])

      # Show result
      #  st.write("Predicted sentiment category = ",predict_news_cat[0])
      st.write("Predicted sentiment category = ",result[0])




