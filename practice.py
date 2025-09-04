import pandas as pd
import pickle as pk
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.title("Sentiments Data with Logistic Regression")

# Load model
load_model = pk.load(open("sentiments_text_logisticregression.pickle", 'rb'))

# Download stopwords
nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

# Define mapping (adjust if your labels are different)
label_mapping = {
    0: "Negative 😞",
    1: "Neutral 😐",
    2: "Positive 😀"
}

# Text input
text = st.text_area("Enter your text:--")

if st.button("Predict"):
    if text.strip() == "":
        st.write("⚠️ Please enter some text")
    else:
        # Put text in dataframe (just like your terminal code)
        sentiment_data = {'predict_sentiments':[text]}
        sentiment_data_df = pd.DataFrame(sentiment_data)

        # Clean text
        sentiment_data_df['predict_sentiments'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]),sentiment_data_df['predict_sentiments']))
        sentiment_data_df['predict_sentiments'] = sentiment_data_df['predict_sentiments'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

        # Predict
        result = load_model.predict(sentiment_data_df['predict_sentiments'])

        # Convert numeric prediction to label
        sentiment_label = label_mapping.get(result[0], "Unknown")

        # Show result
        st.write("Predicted Sentiment Category = ", sentiment_label)