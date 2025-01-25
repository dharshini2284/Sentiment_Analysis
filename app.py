## IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random  # FOR PSEUDO RANDOM NUMBER GENERATION
from wordcloud import WordCloud  # TO VISUALIZE THE TEXT DATA
from wordcloud import STOPWORDS

# IMPORTING THE NLTK PACKAGE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Set NLTK data path to your local directory
nltk.data.path.append(r"C:\Users\dhars\OneDrive\Desktop\nltk_download\nltk_data")

# FROM sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Import the Streamlit library
import streamlit as st

# DEFINING FUNCTIONS TO PERFORM THE NLTK PREPROCESSING STEPS

# 1. USED TO TOKENIZE THE TEXT DATA - EX : I AM ASH =>['I','AM','ASH']
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


# 2. USED TO REMOVE COMMON ENGLISH STOPWORDS FROM TOKENISED DATA
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens


# 3. USED TO NORMALIZE THE TOKEN - EX : ["loving", "cats"] => ["love", "cat"]
def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens


# 4. MAIN FUNCTION TO PERFORM ALL NLTK PREPROCESSING
def preprocess_text(text):
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return ' '.join(tokens)  # Join the tokens back into a single string


# 5. SIMILAR TO LEMMATIZATION
def stem_text(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens


# READING THE CSV FILE
DF2 = pd.read_csv('Preprocessed_DataFrame.csv')
print(DF2)

# ---------------------------------------------
# KNN
# ---------------------------------------------
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(DF2['review'], DF2['Sentiment'], test_size=0.2, random_state=42)

# Create a TfidfVectorizer to convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create and train a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_tfidf, y_train)

# ---------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------
# Create and train a Logistic Regression classifier
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_tfidf, y_train)

# Streamlit App
st.markdown("# DRUG REVIEW SENTIMENT ANALYSIS")
st.title("TEST DATA INPUT")

# User Inputs
drug_name = st.text_input("ENTER THE NAME OF THE DRUG : ", ' ')
condition_name = st.text_input("ENTER THE CONDITION FOR WHICH THE DRUG IS USED : ", " ")
review = str(st.text_input("ENTER THE REVIEW OF THE DRUG : ", " "))

user_input = {'drugName': [drug_name], 'condition': [condition_name], 'review': [review]}
test_data = pd.DataFrame(user_input)

if st.button("Submit"):
    st.success("User Input DataFrame:")
    st.write(test_data)

    # PROCESSING THE TEST DATA
    test_data['review'] = test_data['review'].apply(preprocess_text)

    # Logistic Regression Prediction
    st.markdown("PREDICTED SENTIMENT USING LOGISTIC REGRESSION ALGORITHM")
    test_tfidf = tfidf_vectorizer.transform(test_data['review'])
    y_pred = logistic_regression.predict(test_tfidf)
    st.success(f"The predicted sentiment is: {y_pred[0]}")

    # KNN Prediction
    st.markdown("PREDICTED SENTIMENT USING KNN ALGORITHM")
    y_pred_knn = knn_classifier.predict(test_tfidf)
    st.success(f"The predicted sentiment using KNN is: {y_pred_knn[0]}")
