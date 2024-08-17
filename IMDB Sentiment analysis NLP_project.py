import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import nltk

# Import necessary libraries for model training
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import Sequential
from keras.layers import Dense

# Initialize NLTK tools
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Sample data preprocessing (replace with your actual data processing steps)
import pandas as pd
df = pd.read_csv('IMDB Dataset.csv')
df['clean_text'] = df['review'].apply(lambda x: re.sub("<.*?>", "", x))
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', "", x))
df['clean_text'] = df['clean_text'].str.lower()
df['tokenize_text'] = df['clean_text'].apply(lambda x: word_tokenize(x))
df['filtered_text'] = df['tokenize_text'].apply(lambda x: [word for word in x if word not in stop_words])
df['stem_text'] = df['filtered_text'].apply(lambda x: [stemmer.stem(word) for word in x])
X = df['stem_text']
y = df['sentiment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train.apply(lambda x: ' '.join(x)))
X_test = tfidf.transform(X_test.apply(lambda x: ' '.join(x)))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=2)

# Define and train the model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(2, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3)

# Save the model and TF-IDF vectorizer using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# Load the model and TF-IDF vectorizer using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tf_idf_vector = pickle.load(tfidf_file)

# Function to predict sentiment
def predict_sentiment(review):
    cleaned_review = re.sub('<.*?>', '', review)
    cleaned_review = re.sub(r'[^\w\s]', '', cleaned_review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stemmer.stem(word) for word in filtered_review]
    tfidf_review = tf_idf_vector.transform([' '.join(stemmed_review)])

    # Ensure tfidf_review has the correct shape
    tfidf_review = tfidf_review.toarray()

    sentiment_prediction = model.predict(tfidf_review)

    if sentiment_prediction[0][1] > 0.5:  # Adjust threshold as needed
        return "Positive"
    else:
        return "Negative"

# Streamlit UI
st.title('Sentiment Analysis')
review_to_predict = st.text_area('Enter your review here:')
if st.button('Predict Sentiment'):
    predicted_sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted Sentiment:", predicted_sentiment)
