import os
import json
import joblib
import string
import numpy as np
from catboost import CatBoostClassifier
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the intents from the JSON file
with open('chat.json') as file:
    intents = json.load(file)

# Preprocess the data
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]
    return ' '.join(tokens)

# Load the CatBoost model using joblib
model = joblib.load('chatbot_model.joblib')

# Create a Flask app
app = Flask(__name__)

# Define a route for the incoming WhatsApp messages
@app.route('/bot', methods=['POST'])
def bot():
    # Get the incoming message
    incoming_message = request.values.get('Body', '').lower()

    # Get the sender's phone number
    sender = request.values.get('From', '')

    # Create a Twilio messaging response object
    twilio_response = MessagingResponse()

    # Get the prediction from the chatbot model
          # Load the TfidfVectorizer
    vectorizer = joblib.load('vectorizer.joblib')
   


    X_test = vectorizer.transform([preprocess_text(incoming_message)]).toarray()
    prediction = model.predict(X_test)
    for intent in intents['intents']:
        if intent['tag'] == prediction:
            response = np.random.choice(intent['responses'])
            twilio_response.message(response)

    return str(twilio_response)

if __name__ == '__main__':
 
    # Run the Flask app
    app.run()
