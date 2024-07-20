from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')

app = Flask(__name__)

# Load your pre-trained Keras model
model = load_model('text_summarization_rnn_light.h5')

def scrape_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the first paragraph (or any relevant text)
        first_paragraph = soup.find('p').get_text() if soup.find('p') else ''
        
        return first_paragraph
    except requests.RequestException as e:
        return {'error': str(e)}

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def preprocess_tokens(tokens, max_length=50):
    # Convert tokens to integer sequences (you need a tokenizer here)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(tokens)
    sequences = tokenizer.texts_to_sequences([tokens])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

@app.route('/api/get_url', methods=['POST'])
def get_url():
    try:
        data = request.json
        url = data.get('url')
        if url:
            first_paragraph = scrape_url(url)
            if isinstance(first_paragraph, dict) and 'error' in first_paragraph:
                return jsonify({'status': 'error', 'message': first_paragraph['error']}), 500
            
            # Tokenize the first paragraph
            tokens = tokenize_text(first_paragraph)
            # Preprocess and pad tokens
            padded_tokens = preprocess_tokens(tokens)
            
            # Make prediction with the model
            predictions = model.predict(padded_tokens)
            
            # Convert predictions to a list
            predictions_list = predictions.tolist()
            
            return jsonify({
                'model_output': predictions_list
            }), 200
        else:
            return jsonify({'status': 'error', 'message': 'URL not provided'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
