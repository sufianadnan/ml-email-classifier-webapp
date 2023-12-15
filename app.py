from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, render_template
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report

nltk.download('stopwords')
nltk.download('punkt')
# Initialize Flask app
TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
# Load the models and vectorizers


def preprocess_text(text):
    # Replace contractions and specific words
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # Replace specific patterns
    text = re.sub(r'\n', " ", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+?$', '', text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)

    # Remove stopwords using NLTK
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def retrain_models(csv_file_path):
    print("RETRAINING MODELS")
    try:
        combined_df = pd.read_csv(csv_file_path)
    except UnicodeDecodeError:
        combined_df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

    # Preprocess the dataset
    combined_df['sub_mssg'] = combined_df['sub_mssg'].astype(str)
    combined_df['sub_mssg'] = combined_df['sub_mssg'].apply(preprocess_text)

    # Split the dataset
    X_combined = combined_df['sub_mssg']
    y_combined = combined_df['label']
    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42)

    # Vectorize and retrain with CountVectorizer
    count_vectorizer = CountVectorizer(stop_words='english')
    X_train_combined_count = count_vectorizer.fit_transform(X_train_combined)
    X_test_combined_count = count_vectorizer.transform(X_test_combined)
    naive_bayes_combined_count = MultinomialNB()
    naive_bayes_combined_count.fit(X_train_combined_count, y_train_combined)

    # Vectorize and retrain with TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_combined_vectorized = tfidf_vectorizer.fit_transform(X_train_combined)
    X_test_combined_vectorized = tfidf_vectorizer.transform(X_test_combined)
    naive_bayes_combined = MultinomialNB()
    naive_bayes_combined.fit(X_train_combined_vectorized, y_train_combined)

    # Save the models
    with open('pkl/naive_bayes_tfidf_model.pkl', 'wb') as file:
        pickle.dump(naive_bayes_combined, file)
    with open('pkl/tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    with open('pkl/naive_bayes_count_model.pkl', 'wb') as file:
        pickle.dump(naive_bayes_combined_count, file)
    with open('pkl/count_vectorizer.pkl', 'wb') as file:
        pickle.dump(count_vectorizer, file)
    print("MODEL RETRAINED")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    feedback_data = request.get_json()

    # Flatten and escape the message
    flattened_message = feedback_data['message'].replace('\n', ' ').replace('\r', ' ')
    escaped_message = '"' + flattened_message.replace('"', '""') + '"'

    feedback_file_path = 'final_dataset.csv'

    file_exists = os.path.isfile(feedback_file_path)
    with open(feedback_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            writer.writerow(['sub_mssg', 'label'])

        writer.writerow([escaped_message, feedback_data['feedback']])

    retrain_models('final_dataset.csv')
    return jsonify({'status': 'success'})



@app.route('/classify-email', methods=['POST'])
def classify_email():
    with open('pkl/naive_bayes_tfidf_model.pkl', 'rb') as file:
        naive_bayes_tfidf = pickle.load(file)

    with open('pkl/tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    with open('pkl/naive_bayes_count_model.pkl', 'rb') as file:
        naive_bayes_count = pickle.load(file)

    with open('pkl/count_vectorizer.pkl', 'rb') as file:
        count_vectorizer = pickle.load(file)

    data = request.data.decode('utf-8')
    preprocessed_text = preprocess_text(data)

    # Vectorize and predict using both models
    tfidf_vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
    count_vectorized_text = count_vectorizer.transform([preprocessed_text])

    result_tfidf = naive_bayes_tfidf.predict(tfidf_vectorized_text)[0]
    result_count = naive_bayes_count.predict(count_vectorized_text)[0]

    # Prepare and send the JSON response
    response = {
        'TfidfVectorizer': int(result_tfidf),
        'CountVectorizer': int(result_count)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
