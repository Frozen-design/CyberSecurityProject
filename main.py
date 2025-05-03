import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.preprocessing import sequence


import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

from Dependencies.sup import print_if, parse_csv
from Dependencies.bow import simple_tokenizer
from Dependencies.KNN import train_knn_model, knn_model_stats, vectorize_normalize_data

from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

'''
Phishing attacks are one of the most common cybersecurity threats. 
They use fake emails and text messages that look like they come from trusted sources and while email and phone providers use filters to block these messages, some still get through. 
One missed message can be enough to trick someone. Attackers often copy natural writing patterns, use new tactics, and change their messages to target specific people. 
These attacks can lead to data breaches and identity theft. Most filters use fixed rules or old patterns, which may cause them to block safe messages or miss harmful ones. 
We need a system that can keep up with new phishing methods and learn from past ones. 
We plan to train an AI model that checks both messages already blocked and those that slip through current filters. 
'''

SPRINT = 1
STATS = 1
TXT_COL = 'Text'
CLASS_COL = 'Class'
HEADER_NAMES = ['Text', 'Class']
DTYPES = {TXT_COL: str, CLASS_COL: int}

def main():
    """Main function to execute the KNN model for phishing email detection."""
    print_if("Starting KNN model for phishing email detection...", SPRINT)
    user_file_path = input("Enter the path to the dataset (default: Data/fraud_emails.csv): ").strip()
    if not user_file_path:
        user_file_path = "Data/fraud_emails.csv"
    print_if("Loading dataset...", SPRINT)
    try:
        user_file_path = user_file_path.strip('"').strip("'")  # Remove any surrounding quotes
        user_file_path = user_file_path.replace("\\", "/")  # Normalize path for cross-platform compatibility
    except Exception as e:
        print_if(f"Error processing file path: {e}", SPRINT)
        return

    try:
        data = parse_csv(user_file_path, headers=True, names=HEADER_NAMES, dtype=DTYPES, delimiter=",", quotechar='"')
    except Exception as e:
        print_if(f"Error loading dataset: {e}", SPRINT)
        return

    my_model = tf.keras.models.Sequential()  # Ensure TensorFlow is initialized
    
    # Ensure that the dataset has important data
    assert data[TXT_COL].notnull().all(), "Text column contains null values."
    assert data[CLASS_COL].notnull().all(), "Class column contains null values."
    
    print_if(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns", SPRINT)

    knn, knn_preds, knn_stats = KNN_model(data)
    mnb, mnb_preds, mnb_accuracy = multinomial_NB_model(data)
    cnn_model, cnn_preds, cnn_accuracy = CNN_model(data, user_file_path)
    print(knn_preds.shape, mnb_preds.shape, cnn_preds.shape)

    # concatenate the predictions from all models
    combined_predictions = pd.DataFrame({
        'KNN_Predictions': knn_preds,
        'MNB_Predictions': mnb_preds,
        'CNN_Predictions': (cnn_preds > 0.5).astype(int)  # Convert probabilities to binary predictions
    })
    predictions_mean = (combined_predictions.mean(axis=1) > 0.5).astype(int)  # Average predictions and round to nearest integer
    
    test_labels = np.split(data[CLASS_COL].to_numpy(), [int(0.8 * len(data))])[1]
    combined_predictions['Combined_Predictions'] = predictions_mean
    combined_accuracy = np.mean(predictions_mean == test_labels)
    print_if(f"Combined model accuracy: {combined_accuracy:.2f}", SPRINT)
    


def CNN_model(data, user_file_path: str):
    """Train and evaluate a Convolutional Neural Network (CNN) model for phishing email detection."""
    # This function can be implemented similarly to KNN_model, using a CNN architecture.
    # This function can be implemented similarly to KNN_model, using a Naive Bayes classifier.
    data_x = data[TXT_COL].values
    data_y = data[CLASS_COL].values

    # Split the data into training and testing sets (80% training, 20% testing)
    training_data, testing_data = np.split(data_x, [int(0.8 * len(data_x))])
    training_labels, testing_labels = np.split(data_y, [int(0.8 * len(data_y))])

    # vectorize the text data using CountVectorizer
    vectorizer = CountVectorizer(max_features=5000, lowercase=True, stop_words='english')
    training_vectors = vectorizer.fit_transform(training_data).toarray()
    testing_vectors = vectorizer.transform(testing_data).toarray()

    # Reshape the data for CNN input (samples, time steps, features)
    training_vectors = training_vectors[..., np.newaxis]
    testing_vectors = testing_vectors[..., np.newaxis]

    # Attempt to load the model if it exists, otherwise create a new one
    model_path = "Models/" + user_file_path.split('/', maxsplit = 1)[-1] + "/cnn_model.h5" if user_file_path else 'models/cnn_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print_if("Loaded existing CNN model.", SPRINT)

    else:
        print_if("Creating new CNN model.", SPRINT)
        model = Sequential([
            Embedding(input_dim=training_vectors.shape[1], output_dim=128),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(training_vectors, training_labels, epochs=3, validation_data=(testing_vectors, testing_labels), batch_size=64)


    # Predict using the trained model

    predictions = model.predict(testing_vectors)
    predictions = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
    accuracy = np.mean(predictions == testing_labels)

    if STATS == 1:
        print(f"CNN model accuracy: {accuracy:.2f}")
        print(f"Predictions: {predictions}")

    # save the model
    model.save(model_path)
    print_if(f"CNN model saved as '{model_path}'", SPRINT)

    return model, predictions, accuracy

def multinomial_NB_model(data):
    """Train and evaluate a Multinomial Naive Bayes model for phishing email detection."""
    # This function can be implemented similarly to KNN_model, using a Naive Bayes classifier.
    data_x = data[TXT_COL].values
    data_y = data[CLASS_COL].values

    # Split the data into training and testing sets (80% training, 20% testing)
    training_data, testing_data = np.split(data_x, [int(0.8 * len(data_x))])
    training_labels, testing_labels = np.split(data_y, [int(0.8 * len(data_y))])

    # vectorize the text data using CountVectorizer
    vectorizer = CountVectorizer()
    training_vectors = vectorizer.fit_transform(training_data)
    testing_vectors = vectorizer.transform(testing_data) 

    # Train the Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(training_vectors, training_labels)

    # Predict using the trained model
    predictions = model.predict(testing_vectors)
    accuracy = model.score(testing_vectors, testing_labels)

    if STATS == 1:
        print(f"Multinomial Naive Bayes model accuracy: {accuracy:.2f}")
        print(f"Predictions: {predictions}")
    
    return model, predictions, accuracy

    








# KNN_model function to train and evaluate the KNN model for phishing email detection
def KNN_model(data):
    print_if("debugging", SPRINT)

    # Ensure that the dataset has important data
    assert data[TXT_COL].notnull().all(), "Text column contains null values."
    assert data[CLASS_COL].notnull().all(), "Class column contains null values."

    # Tokenize the text and count word occurrences
    data['Count'] = data[TXT_COL].apply(lambda x: Counter(simple_tokenizer(x)))
                
    # remove words with less than 2 occurrences from the counts
    data['Count'] = data['Count'].apply(lambda x: {word: cnt for word, cnt in x.items() if cnt >= 2})
    
    total_counts = Counter()
    for count in data['Count']:
        total_counts.update(count)
    
    # Sort the vocabulary and create mappings
    vocab = sorted(total_counts.keys())
    vocab_size = len(vocab)
    print_if(f"Vocabulary size: {vocab_size}", SPRINT)

    # Split data into training and testing sets
    train_data, test_data = np.split(data['Count'].to_numpy(), [int(0.8 * len(data))])  
    train_labels, test_labels = np.split(data[CLASS_COL].to_numpy(), [int(0.8 * len(data))])

    # Train the KNN model
    assert len(vocab) > 0, "Vocabulary is empty. Cannot train KNN model."
    train_data_KNN = vectorize_normalize_data(train_data, vocab)
    test_data_KNN = vectorize_normalize_data(test_data, vocab)

    neighbors = [3, 5, 10, 20, 40]
    max_accuracy = 0
    best_n_neighbors = 0
    for n_neighbors in neighbors:
        knn = train_knn_model(train_data_KNN, train_labels, n_neighbors=n_neighbors)
        knn_predictions = knn.predict(test_data_KNN)
        stats = knn_model_stats(knn, test_data_KNN, knn_predictions, test_labels)
        if stats['accuracy'] > max_accuracy:
            max_accuracy = stats['accuracy']
            best_n_neighbors = n_neighbors



    knn = train_knn_model(train_data_KNN, train_labels, n_neighbors=best_n_neighbors)
    print_if("KNN model trained.", SPRINT)

    # Predict using the trained KNN model
    knn_predictions = knn.predict(test_data_KNN)
    stats = knn_model_stats(knn, test_data_KNN, knn_predictions, test_labels)

    # print the precision, recall, and F1 score
    if STATS == 1:
        print(f"KNN predictions: {knn_predictions}")
        print(f"Accuracy: {stats['accuracy']}")
        print_if(f"Precision: {stats['precision']}")
        print_if(f"Recall: {stats['recall']}")
        print_if(f"F1 score: {stats['f1']}")
        print_if(f"AUC score: {stats['auc_score']}")
        print_if(f"Mean Squared Error: {stats['mse']}")

    return knn, knn_predictions, stats

if __name__ == "__main__":
    main()