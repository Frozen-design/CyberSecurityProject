import tensorflow as tf
#import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from Dependencies.sup import print_if, parse_csv
from Dependencies.bow import simple_tokenizer, sparse_vectorize, vectorize_data
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

SPRINT = 0
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
    
    # Ensure that the dataset has important data
    assert data[TXT_COL].notnull().all(), "Text column contains null values."
    assert data[CLASS_COL].notnull().all(), "Class column contains null values."
    
    print_if(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns", SPRINT)

    knn, knn_preds, knn_stats = KNN_model(data)
    mnb, mnb_preds, mnb_accuracy = multinomial_NB_model(data)
    print_if("KNN model execution completed.", SPRINT)


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
        print(f"Accuracy: {stats["accuracy"]}")
        print(f"Precision: {stats['precision']}")
        print(f"Recall: {stats['recall']}")
        print(f"F1 score: {stats['f1']}")
        print(f"AUC score: {stats['auc_score']}")
        print(f"Mean Squared Error: {stats['mse']}")

    return knn, knn_predictions, stats

if __name__ == "__main__":
    main()