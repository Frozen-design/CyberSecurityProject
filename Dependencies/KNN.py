from Dependencies.sup import print_if, parse_csv
from Dependencies.bow import simple_tokenizer, sparse_vectorize, vectorize_data
from collections import Counter
#import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def normalize_data(data):
    """
    Normalize the data using L2 normalization.
    
    Parameters:
    data (array-like): Input data to be normalized.
    
    Returns:
    array-like: Normalized data.
    """
    return normalize(data, axis=1, norm='l2')

def vectorize_normalize_data(data, vocab, axis = 1, norm = 'l2'):
    """
    Vectorize and normalize the data using the provided vocabulary.
    
    Parameters:
    data (list of Counter): List of tokenized text data.
    vocab (set): Set of vocabulary words.
    axis (int): Axis along which to normalize the data. Default is 1 (rows).
    norm (str): Norm to use for normalization. Default is 'l2'.
    
    Returns:
    scipy.sparse.csr_matrix: Normalized sparse matrix representation of the data.
    """
    sparse_matrix = vectorize_data(data, vocab)
    normalized_matrix = normalize(sparse_matrix, axis=axis, norm=norm).tocsr()
    return normalized_matrix

def train_knn_model(train_data, train_labels, n_neighbors=5):
    """
    Train a KNN model using the provided training data and labels.
    
    Parameters:
    train_data (scipy.sparse.csr_matrix): Training data in sparse matrix format.
    train_labels (array-like): Labels corresponding to the training data.
    n_neighbors (int): Number of neighbors to use for KNN. Default is 5.
    
    Returns:
    KNeighborsClassifier: Trained KNN model.
    """
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn_model.fit(train_data, train_labels)
    return knn_model

def knn_model_stats(knn_model, test_data_KNN, knn_predictions, test_labels):
    """
    Calculate and return various evaluation metrics for the KNN model.
    """

    accuracy = np.mean(knn_predictions == test_labels)
    precision = precision_score(test_labels, knn_predictions, average='weighted')
    recall = recall_score(test_labels, knn_predictions, average='weighted')
    f1 = f1_score(test_labels, knn_predictions, average='weighted')
    auc_score = roc_auc_score(test_labels, knn_model.predict_proba(test_data_KNN)[:, 1])
    mse = np.mean((knn_predictions - test_labels) ** 2)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc_score": auc_score, "mse": mse}
