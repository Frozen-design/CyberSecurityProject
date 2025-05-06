import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.utils import pad_sequences

import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pickle
import os

from Dependencies.sup import print_if, parse_csv
from Dependencies.bow import simple_tokenizer

from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def model_stats(predictions, test_labels):
    """
    Calculate and return various evaluation metrics for the model.
    """

    accuracy = np.mean(predictions == test_labels)
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')
    mse = np.mean((predictions - test_labels) ** 2)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "mse": mse}

class ModelContainer():
    def __init__(self, user_file_path: str, SPRINT: bool = True, STATS: int = 1, TXT_COL: str = 'Text', CLASS_COL: str = 'Class'):
        
        self.user_file_path = user_file_path
        self.SPRINT = SPRINT
        self.STATS = STATS
        self.TXT_COL = TXT_COL
        self.CLASS_COL = CLASS_COL
        self.data = self.load_data()

    def load_data(self):
        """Load the dataset from the specified file path."""
        if self.user_file_path.endswith('.csv'):
            csv_data = parse_csv(self.user_file_path, 
                                 headers=True, 
                                 names=[self.TXT_COL, self.CLASS_COL], 
                                 dtype={self.TXT_COL: str, self.CLASS_COL: int}, 
                                 delimiter=',', 
                                 quotechar='"')
            if self.TXT_COL not in csv_data.columns or self.CLASS_COL not in csv_data.columns:
                raise ValueError(f"CSV file must contain '{self.TXT_COL}' and '{self.CLASS_COL}' columns.")
            return csv_data
        else:
            raise ValueError("Unsupported file format. Please provide a CSV file.")
        
    def get_xy_data(self, test_size: float = 0.2):
        """Split the dataset into training and testing sets."""
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        
        data = self.data
        train_size = int((1 - test_size) * len(data))
        training_data = data.iloc[:train_size]
        testing_data = data.iloc[train_size:]
        
        assert len(training_data) + len(testing_data) == len(data), "Training and testing data sizes do not match original data size."
        
        return training_data, testing_data
        
    def create_model_h5(self, model_name: str, model):
        """Save the trained model to an HDF5 file."""
        model_dir = os.path.join(os.getcwd(), "Models", os.path.splitext(os.path.basename(self.user_file_path))[0])
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(model_path)
        print_if(f"Model saved to {model_path}", self.SPRINT)
    
    def create_model_pkl(self, model_name: str, model):
        """Save the trained model to a pickle file."""
        model_dir = os.path.join(os.getcwd(), "Models", os.path.splitext(os.path.basename(self.user_file_path))[0])
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print_if(f"Model saved to {model_path}", self.SPRINT)

    def load_model_h5(self, model_name: str):
        """Load a model from an HDF5 file."""
        model_dir = os.path.join(os.getcwd(), "Models", os.path.splitext(os.path.basename(self.user_file_path))[0])
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print_if(f"Model loaded from {model_path}", self.SPRINT)
            return model
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    def load_model_pkl(self, model_name: str):
        """Load a model from a pickle file."""
        model_dir = os.path.join(os.getcwd(), "Models", os.path.splitext(os.path.basename(self.user_file_path))[0])
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print_if(f"Model loaded from {model_path}", self.SPRINT)
            return model
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        
    def try_load_model(self, model_name: str, is_cnn: bool = False):
        """Attempt to load a model by its name. If the model does not exist, return None and print a message.
        If is_cnn is True, it will try to load a CNN model; otherwise, it will load a standard model.
        """
        try:
            if is_cnn:
                # For CNN models, we need to ensure the model is loaded correctly
                return self.load_model_h5(model_name)
            else:
                # For other models, we can load them directly
                return self.load_model_pkl(model_name)
        except FileNotFoundError:
            print_if(f"Model {model_name} not found. Creating a new model.", self.SPRINT)
            return None
        
    def predict(self, model, data):
        """Make predictions using the provided model on the given data."""

        # Ensure data is an iterable (e.g., list of strings)
        if isinstance(data, str):
            data = [data]

        if isinstance(model, KNeighborsClassifier):
            
            vectorizer = self.prepare_data_KNN()[4]
            data_vectorized = vectorizer.transform(data)

            return model.predict(data_vectorized)
        
        elif isinstance(model, MultinomialNB):
            
            vectorizer = self.prepare_data_MBN()[4]  
            data_vectorized = vectorizer.transform(data)

            return model.predict(data_vectorized)
        
        elif isinstance(model, Sequential):
            vectorizer = self.prepare_data_CNN()[4]  # Get the tokenizer
            max_length = self.prepare_data_CNN()[5]
            sequences = vectorizer.texts_to_sequences(data)
            padded_sequences = pad_sequences(sequences, maxlen=max_length)
            data_predict = padded_sequences

            return model.predict(data_predict)
        
        else:
            raise ValueError("Unsupported model type for prediction.")
        
    def prepare_data_CNN(self):
        data = self.data

        # Tokenize the text and create sequences
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(data[self.TXT_COL])
        sequences = tokenizer.texts_to_sequences(data[self.TXT_COL])
        
        # Pad sequences to ensure uniform length
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        # Split the data into training and testing sets (80% training, 20% testing)
        training_data, testing_data = np.split(padded_sequences, [int(0.8 * len(padded_sequences))])
        training_labels, testing_labels = np.split(data[self.CLASS_COL].values, [int(0.8 * len(data[self.CLASS_COL]))])

        return training_data, training_labels, testing_data, testing_labels, tokenizer, max_length

    def create_CNN_model(self):
        """Train and evaluate a CNN model for phishing email detection."""

        # If the model already exists, load it
        model = self.try_load_model("CNN_Model", is_cnn=True)
        if model is not None:
            return model, None
        
        print_if("Creating and training CNN model...", self.SPRINT)

        # Load the data
        training_data, training_labels, testing_data, testing_labels, tokenizer, max_length = self.prepare_data_CNN()

        assert len(training_data) == len(training_labels), "Training data and labels must have the same length."
        assert len(testing_data) == len(testing_labels), "Testing data and labels must have the same length."

        # Define the CNN model architecture
        model = Sequential([
            Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
            Conv1D(128, 3, activation='relu', strides=2),
            GlobalMaxPooling1D(),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        model.fit(training_data, training_labels, epochs=5, batch_size=128, validation_data=(testing_data, testing_labels), callbacks=[early_stopping])

        # Save the model
        self.create_model_h5("CNN_Model", model)

        # Evaluate the model on the test set
        loss, accuracy, mse = model.evaluate(testing_data, testing_labels)

        if self.STATS == 1:
            print(f"CNN Model Accuracy: {accuracy:.2f}")

        return model, accuracy

    def prepare_data_MBN(self):
        """Prepare the data for Multinomial Naive Bayes model training and testing."""
        
        # Load the data
        data = self.data

        data_x = data[self.TXT_COL].values
        data_y = data[self.CLASS_COL].values

        # Split the data into training and testing sets (80% training, 20% testing)
        training_data, testing_data = np.split(data_x, [int(0.8 * len(data_x))])
        training_labels, testing_labels = np.split(data_y, [int(0.8 * len(data_y))])

        assert len(training_data) == len(training_labels), "Training data and labels must have the same length."
        assert len(testing_data) == len(testing_labels), "Testing data and labels must have the same length."

        # Vectorize the text data using CountVectorizer

        vectorizer = CountVectorizer(lowercase=True)

        training_vectors = vectorizer.fit_transform(training_data)
        testing_vectors = vectorizer.transform(testing_data)

        # return the vectors and labels along with the vectorizer for later use
        return training_vectors, training_labels, testing_vectors, testing_labels, vectorizer



    def create_Multinomial_NB_model(self):
        """Train and evaluate a Multinomial Naive Bayes model for phishing email detection."""
        
        # If the model already exists, load it
        model = self.try_load_model("Multinomial_NB_Model")
        if model is not None:
            return model, None
        
        print_if("Creating and training Multinomial Naive Bayes model...", self.SPRINT)

        # Load the data
        training_vectors, training_labels, testing_vectors, testing_labels, _ = self.prepare_data_MBN()

        # Train the Multinomial Naive Bayes model
        model = MultinomialNB()
        model.fit(training_vectors, training_labels)

        # Save the model
        self.create_model_pkl("Multinomial_NB_Model", model)

        # Predict using the trained model
        predictions = model.predict(testing_vectors)
        accuracy = np.mean(predictions == testing_labels)

        if self.STATS == 1:
            print(f"Multinomial Naive Bayes model accuracy: {accuracy:.2f}")

        return model, accuracy

    def prepare_data_KNN(self):
        """Prepare the data for KNN model training and testing."""
        
        # Load the data
        data = self.data

        data_x = data[self.TXT_COL].values
        data_y = data[self.CLASS_COL].values

        # Split the data into training and testing sets (80% training, 20% testing)
        training_data, testing_data = np.split(data_x, [int(0.8 * len(data_x))])
        training_labels, testing_labels = np.split(data_y, [int(0.8 * len(data_y))])

        assert len(training_data) == len(training_labels), "Training data and labels must have the same length."
        assert len(testing_data) == len(testing_labels), "Testing data and labels must have the same length."

        # Vectorize the text data using CountVectorizer
        vectorizer = TfidfVectorizer()
        training_vectors = vectorizer.fit_transform(training_data)
        testing_vectors = vectorizer.transform(testing_data)

        return training_vectors, training_labels, testing_vectors, testing_labels, vectorizer

    def create_KNN_model(self):
        """Train and evaluate a KNN model for phishing email detection."""

        # If the model already exists, load it
        model = self.try_load_model("KNN_Model")
        if model is not None:
            return model, None
        
        print_if("Creating and training KNN model...", self.SPRINT)
        
        # Load the data
        training_vectors, training_labels, testing_vectors, testing_labels, _ = self.prepare_data_KNN()

        neighbors = [3, 5, 10, 20, 40]
        max_accuracy = 0
        best_n_neighbors = 0
        best_knn_model = None
        for n_neighbors in neighbors:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1, metric='cosine')
            knn.fit(training_vectors, training_labels)
            
            # Evaluate the model on the test set
            predictions = knn.predict(testing_vectors)
            accuracy = np.mean(predictions == testing_labels)

            if self.STATS == 1:
                print(f"KNN Model Accuracy with {n_neighbors} neighbors: {accuracy:.2f}")

            if accuracy > max_accuracy:
                best_knn_model = knn
                max_accuracy = accuracy
                best_n_neighbors = n_neighbors

        # Save the model
        self.create_model_pkl("KNN_Model", best_knn_model)

        if self.STATS == 1:
            print(f"KNN Model Accuracy: {max_accuracy:.2f} with {best_n_neighbors} neighbors")

        return best_knn_model, max_accuracy

