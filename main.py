import tensorflow as tf

import pandas as pd
import numpy as np

from Dependencies.sup import print_if
from Dependencies.models import ModelContainer

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
    print_if("Starting model for phishing email detection...", SPRINT)
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

    model_container = ModelContainer(user_file_path, SPRINT, STATS, TXT_COL, CLASS_COL)
    training_data, testing_data = model_container.get_xy_data()
    testing_labels = testing_data[CLASS_COL]  # Extract testing labels
    testing_data = testing_data[TXT_COL]  # Remove labels from testing data

    knn, knn_accuracy = model_container.create_KNN_model()
    mnb, mnb_accuracy = model_container.create_Multinomial_NB_model()
    cnn, cnn_accuracy = model_container.create_CNN_model()

    knn_preds = model_container.predict(knn, testing_data)
    mnb_preds = model_container.predict(mnb, testing_data)
    cnn_preds = model_container.predict(cnn, testing_data)

    # concatenate the predictions from all models
    combined_predictions = pd.DataFrame({
        'KNN_Predictions': knn_preds,
        'MNB_Predictions': mnb_preds,
        'CNN_Predictions': cnn_preds
    })
    predictions_mean = (combined_predictions.mean(axis=1) > 0.5).astype(int).to_numpy().flatten()  # Average predictions and round to nearest integer
    knn_accuracy = np.mean(knn_preds == testing_labels)
    mnb_accuracy = np.mean(mnb_preds == testing_labels)
    cnn_accuracy = np.mean(cnn_preds == testing_labels)

    print_if(f"KNN model accuracy: {knn_accuracy:.2f}", SPRINT)
    print_if(f"Multinomial NB model accuracy: {mnb_accuracy:.2f}", SPRINT)
    print_if(f"CNN model accuracy: {cnn_accuracy:.2f}", SPRINT)


    combined_predictions['Combined_Predictions'] = predictions_mean
    combined_accuracy = np.mean(predictions_mean == testing_labels)
    print_if(f"Combined model accuracy: {combined_accuracy:.2f}", SPRINT)

if __name__ == "__main__":
    main()