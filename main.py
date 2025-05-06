import tensorflow as tf

import pandas as pd
import numpy as np

from Dependencies.sup import print_if
from Dependencies.models import ModelContainer, model_stats

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

    message0 = "+ Welcome to this AI driven phishing email detector"
    menu = f"| Menu\n| 1. About this model\n| 2. Run prediction on new email\n| 3. Exit"
    messageList = [
    "This model combines the strengths of 2 machine learning algorithms and 1 deep learning algorithm to make a prediction",
    "It uses a K-nearest neighbors classifer, a Multinominal Naive Bayes classifier, and a Convolutional Neural Network",
    "The KNN model uses TF-IDF (term frequency-inverse document frequency) to normalize the data sent to it",
    "The Multinominal NB model doesn't normalize the data sent to it.",
    "It takes the data as a 2D vectors with values = term frequency",
    "The CNN model transforms the data into a list of numbers to run through its neural network",
    "This new model takes a weighted average of the predictions of the previous models",
    "and then makes a decision whether the input is spam or not."
    ]

    length0 = max(len(message) for message in messageList)
    print("=" * len(message0))
    print(message0)
    while(1):
        print("=" * len(message0))
        print(menu)
        print("=" * len(message0))
        try:
            choice = int(input("Enter your choice: ")[0])
            print("=" * len(message0))
        except Exception as e:
            print("Make a valid choice (1, 2, or 3)")
            print()
        match choice:
            case 1:
                print("About")
                print("=" * length0)
                for i in range(len(messageList)):
                    print(messageList[i])
                print("=" * length0)
                knn_preds = model_container.predict(knn, testing_data)
                mnb_preds = model_container.predict(mnb, testing_data)
                cnn_preds = model_container.predict(cnn, testing_data)
                cnn_preds = (cnn_preds > 0.5).astype(int).flatten()

                # concatenate the predictions from all models
                combined_predictions = pd.DataFrame({
                    'KNN_Predictions': knn_preds,
                    'MNB_Predictions': mnb_preds,
                    'CNN_Predictions': cnn_preds
                })
                predictions_mean = (combined_predictions.mean(axis=1) > 0.5).astype(int).to_numpy().flatten()  # Average predictions and round to nearest integer

                input("Enter anything to continue to extra stats of the models... ")
                print("=" * length0)

                stats_KNN = model_stats(knn_preds, testing_labels)
                stats_MNB = model_stats(mnb_preds, testing_labels)
                stats_CNN = model_stats(cnn_preds, testing_labels)
                stats_combined = model_stats(predictions_mean, testing_labels)

                print("KNN Model Stats:")
                for key, value in stats_KNN.items():
                    print(f"{key}: {value}")
                print("=" * length0)

                print("Multinomial NB Model Stats:")
                for key, value in stats_MNB.items():
                    print(f"{key}: {value}")
                print("=" * length0)

                print("CNN Model Stats:")
                for key, value in stats_CNN.items():
                    print(f"{key}: {value}")
                print("=" * length0)

                print("Combined Model Stats:")
                for key, value in stats_combined.items():
                    print(f"{key}: {value}")
                print("=" * length0)
                
            case 2:
                print("Input an email to run through this model:")
                predict_text = input("Type here: ")
                knn_pred = model_container.predict(knn, predict_text)[0]
                mnb_pred = model_container.predict(mnb, predict_text)[0]
                cnn_pred_0 = model_container.predict(cnn, predict_text)[0]
                cnn_pred_1 = int((cnn_pred_0 > 0.5).astype(int).flatten()[0])  # Ensure single element extraction
                cnn_confidence = abs(float(cnn_pred_0.flatten()[0]) - 0.5) * 2  # Extract single element for calculation
                cnn_confidence = cnn_confidence*100

                print(f"K.N.N. model prediction: {'Spam' if knn_pred == 1 else 'Not Spam'}")
                print(f"M.N.B. model prediction: {'Spam' if mnb_pred == 1 else 'Not Spam'}")
                print(f"C.N.N. model prediction: {'Spam' if cnn_pred_1 == 1 else 'Not Spam'}")
                print(f"C.N.N. model confidence: {cnn_confidence:.2f}%")
                pred = int((0.3 * float(knn_pred) + 0.3 * float(mnb_pred) + 0.4 * float(cnn_pred_1)) > 0.5)
                print(f"New model prediction: {'Spam' if pred == 1 else 'Not Spam'}")
                
            case 3:
                exit()
            case _:
                print("Make a valid choice.")
                continue
        print()

if __name__ == "__main__":
    main()