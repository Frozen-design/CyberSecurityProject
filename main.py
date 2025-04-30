from Dependencies.sup import print_if
from Dependencies.bow import update_bow, bag_of_words, simple_tokenizer
from collections import Counter
#import tensorflow as tf
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

'''
Phishing attacks are one of the most common cybersecurity threats. 
They use fake emails and text messages that look like they come from trusted sources and while email and phone providers use filters to block these messages, some still get through. 
One missed message can be enough to trick someone. Attackers often copy natural writing patterns, use new tactics, and change their messages to target specific people. 
These attacks can lead to data breaches and identity theft. Most filters use fixed rules or old patterns, which may cause them to block safe messages or miss harmful ones. 
We need a system that can keep up with new phishing methods and learn from past ones. 
We plan to train an AI model that checks both messages already blocked and those that slip through current filters. 
'''

SPRINT = 0

def main():
    print_if("debugging", SPRINT)
    bow = Counter()
    # Example usage of bag_of_words and update_bow functions
    data = pd.DataFrame({'text': ['Hello World!', 'This is a test.', 'Another test message.'], 'class': [1, 0, 1]})
    training_labels = data['class'].values
    data['tokens'] = data['text'].apply(lambda x: simple_tokenizer(x))


    update_bow(bow, "Hello World! This is a test.")
    update_bow(bow, "Another test message.")
    bow_size = len(bow)
    








    # Uncomment below to enable TensorFlow model training (if needed)
    """model = tf.keras.Sequential([
         tf.keras.layers.Dense(64, activation='relu', input_shape=(bow_size,)),
         tf.keras.layers.Dense(1, activation='sigmoid')
     ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError, metrics=['accuracy'])
    model.fit(training_data, training_labels, epochs=10, batch_size=32)"""

    # Placeholder for saving the model
    # model.save('path_to_save_model')

    # Placeholder for loading the model
    # loaded_model = tf.keras.models.load_model('path_to_save_model')

    # Placeholder for making predictions
    # predictions = loaded_model.predict(testing_data)
    # print(predictions)

if __name__ == "__main__":
    main()