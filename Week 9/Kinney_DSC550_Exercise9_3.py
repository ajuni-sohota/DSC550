"""
# D. Kinney DSC 550 9.3 Exercise: Neural Network Classifiers
1. **Neural Network Classifier with Scikit**
Using the multi-label classifier dataset from earlier exercises
(categorized-comments.jsonl in the reddit folder), fit a neural network
classifier using scikit-learn. Use the code found in chapter 12 of the
Applied Text Analysis with Python book as a guideline. Report the accuracy,
precision, recall, F1-score, and confusion matrix.

2. **Neural Network Classifier with Keras**
Using the multi-label classifier dataset from earlier exercises
(categorized-comments.jsonl in the reddit folder), fit a neural network
classifier using Keras. Use the code found in chapter 12 of the Applied
Text Analysis with Python book as a guideline. Report the accuracy, precision,
recall, F1-score, and confusion matrix.

3. **Classifying Images**
In chapter 20 of the Machine Learning with Python Cookbook, implement the code
found in section 20.15 classify MSINT images using a convolutional neural
network. Report the accuracy of your results.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from transformer import TextNormalizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

from keras.wrappers.scikit_learn import KerasClassifier

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


def nnc_with_scikit():
    """
    1. Neural Network Classifier with Scikit
    """
    corpus = pd.read_json('data/categorized-comments.jsonl',
                          lines=True, encoding='utf8')
    classifier = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer()),
        ('ann', MLPClassifier(hidden_layer_sizes=[500,150], verbose = True))])

    cv=12
    X = corpus['txt']
    y = corpus['cat']
    scoring = 'f1'
    scores = cross_val_score(classifier, X, y, cv=cv, scoring=scoring)
    classifier.fit(X, y)
    return scores


def nnc_with_keras():
    """
    2. Neural Network Classifier with Keras
    """
N_FEATURES = 5000
N_CLASSES = 4 

    def build_network():
        """
        Create a function that returns a compiled neural network
        """
        nn = Sequential()
        nn.add( Dense( 500, activation ='relu', input_shape =( N_FEATURES,)))
        nn.add( Dense( 150, activation ='relu'))
        nn.add( Dense( N_CLASSES, activation ='softmax'))
        nn.compile(
            loss ='categorical_crossentropy',
            optimizer ='adam',
            metrics =['accuracy'] )
        return nn

    pipeline = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', TfidfVectorizer( max_features = N_FEATURES)),
        ('nn', KerasClassifier( build_fn = build_network,
                                epochs = 200,
                                batch_size = 128)) ])

    scores = cross_val_score( model, X, y, cv = cv, scoring =' accuracy', n_jobs =-1) 
    model.fit( X, y) 
    return scores


def classify_images():
    """
    3. Classifying Images
    """

    # Set that the color channel value will be first
    K.set_image_data_format("channels_first")

    # Set seed
    np.random.seed(0)

    # Set image information
    channels = 1
    height = 28
    width = 28

    # Load data and target from MNIST data
    (data_train, target_train), (data_test, target_test) = mnist.load_data()

    # Reshape training image data into features
    data_train = data_train.reshape(data_train.shape[0], channels, height, width)

    # Reshape test image data into features
    data_test = data_test.reshape(data_test.shape[0], channels, height, width)

    # Rescale pixel intensity to between 0 and 1
    features_train = data_train / 255
    features_test = data_test / 255

    # One-hot encode target
    target_train = np_utils.to_categorical(target_train)
    target_test = np_utils.to_categorical(target_test)
    number_of_classes = target_test.shape[1]

    # Start neural network
    network = Sequential()

    # Add convolutional layer with 64 filters, a 5x5 window, and ReLU activation function
    network.add(Conv2D(filters=64,
                       kernel_size=(5, 5),
                       input_shape=(channels, width, height),
                       activation='relu'))

    # Add max pooling layer with a 2x2 window
    network.add(MaxPooling2D(pool_size=(2, 2)))

    # Add dropout layer
    network.add(Dropout(0.5))

    # Add layer to flatten input
    network.add(Flatten())

    # # Add fully connected layer of 128 units with a ReLU activation function
    network.add(Dense(128, activation="relu"))

    # Add dropout layer
    network.add(Dropout(0.5))

    # Add fully connected layer with a softmax activation function
    network.add(Dense(number_of_classes, activation="softmax"))

    # Compile neural network
    network.compile(loss="categorical_crossentropy", # Cross-entropy
                    optimizer="rmsprop", # Root Mean Square Propagation
                    metrics=["accuracy"]) # Accuracy performance metric

    # Train neural network
    network.fit(features_train, # Features
                target_train, # Target
                epochs=2, # Number of epochs
                verbose=0, # Don't print description after each epoch

    batch_size=1000, # Number of observations per batch
                validation_data=(features_test, target_test)) # Data for evaluation

    score = network.evaluate(features_test, target_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return


if __name__ == '__main__':







