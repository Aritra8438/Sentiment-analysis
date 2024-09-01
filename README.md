# Sentiment Analysis Comparison

## Overview

This repository contains two Jupyter notebooks that explore different approaches to sentiment analysis on text data. The first notebook uses a Long Short-Term Memory (LSTM) model, a type of Recurrent Neural Network (RNN), while the second notebook employs a more traditional machine learning approach using logistic regression with TF-IDF (Term Frequency-Inverse Document Frequency).

### Notebooks

1. **Sentiment_analysis_LSTM.ipynb**
2. **Sentiment_analysis.ipynb**

## Approach 1: LSTM Model

### Summary
- **Model Type**: LSTM (Long Short-Term Memory)
- **Framework**: TensorFlow/Keras
- **Key Libraries**: TensorFlow, Keras, NLTK, Scikit-learn
- **Data Processing**: Tokenization, padding, and embedding of sequences
- **Key Features**:
  - Use of an embedding layer to convert words into dense vectors of fixed size.
  - Implementation of LSTM layers to capture long-term dependencies in the text.
  - Inclusion of dropout layers for regularization.
  - The model is trained and evaluated on a dataset split into training and testing sets.

### Advantages
- **Captures Sequential Information**: LSTM models are well-suited for capturing the sequential nature of text, making them effective for tasks like sentiment analysis where context matters.
- **Ability to Handle Long Texts**: Due to the sequential processing, LSTMs can handle longer text sequences more effectively than traditional models.

### Disadvantages
- **Computationally Intensive**: Training LSTM models requires more computational resources and time compared to traditional models like logistic regression.
- **Complexity**: LSTMs are more complex and require careful tuning of hyperparameters such as the number of LSTM units, learning rate, etc.

## Approach 2: Logistic Regression with TF-IDF

### Summary
- **Model Type**: Logistic Regression
- **Framework**: Scikit-learn
- **Key Libraries**: NLTK, Scikit-learn
- **Data Processing**: Tokenization, stopword removal, lemmatization, TF-IDF vectorization
- **Key Features**:
  - TF-IDF is used to convert the text data into a matrix of word features.
  - Logistic regression is applied to classify the sentiment based on these features.
  - The model is trained on a TF-IDF transformed feature set and evaluated using common metrics.

### Advantages
- **Simplicity and Speed**: Logistic regression is simple to implement and fast to train, making it suitable for quick prototyping.
- **Interpretability**: The model coefficients in logistic regression are easy to interpret, allowing insights into the importance of features.

### Disadvantages
- **Lacks Contextual Understanding**: Unlike LSTM, logistic regression with TF-IDF does not capture the sequential or contextual information of words in a sentence.
- **Limited to Fixed-Length Vectors**: TF-IDF represents the text as a fixed-length vector, which may not capture the nuances of the text as effectively as word embeddings in LSTM.

## Conclusion

- **LSTM**: Best suited for scenarios where the sequential nature of the text and the context are critical for sentiment analysis. It offers more depth and accuracy at the cost of increased computational resources and complexity.
- **Logistic Regression with TF-IDF**: An excellent choice for quick, interpretable, and computationally efficient sentiment analysis. However, it may not perform as well on more complex datasets where context and word order are important.
