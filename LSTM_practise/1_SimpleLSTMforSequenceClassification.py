# LSTM for sequence classification in the IMDB dataset

# Sequence classification is a predictive modeling problem where you have some sequence
# of inputs over space or time and the task is to predict a category for the sequence.

# The Large Movie Review Dataset (often referred to as the IMDB dataset) contains 25,000
# highly-polar movie reviews (good or bad) for training and the same amount again for testing.
# The problem is to determine whether a given movie review has a positive or negative sentiment.

# The words have been replaced by integers that indicate the ordered frequency of each word in the
# dataset. The sentences in each review are therefore comprised of a sequence of integers.

# We will map each movie review into a real vector domain, a popular technique when working
# with text called word embedding. This is a technique where words are encoded as real-valued
# vectors in a high dimensional space, where the similarity between words in terms of meaning
# translates to closeness in the vector space.

# We will map each word onto a 32 length real valued vector. We will also limit the total number
# of words that we are interested in modeling to the 5000 most frequent words, and zero out the rest.
# Finally, the sequence length (number of words) in each review varies, so we will constrain each
# review to be 500 words, truncating long reviews and pad the shorter reviews with zero values.

# It is good practice to grid search over each of these parameters and
# select for best performance and model robustness.

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# You can convert each character to an integer. Then each input will be a
# vector of integers. You can then use an Embedding layer to convert your
# vectors of integers to real-valued vectors in a projected space.

# Y is the output variables and Y_train are the output variables for the training dataset.
# For this dataset, the output values are movie sentiment values (positive or negative sentiment).

# when we take X_test as input, the output will be compared to y_test to compute
# the accuracy (the predictions made by the model are compared to y_test.)

#  truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# create the model
embedding_vector_length = 32
model = Sequential()


# The first layer is the Embedded layer that uses 32 length vectors to represent each word.
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))


# The next layer is the LSTM layer with 100 memory units (smart neurons).
model.add(LSTM(100))


# Finally, because this is a classification problem we use a Dense output layer
# with a single neuron and a sigmoid activation function to make 0 or 1 predictions
# for the two classes (good and bad) in the problem.
model.add(Dense(1, activation='sigmoid'))


# Because it is a binary classification problem, log loss is used as the loss function
# (binary_crossentropy in Keras). The efficient ADAM optimization algorithm is used
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print(model.summary())


# The model is fit for only 2 epochs because it quickly overfits the problem.
# A large batch size of 64 reviews is used to space out weight updates.
model.fit(X_train, y_train, epochs=3, batch_size=64)


# Once fit, we estimate the performance of the model on unseen reviews.
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))





# OUTPUT
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 500, 32)           160000
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               53200
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 213,301
# Trainable params: 213,301
# Non-trainable params: 0
# _________________________________________________________________
# None
# Epoch 1/3
# 25000/25000 [==============================] - 193s 8ms/step - loss: 0.4538 - acc: 0.7856
# Epoch 2/3
# 25000/25000 [==============================] - 304s 12ms/step - loss: 0.2969 - acc: 0.8814
# Epoch 3/3
# 25000/25000 [==============================] - 104s 4ms/step - loss: 0.2996 - acc: 0.8851
# Accuracy: 87.50%