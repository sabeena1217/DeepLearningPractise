# Convolutional neural networks excel at learning the spatial structure in input data.

# The IMDB review data does have a one-dimensional spatial structure in the sequence
# of words in reviews and the CNN may be able to pick out invariant features for
# good and bad sentiment. This learned spatial features may then be learned as sequences by an LSTM layer.

# We can easily add a one-dimensional CNN and max pooling layers after the Embedding layer
# which then feed the consolidated features to the LSTM.

# We can use a smallish set of 32 features with a small filter length of 3.

# The pooling layer can use the standard length of 2 to halve the feature map size.


import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
embedding_vecor_length = 32


model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs=3, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# We can see that we achieve similar results to the first example although with less weights and faster training time.

# I would expect that even better results could be achieved if this example was further extended to use dropout.

# OUTPUT
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 500, 32)           160000
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None, 500, 32)           3104
# _________________________________________________________________
# max_pooling1d_1 (MaxPooling1 (None, 250, 32)           0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               53200
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 216,405
# Trainable params: 216,405
# Non-trainable params: 0
# _________________________________________________________________
# None
# Epoch 1/3
# 25000/25000 [==============================] - 63s 3ms/step - loss: 0.4490 - acc: 0.7757
# Epoch 2/3
# 25000/25000 [==============================] - 99s 4ms/step - loss: 0.2460 - acc: 0.9038
# Epoch 3/3
# 25000/25000 [==============================] - 125s 5ms/step - loss: 0.1973 - acc: 0.9261
# Accuracy: 88.27%