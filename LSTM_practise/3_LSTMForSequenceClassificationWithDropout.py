# Alternately, dropout can be applied to the input and recurrent connections of
# the memory units with the LSTM precisely and separately.

# Keras provides this capability with parameters on the LSTM layer,
# the dropout for configuring the input dropout and
# recurrent_dropout for configuring the recurrent dropout.
# For example, we can modify the first example to add dropout to the input and recurrent connections as follows:

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs=3, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# We can see that the LSTM specific dropout has a more pronounced effect on the
# convergence of the network than the layer-wise dropout.

# Dropout is a powerful technique for combating overfitting in your LSTM models and
# it is a good idea to try both methods, but you may bet better results with the
# gate-specific dropout provided in Keras


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
# 25000/25000 [==============================] - 301s 12ms/step - loss: 0.5095 - acc: 0.7502
# Epoch 2/3
# 25000/25000 [==============================] - 253s 10ms/step - loss: 0.3661 - acc: 0.8451
# Epoch 3/3
# 25000/25000 [==============================] - 287s 11ms/step - loss: 0.3597 - acc: 0.8504
# Accuracy: 84.26%