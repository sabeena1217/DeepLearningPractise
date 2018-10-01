# Recurrent Neural networks like LSTM generally have the problem of overfitting.

# Dropout can be applied between layers using the Dropout Keras layer. We can do this easily by adding
# new Dropout layers between the Embedding and LSTM layers and the LSTM and Dense output layers.

# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
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
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs=3, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# We can see dropout having the desired impact on training with a slightly slower
# trend in convergence and in this case a lower final accuracy. The model could probably
# use a few more epochs of training and may achieve a higher skill (try it an see).


# OUTPUT
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 500, 32)           160000
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 500, 32)           0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               53200
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 213,301
# Trainable params: 213,301
# Non-trainable params: 0
# _________________________________________________________________
# None
# Epoch 1/3
# 25000/25000 [==============================] - 210s 8ms/step - loss: 0.4980 - acc: 0.7470
# Epoch 2/3
# 25000/25000 [==============================] - 112s 4ms/step - loss: 0.3082 - acc: 0.8771
# Epoch 3/3
# 25000/25000 [==============================] - 111s 4ms/step - loss: 0.2501 - acc: 0.9021
# Accuracy: 87.43%