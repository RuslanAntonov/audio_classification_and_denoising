import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM

X_train, \
Y_train, \
X_val, \
Y_val = np.load('X_train.npy'),\
        np.load('Y_train.npy'),\
        np.load('X_val.npy', allow_pickle=True),\
        np.load('Y_val.npy', allow_pickle=True)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)



print('Build model')
classifier = Sequential()
classifier.add(LSTM(units=128, input_shape=(16, 80), return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units=64, return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.Recall()])

print(classifier.summary())

print('Train')
cnnhistory = classifier.fit(X_train,Y_train, batch_size=32, epochs=5, validation_data=(X_val, Y_val))

classifier.save('classifier.h5')

result = classifier.evaluate(X_val, Y_val)
print("validation accuracy:", result[1])

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
