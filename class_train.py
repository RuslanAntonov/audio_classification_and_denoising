import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

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
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_shape=(250, 80)))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

print('Train')
#Fitting the data to the training dataset
cnnhistory = classifier.fit(X_train,Y_train, batch_size=64, epochs=60, validation_data=(X_val, Y_val))

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