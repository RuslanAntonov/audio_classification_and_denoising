import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Model
from keras import Input
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.utils import plot_model

X_train, \
Y_train, \
X_val, \
Y_val = np.load('X_train_den.npy'),\
        np.load('Y_train_den.npy'),\
        np.load('X_val_den.npy', allow_pickle=True),\
        np.load('Y_val_den.npy', allow_pickle=True)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

print('Build model')

# Define Shapes
n_inputs=X_train.shape[2] # number of input neurons = the number of features X_train
# Input Layer
visible = Input(shape=(250,80), name='Input-Layer') # Specify input shape
# Encoder Layer
e = Dense(units=n_inputs, name='Encoder-Layer')(visible)
e = BatchNormalization(name='Encoder-Layer-Normalization')(e)
e = LeakyReLU(name='Encoder-Layer-Activation')(e)
# Middle Layer
middle = Dense(units=n_inputs, activation='linear',
               activity_regularizer=keras.regularizers.L1(0.0001),
               name='Middle-Hidden-Layer')(e)
# Decoder Layer
d = Dense(units=n_inputs, name='Decoder-Layer')(middle)
d = BatchNormalization(name='Decoder-Layer-Normalization')(d)
d = LeakyReLU(name='Decoder-Layer-Activation')(d)
# Output layer
output = Dense(units=80, activation='sigmoid', name='Output-Layer')(d)
# Define denoising autoencoder model
model = Model(inputs=visible, outputs=output, name='Denoising-Autoencoder-Model')
# Compile denoising autoencoder model
model.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.MeanSquaredError()])
# Print model summary
print(model.summary())
# Plot the denoising autoencoder model diagram
# plot_model(model, to_file='Denoising_Autoencoder.png', show_shapes=True, dpi=300)

print('Train')
history = model.fit(X_train, Y_train, epochs=15, batch_size=64, verbose=1, validation_data=(X_val, Y_val))

model.save('denoising.h5')

result = model.evaluate(X_val, Y_val)
print("validation accuracy:", result[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()