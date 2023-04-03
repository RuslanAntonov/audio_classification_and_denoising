import math
import numpy as np

from keras.models import load_model

classifier = load_model('classifier.h5')

test_file = np.load('test_class.npy')

test_X = []
test_Y = []
maxlen = 16

if len(test_file) > maxlen:
    file = np.array_split(test_file, (math.ceil(len(test_file) / maxlen)))
    for f in file:
        f = np.pad(f, [(0, maxlen - len(f)), (0, 0)], mode='constant')
        test_X.append(f)
elif len(test_file) < maxlen:
    file = np.pad(test_file, [(0, maxlen - len(test_file)), (0, 0)], mode='constant')
    test_X.append(test_file)
else:
    test_X.append(test_file)


test_X = np.array(test_X)
print(test_X.shape)

res = round(np.mean(classifier.predict(test_X)))
print('Predicted label:', res)