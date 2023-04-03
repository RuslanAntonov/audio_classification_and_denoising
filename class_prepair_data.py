import glob
import os
import numpy as np
import math

maxlen = 16
if os.path.isfile('X_train.npy') != True or os.path.isfile('Y_train.npy') != True:
    files = []
    labels = []
    for filepath in glob.iglob('train/*/*/*.npy'):

        file = np.load(filepath)
        class_name = os.path.split(os.path.realpath(os.path.join(filepath, "..\..")))[-1]

        for i in range(math.ceil(len(file) / maxlen)):
            if class_name == 'clean':
                labels.append(0)
            else:
                labels.append(1)

        if len(file) > maxlen:
            file = np.split(file, np.arange(maxlen, len(file), maxlen))
            for f in file:
                f = np.pad(f, [(0, maxlen - len(f)), (0, 0)], mode='constant')
                files.append(f)
        elif len(file) < maxlen:
            file = np.pad(file, [(0, maxlen - len(file)), (0, 0)], mode='constant')
            files.append(file)
        else:
            files.append(file)

    labels = np.asarray(labels).astype('float16').reshape((-1, 1))
    np.save('X_train', files)
    np.save('Y_train', labels)

    print('train dataset is ready')



if os.path.isfile('X_val.npy') != True or os.path.isfile('Y_val.npy') != True:
    files = []
    labels = []
    for filepath in glob.iglob('val/*/*/*.npy'):

        file = np.load(filepath)
        class_name = os.path.split(os.path.realpath(os.path.join(filepath, "..\..")))[-1]

        for i in range(math.ceil(len(file) / maxlen)):
            if class_name == 'clean':
                labels.append(0)
            else:
                labels.append(1)

        if len(file) > maxlen:
            file = np.split(file, np.arange(maxlen, len(file), maxlen))
            for f in file:
                f = np.pad(f, [(0, maxlen - len(f)), (0, 0)], mode='constant')
                files.append(f)
        elif len(file) < maxlen:
            file = np.pad(file, [(0, maxlen - len(file)), (0, 0)], mode='constant')
            files.append(file)
        else:
            files.append(file)

    labels = np.asarray(labels).astype('float16').reshape((-1, 1))
    np.save('X_val', files)
    np.save('Y_val', labels)

    print('val dataset is ready')



X_train, \
Y_train, \
X_val, \
Y_val = np.load('X_train.npy'),\
        np.load('Y_train.npy'),\
        np.load('X_val.npy', allow_pickle=True),\
        np.load('Y_val.npy', allow_pickle=True)

print("X_train shape:", X_train.shape)
print("y_train shape:", Y_train.shape)
print("X_test shape:", X_val.shape)
print("y_test shape:", Y_val.shape)
