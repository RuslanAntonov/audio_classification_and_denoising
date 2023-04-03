import glob
import os
import numpy as np
import math


maxlen = 16
if os.path.isfile('X_train_den.npy') != True or os.path.isfile('Y_train_den.npy') != True:
    files = []
    labels = []
    for filepath in glob.iglob('train/*/*/*.npy'):

        file = np.load(filepath)
        class_name = os.path.split(os.path.realpath(os.path.join(filepath, "..\..")))[-1]

        if len(file) > maxlen:
            file = np.split(file, np.arange(maxlen, len(file), maxlen))
            for f in file:
                f = np.pad(f, [(0, maxlen - len(f)), (0, 0)], mode='constant')
                if class_name == 'noisy':
                    files.append(f)
                else:
                    labels.append(f)
        elif len(file) < maxlen:
            file = np.pad(file, [(0, maxlen - len(file)), (0, 0)], mode='constant')
            if class_name == 'noisy':
                files.append(file)
            else:
                labels.append(file)
        else:
            if class_name == 'noisy':
                files.append(file)
            else:
                labels.append(file)

    np.save('X_train_den', files)
    np.save('Y_train_den', labels)

    print('train dataset is ready')



if os.path.isfile('X_val_den.npy') != True or os.path.isfile('Y_val_den.npy') != True:
    files = []
    labels = []
    for filepath in glob.iglob('val/*/*/*.npy'):

        file = np.load(filepath)
        class_name = os.path.split(os.path.realpath(os.path.join(filepath, "..\..")))[-1]

        if len(file) > maxlen:
            file = np.split(file, np.arange(maxlen, len(file), maxlen))
            for f in file:
                f = np.pad(f, [(0, maxlen - len(f)), (0, 0)], mode='constant')
                if class_name == 'noisy':
                    files.append(f)
                else:
                    labels.append(f)
        elif len(file) < maxlen:
            file = np.pad(file, [(0, maxlen - len(file)), (0, 0)], mode='constant')
            if class_name == 'noisy':
                files.append(file)
            else:
                labels.append(file)
        else:
            if class_name == 'noisy':
                files.append(file)
            else:
                labels.append(file)

    np.save('X_val_den', files)
    np.save('Y_val_den', labels)

    print('val dataset is ready')



X_train, \
Y_train, \
X_val, \
Y_val = np.load('X_train_den.npy'),\
        np.load('Y_train_den.npy'),\
        np.load('X_val_den.npy'),\
        np.load('Y_val_den.npy')

print("X_train shape:", X_train.shape)
print("y_train shape:", Y_train.shape)
print("X_test shape:", X_val.shape)
print("y_test shape:", Y_val.shape)

print(len(X_train))