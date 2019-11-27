
# coding: utf-8

# In[1]:

import pickle, gzip
import numpy as np
import matplotlib.pyplot as plt

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f ,encoding='latin1')
f.close()



def transform_y_vectors(y_vec, new_dim=10):
    l = len(y_vec)
    new_y = np.zeros((l, new_dim))
    for i, y_val in enumerate(y_vec):
        new_y[i][y_val] = 1
    
    return new_y



from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

train = np.reshape(train_set[0], (train_set[0].shape[0], 28, 28,1))
valid = np.reshape(valid_set[0], (valid_set[0].shape[0], 28, 28,1))

train_set_label = transform_y_vectors(train_set[1])
valid_set_label = transform_y_vectors(valid_set[1])

model = Sequential()
model.add(Convolution2D(filters = 64,kernel_size =  (3, 3), input_shape=(28, 28,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(kernel_size = (3,3), filters = 64))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(output_dim=128))
model.add(Dropout(p= 0.5))
model.add(Dense(output_dim=10))

model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train, train_set_label,epochs=15, batch_size=32)

result = model.evaluate(valid, valid_set_label, batch_size=128)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("\nSaved model to disk")

print ("Loss on valid set:"  + str(result[0]) + " Accuracy on valid set: " + str(result[1]))

