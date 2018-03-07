import numpy as np
import math
from PIL import Image
import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization


#data_Asia = np.fromfile("../Data/Asia.raw", dtype = 'float32')
data_Africa = np.fromfile("../Data/Africa.raw", dtype = 'float32')
data_label = np.fromfile("../Data/Africa_label.raw", dtype = 'int8')
#data_Europe = np.fromfile("../Data/Europe.raw", dtype = 'float32')
#data_Taiwan = np.fromfile("../Data/Taiwan.raw", dtype = 'float32')

#data_Asia = np.reshape(data_Asia,(566,488,648,1))
data_Africa = np.reshape(data_Africa,(541,488,648))
data_label = np.reshape(data_label,(488,541,648))
#data_Europe = np.reshape(data_Europe,(464,488,648,1))
data_label = np.transpose(data_label,(1,0,2))

upper = np.zeros((488,648))
lower = np.zeros((488,648))
for i in range(488):
  for j in range(648):
    tmp = np.nonzero(data_label[:,i,j])[0]
    if len(tmp) == 0:
      tmp = pre_tmp
    upper[i,j] = min(tmp)
    lower[i,j] = max(tmp)
    pre_tmp = tmp

'''
data_label_new = data_label
upper_bound = np.zeros((648,),dtype = 'int8')
lower_bound = np.zeros((648,),dtype = 'int8')
upper_bound_next = np.zeros((648,),dtype = 'int8')
lower_bound_next = np.zeros((648,),dtype = 'int8')

for i in range(20):
  if(np.count_nonzero(data_label[i,:,:,0]) != 0):
    last_i = i
    [x,y] = np.nonzero(data_label[i,:,:,0])
    for j in range(648):
      tmp = x[y == j]
      upper_bound[j] = max(tmp)
      lower_bound[j] = min(tmp)
    if (i+20 < 488):
      next_i = i+20
    else:
      next_i = 487
    [x,y] = np.nonzero(data_label[next_i,:,:,0])
    for j in range(648):
      tmp = x[y == j]
      upper_bound_next[j] = max(tmp)
      lower_bound_next[j] = min(tmp)
  else:
    for j in range(648):
      upper = int((upper_bound[j] * (i-last_i) + upper_bound_next[j] * (next_i - i))/(next_i-i))
      lower = int((lower_bound[j] * (i-last_i) + lower_bound_next[j] * (next_i - i))/(next_i-i))
      data_label_new[i,lower:upper,j,:] = 255
tmp = data_label_new.flatten()
tmp.tofile("../Data/label_2_int.raw")
''' 

p_size = 90
w = int(math.floor(488/p_size))
h = int(math.floor(648/p_size))
x_train = np.zeros((0,p_size,p_size))
y_train = np.zeros((0))
up_length = 0
mid_length = 0
low_length = 0
for i in range(w):
  for j in range(h):
    up = int(np.amin(upper[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]))
    low = int(np.amax(lower[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]))
    l1 = up-90
    l2 = 541-low
    l3 = low-up
    x_data_train_epi = data_Africa[90:up,i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
    y_data_train_epi = np.ones((l1)) 

    x_data_train_der = data_Africa[low:,i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
    y_data_train_der = np.zeros((l2)) 
    
    x_data_train_tra = data_Africa[up:low,i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
    y_data_train_tra = np.ones((l3))*2
 
    up_length = up_length + l1
    low_length = low_length + l2
    mid_length = mid_length + l3
    x_train = np.append(x_train,x_data_train_epi,axis = 0)
    x_train = np.append(x_train,x_data_train_der,axis = 0)
    x_train = np.append(x_train,x_data_train_tra,axis = 0)
    y_train = np.append(y_train,y_data_train_epi,axis = 0)
    y_train = np.append(y_train,y_data_train_der,axis = 0)
    y_train = np.append(y_train,y_data_train_tra,axis = 0)

total = up_length + low_length + mid_length
x_train = np.reshape(x_train,(total,p_size,p_size,1))
y_train = np.reshape(y_train,(total,1))
print(x_train.shape)
print(y_train.shape)
print(y_train)


x_train_norm = x_train/np.amax(x_train)
#x_test_norm = x_test/np.amax(x_test)
y_train_cat = keras.utils.to_categorical(y_train)
print(y_train_cat)
#y_test_cat = keras.utils.to_categorical(y_test)


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(p_size, p_size, 1)))
#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train_norm, y_train_cat, batch_size=32, epochs=30, validation_split = 0.1)
model.save('../Data/Africa_model2.h5')
'''
scores = model.evaluate(x_test_norm, y_test_cat, batch_size=32)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))
'''
