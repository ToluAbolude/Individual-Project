import keras
import numpy as np
from parser import load_data

#Step 1 - Colllect Data
trainset_data = load_data("folder/file_location")
input_data = load_data("folder/file_location")

#Step 2 - Build Model
model = Sequential()
model.add(Convolution2D(32,3,3 input_shape = (img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics = ['accuracy'])

#Step 3 - Train Model
model.fit_generator(trainset_data,samples_per_epoch = 2048, nb_epoch =30,input_data = input_data, nb_val_samples = 832)
model.save_weights('models/simple_CNN.hs')

#Step 4 - Test Model
img = image.load_img('location of image to test',target_size = (224,224))
prediction = model.predict(img)
print prediction
